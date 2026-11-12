import pathlib
import sys

import elements
import numpy as np
import pytest

from embodied.run.qlam import QLambda, load_or_save_training_checkpoint
from ag_utils import act, make_ag_env, env_at_stance
from embodied.envs.legmm import ag_scenes, ag_skill, ag_contract
from drone_signature import load_config

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tools'))
import legmm_ag_eval as ev


def populated():
  q = QLambda(window=10)
  for x in [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]:
    q.update(x)
  assert q.lam == pytest.approx(2 / 3)
  return q


def checkpoint(path):
  cp = elements.Checkpoint(str(path))
  # Same registered legacy keys as train.py; real elements save/load, no mocked I/O.
  cp.step, cp.agent, cp.replay = elements.Counter(123), elements.Counter(456), elements.Counter(789)
  return cp


def test_lambda_checkpoint_roundtrip_and_restart(tmp_path):
  q = populated()
  cp = checkpoint(tmp_path / 'ckpt')
  load_or_save_training_checkpoint(cp, q)
  restored = QLambda(window=10)
  second = checkpoint(tmp_path / 'ckpt')
  load_or_save_training_checkpoint(second, restored)
  assert restored.save() == q.save()
  assert int(second.step) == 123
  assert (cp.latest() / 'qlam.pkl').exists()
  for x in [0, 0, 1, 0, 0, 1, 1, 1, 1, 0]:
    q.update(x)
    restored.update(x)
    assert restored.save() == q.save()
  assert restored.lam < 2 / 3


def test_lambda_old_checkpoint_compatibility(tmp_path):
  cp = checkpoint(tmp_path / 'ckpt')
  cp.save()
  restored = QLambda(window=10)
  second = checkpoint(tmp_path / 'ckpt')
  second.step = elements.Counter(0)
  load_or_save_training_checkpoint(second, restored)
  assert int(second.step) == 123
  assert restored.lam == 1 and restored.episodes == 0 and list(restored.recent) == []
  second.save()
  assert (second.latest() / 'qlam.pkl').exists()


def test_non_qstar_checkpoint_path_unchanged(tmp_path):
  cp = checkpoint(tmp_path / 'ckpt')
  load_or_save_training_checkpoint(cp, None)
  assert not (cp.latest() / 'qlam.pkl').exists()
  cp.step = elements.Counter(0)
  load_or_save_training_checkpoint(cp, None)
  assert int(cp.step) == 123


@pytest.mark.parametrize('mode', ['fixed', 'none'])
def test_lambda_other_modes_roundtrip(mode):
  q = QLambda(mode=mode)
  q.update(1)
  r = QLambda(mode=mode)
  r.load(q.save())
  assert r.save() == q.save()


def test_schedule_mismatch_rejected():
  with pytest.raises(ValueError, match='schedule mismatch'):
    QLambda(window=20).load(populated().save())


def test_train_filter_uses_new_search(monkeypatch):
  original = ag_skill.stance_candidates
  calls = []
  def capture(*args, **kwargs):
    calls.append(kwargs.copy())
    yield from original(*args, **kwargs)
  monkeypatch.setattr(ag_skill, 'stance_candidates', capture)
  env = make_ag_env(split='train', seed=1, contract='true_state', scene_filter='solvable')
  env.step(act(reset=True))
  assert env.scene.meta['solvable'] is True
  assert env.skill_cfg['contract'] == 'true_state'
  assert env.scene_attempts > 0 and calls
  assert all(c['dist_max'] > .70 and max(c['rhos']) > .70 and c['margin'] == .01 for c in calls)
  stances = ag_skill.feasible_stances(env.backend, env.scene, env.skill_cfg, first=True, margin=.01)
  assert stances and ag_contract.report(env.backend, stances[0]['base'])['clearance_ok']


def test_contract_failure_logs_and_policy_spaces_unchanged(monkeypatch):
  env, _ = env_at_stance(rho=.65, contract='true_state')
  # A genuine post-settle contract failure: move beyond measured reach while keeping estimated gate valid.
  def settle(b):
    pose = ag_scenes.head_on_stance(env.scene, ag_contract.reach_limit(env.scene.obj.support_z) + .001)
    b.state.x, b.state.y, b.state.yaw = pose
    return 0, False
  monkeypatch.setattr(env, '_settle', settle)
  phi = env._phis
  obs = env.step(act(trigger=1))
  assert obs['is_last'] and obs['is_terminal']
  assert obs['log/contract_violation'] == 1 and obs['log/contract_dist'] == 1
  assert obs['log/contract_height'] == 0
  assert obs['rew_task'] == pytest.approx(-.01 - 5 - phi[0], abs=1e-5)
  assert obs['rew_qshape'] == pytest.approx(-phi[1], abs=1e-5)
  assert obs['log/invalid'] == 0
  policy_keys = lambda e: {k for k in e.obs_space if not k.startswith('log/')}
  assert policy_keys(env) == policy_keys(make_ag_env())
  assert not any('contract' in k for k in policy_keys(env))


def test_validation_split_and_seeds():
  a, extra = ev.parse_args(['--checkpoint', 'x', '--out', 'x', '--split', 'val', '--n-episodes', '300'])
  assert not extra and ev.scene_seeds(a) == list(range(10000, 10300))
  a, _ = ev.parse_args(['--checkpoint', 'x', '--out', 'x'])
  assert a.split == 'test' and ev.scene_seeds(a) == list(range(20000, 20700))
  sets = [set(range(*ag_scenes.AG_SPLITS[k])) for k in ('train', 'val', 'test')]
  assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
  a.seed_start = 9999
  with pytest.raises(ValueError):
    ev.scene_seeds(a)


def test_val_config_is_unfiltered_and_metadata_uses_contract(tmp_path, monkeypatch):
  a, extra = ev.parse_args(['--checkpoint', 'x', '--out', 'x', '--split', 'val', '--n-episodes', '1',
                            '--env.legmm_ag.contract', 'true_state'])
  cfg = ev.load_config(a, extra, tmp_path)
  assert cfg.env.legmm_ag.split == 'val' and cfg.env.legmm_ag.scene_filter == 'none'
  calls = []
  original = ag_skill.solvable_scene
  def capture(backend, scene, skill_cfg, **kw):
    calls.append((scene.split, scene.seed, skill_cfg['contract'], kw['margin']))
    return original(backend, scene, skill_cfg, **kw)
  monkeypatch.setattr(ag_skill, 'solvable_scene', capture)
  meta = ev.scene_metadata(cfg, [10000])
  assert calls == [('val', 10000, 'true_state', .01), ('val', 10000, 'true_state', 0.)]
  assert set(meta) == {10000} and meta[10000]['solvable'] == meta[10000]['solvable_margin']
  assert load_config(['legmm_ag']).env.legmm_ag.contract == 'legacy'



def test_contract_logging_counts_and_rate_reset():
  from embodied.run.train import AgContractMetrics
  metrics = AgContractMetrics()
  def tran(last=False, **counts):
    return dict(is_last=last, **{'log/contract_' + k: counts.get(k, 0)
                for k in ('violation', 'dist', 'bearing', 'height', 'clearance')})
  metrics.on_step(tran())
  metrics.on_step(tran(True, violation=1, dist=1, clearance=1))
  metrics.on_step(tran(True))
  metrics.on_step(tran(True, violation=1, clearance=1))
  assert metrics.result() == dict(episodes=3, violation_count=2, violation_rate=2/3,
                                  dist_count=1, bearing_count=0, height_count=0, clearance_count=2)
  assert metrics.result() == dict(episodes=0, violation_count=0, violation_rate=0,
                                  dist_count=0, bearing_count=0, height_count=0, clearance_count=0)
