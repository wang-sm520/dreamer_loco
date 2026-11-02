"""Offline dataset: replay chunk round trip, action masks, labeler purity."""
import copy

import numpy as np
import pytest

import embodied
from embodied.envs.legmm import collect as collect_lib
from embodied.envs.legmm import expert as ex
from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import labeler as lb
from legmm_utils import make_env, random_action, zero_action


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
  out = tmp_path_factory.mktemp('legmm_data')
  info, stats = collect_lib.collect('train', hours=600 / 3600 / I.HL_HZ, workers=1, out=out,
                                    seed=3, chunksize=256)
  return out, info, stats


def test_offline_replay_roundtrip(dataset, tmp_path):
  out, info, stats = dataset
  assert stats['summary']['steps'] == 600
  # Replay.load() needs a replay that owns a directory (as in train_offline).
  replay = embodied.replay.Replay(length=8, capacity=None, directory=tmp_path / 'replay')
  replay.load(directory=out)
  assert len(replay) > 0
  batch = replay.sample(4)
  obs_space = I.make_obs_space()
  act_space = I.make_act_space()
  for k, space in {**obs_space, **act_space}.items():
    if k.startswith('log/') or k == 'reset':
      assert k not in batch
      continue
    assert batch[k].shape == (4, 8, *space.shape), (k, batch[k].shape)
    assert batch[k].dtype == space.dtype, (k, batch[k].dtype, space.dtype)
  assert batch['sample_id'].dtype == np.int64


def test_mask_roundtrip_replay(dataset):
  out, _, _ = dataset
  for f in sorted(out.glob('*.npz')):
    d = np.load(f)
    skill = d['skill']
    for i, key in ((0, 'nav'), (1, 'pick'), (2, 'place')):
      inactive = skill != i
      assert not np.any(d[key][inactive]), (f.name, key)
    assert np.all(np.abs(d['nav']) <= 1) and np.all(np.abs(d['pick']) <= 1)
    # Driver layout: the action stored with an is_last observation is zero.
    last = d['is_last']
    assert not np.any(d['nav'][last]) and not np.any(d['skill'][last])


def test_expert_actions_are_masked_and_executed():
  env = make_env('train', seed=11)
  env.step(zero_action(reset=True))
  e = ex.ScriptedExpert(ex.ExpertConfig(diversity=True))
  e.reset(env.scene, np.random.default_rng(0), env.backend)
  for _ in range(120):
    act = e.act()
    ref = I.mask_inactive_np(act)
    for k in ('nav', 'pick', 'place'):
      np.testing.assert_array_equal(act[k], ref[k])
      assert np.all(np.abs(act[k]) <= 1)
    obs = env.step({**act, 'reset': False})
    for k in ('nav', 'pick', 'place'):
      np.testing.assert_array_equal(env.executed[k], act[k])
    if obs['is_last']:
      break


def _episode_return(env, policy, seed):
  env.forced_seed = seed
  obs = env.step(zero_action(reset=True))
  ret = 0.0
  while not obs['is_last']:
    obs = env.step({**policy(), 'reset': False})
    ret += float(obs['reward'])
  return ret


def test_reward_expert_beats_random():
  env = make_env('train', seed=21)
  rng = np.random.default_rng(0)
  rnd_ret = []
  exp_ret = []
  for seed in (3, 4, 7, 9):
    env.forced_seed = seed
    obs = env.step(zero_action(reset=True))
    e = ex.ScriptedExpert(ex.ExpertConfig(diversity=False))
    e.reset(env.scene, np.random.default_rng(seed), env.backend)
    ret = 0.0
    while not obs['is_last']:
      obs = env.step({**e.act(), 'reset': False})
      ret += float(obs['reward'])
    exp_ret.append(ret)
    rnd_ret.append(_episode_return(env, lambda: random_action(rng), seed))
  assert np.mean(exp_ret) > np.mean(rnd_ret) + 10.0, (exp_ret, rnd_ret)


def test_labeler_leaves_state_untouched():
  env = make_env('train', seed=5)
  labeler = lb.AffordanceLabeler(seed=0)
  env.step(zero_action(reset=True))
  e = ex.ScriptedExpert(ex.ExpertConfig(diversity=False))
  e.reset(env.scene, np.random.default_rng(0), env.backend)
  labeled = 0
  for _ in range(200):
    obs = env.step({**e.act(), 'reset': False})
    if obs['is_last']:
      break
    before = copy.deepcopy(env.backend.state)
    rng_before = copy.deepcopy(env.backend.rng.bit_generator.state)
    aff, mask = labeler(env)
    after = env.backend.state
    for f in ('x', 'y', 'yaw', 'h', 'holding', 'n_failed', 'grip'):
      assert getattr(before, f) == getattr(after, f), f
    np.testing.assert_array_equal(before.ee_b, after.ee_b)
    np.testing.assert_array_equal(before.obj_center, after.obj_center)
    assert env.backend.rng.bit_generator.state == rng_before
    assert np.all((aff >= 0) & (aff <= 1)) and set(np.unique(mask)) <= {0.0, 1.0}
    assert np.all(aff[mask == 0] == 0)
    labeled += int(mask.any())
  assert labeled > 0


def test_label_qa_only_drops_masks(dataset, tmp_path):
  """A judge contradicting a label with confidence >= 0.8 removes the sample
  (mask -> 0) and never rewrites the label value."""
  import shutil
  import sys
  import pathlib
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tools'))
  import label_qa
  from embodied.envs.legmm import dataset_stats
  out, _, _ = dataset
  data = tmp_path / 'data'
  shutil.copytree(out, data)
  side, _ = dataset_stats.load_meta(data)
  rows = label_qa.select(side)
  assert rows, 'no samples selected for review'
  index = label_qa.ChunkIndex(data)
  judge = lambda packet: {'verdict': 'feasible' if packet['state']['aff_rollout'] < 0.5 else 'infeasible',
                          'confidence': 0.9, 'reason': 'test'}
  rejects = []
  for row, which in rows[:5]:
    packet = label_qa.make_packet(side, row, which, index, 'train', tmp_path / 'packets')
    verdict = judge(packet)
    assert label_qa.contradicts(verdict['verdict'], packet['state']['aff_rollout'])
    rejects.append((packet['state']['sample_id'], which))
  before = {f.name: dict(np.load(f)) for f in data.glob('*.npz')}
  label_qa.drop_masks(data, index, rejects)
  index2 = label_qa.ChunkIndex(data)
  for sid, which in rejects:
    f, i = index2.where[sid]
    d = np.load(f)
    assert d['label_aff_mask'][i, which] == 0
    np.testing.assert_array_equal(d['label_aff'], before[f.name]['label_aff'])
  # Low-confidence or 'uncertain' verdicts never reject.
  assert not label_qa.contradicts('uncertain', 0.0)
