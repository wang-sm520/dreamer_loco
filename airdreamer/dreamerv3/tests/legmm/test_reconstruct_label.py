"""A labeled state rebuilt from its side-table row must give the same label."""
import numpy as np


def _state_before_manip(seed):
  """Runs the expert (domain randomization on, arm latency 1) until it is about
  to switch to PICK, i.e. a state whose pick label is typically positive."""
  from embodied.envs.legmm import collect, expert
  from embodied.envs.legmm.env import LegMMEnv
  cfg = collect.load_env_cfg()
  cfg.pop('use_seed', None)
  cfg['split'] = 'train'
  cfg['domain_rand'] = dict(cfg['domain_rand'], enabled=True, arm_latency_steps=(1, 1))
  env = LegMMEnv('mm', seed=seed, **cfg)
  zero = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items() if k != 'reset'}
  env.step({'reset': True, **zero})
  ctrl = expert.ScriptedExpert(expert.ExpertConfig(diversity=False))
  ctrl.reset(env.scene, np.random.default_rng(seed), env.backend)
  for _ in range(450):
    act = ctrl.act()
    if ctrl.mode == 'manip':
      return env
    obs = env.step({**act, 'reset': False})
    if obs['is_last']:
      return None
  return None


def test_reconstructed_labeled_state_reproduces_production_rate():
  from embodied.envs.legmm import collect
  from embodied.envs.legmm.labeler import AffordanceLabeler
  from tools.legmm_judge_eval import reconstruct, side_row
  for seed in range(5):
    env = _state_before_manip(seed)
    if env is None:
      continue
    s, b = env.backend.state, env.backend
    side = {k: np.asarray([v]) for k, v in collect.state_side_fields(s, b).items()}
    side.update(sample_id=np.asarray([1]), seed=np.asarray([env.scene.seed]),
                label_aff=np.zeros((1, 2), np.float32), aff_geom=np.zeros((1, 2), np.float32))
    original = AffordanceLabeler(seed=19).rate(env, 0)
    if original <= 0:
      continue
    rebuilt = AffordanceLabeler(seed=19).rate(reconstruct('train', side_row(side, 0)), 0)
    assert rebuilt == original, (seed, original, rebuilt)
    return
  raise AssertionError('no seed reached a pre-manipulation state with a positive label')
