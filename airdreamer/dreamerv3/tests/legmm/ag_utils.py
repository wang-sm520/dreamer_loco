"""Helpers for the legmm_ag tests."""
import functools
import math

import numpy as np

from embodied.envs.legmm import ag_env, ag_scenes, ag_skill as sk
from drone_signature import load_config


def ag_cfg(**overrides):
  cfg = dict(load_config(['legmm_ag']).env.legmm_ag)
  for k in ('use_seed', 'split', 'backend', 'spaces_only'):
    cfg.pop(k, None)
  cfg['labels'] = dict(cfg['labels'], rate=0.0)
  cfg.update(overrides)
  return cfg


def make_ag_env(split='test', seed=0, **overrides):
  return ag_env.LegMMAgEnv('ag', split=split, seed=seed, **ag_cfg(**overrides))


def act(nav=(0.0, 0.0, 0.0), trigger=0, reset=False):
  return {'reset': reset, 'nav': np.asarray(nav, np.float32), 'trigger': np.int32(trigger)}


@functools.lru_cache(maxsize=None)
def good_seed():
  """A test seed with an unblocked flat box whose narrow face looks straight at the nearest edge and a
  head-on stance (rho 0.55) that the skill planner accepts."""
  env = make_ag_env()
  for seed in range(20000, 20400):
    env.forced_seed = seed
    env.step(act(reset=True))
    m = env.scene.meta
    if m['shape'] != 'flat' or m['blocked'] or m['alpha_fold_deg'] < 80 or not m['solvable']:
      continue
    base = ag_scenes.head_on_stance(env.scene, 0.55)
    if sk.plan_skill(env.backend, env.skill_cfg, base=base, h=env.backend.state.h)[0] is not None:
      return seed
  raise RuntimeError('no good seed')


def place(env, base, speed=0.0):
  """Teleport the robot (body-frame forward speed `speed`) and refresh the cached potentials."""
  s = env.backend.state
  s.x, s.y, s.yaw = map(float, base)
  s.v = np.array([speed, 0.0, 0.0])
  env._phis = env._potentials()


def set_est(env, pos=None, offset=(0.0, 0.0, 0.0)):
  """Pin the world-frame target estimate the gate reads (default: the true object centre; `offset` shifts
  it), leaving the true position and the goal provider untouched."""
  base = env.backend.state.obj_center if pos is None else pos
  env._est = np.asarray(base, np.float64) + np.asarray(offset, np.float64)


def env_at_stance(rho=0.55, speed=0.0, exact=True, **overrides):
  """Env with the robot teleported to a head-on stance. exact=True pins the target estimate to the true
  position so that the gate boundaries are exact; exact=False keeps the noisy estimate of reset()."""
  env = make_ag_env(**overrides)
  env.forced_seed = good_seed()
  obs = env.step(act(reset=True))
  place(env, ag_scenes.head_on_stance(env.scene, rho), speed)
  if exact:
    set_est(env)
  return env, obs


def shoulder_dist(env):
  return env._gate()['dist']
