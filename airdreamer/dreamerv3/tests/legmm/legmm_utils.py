"""Shared helpers for the legmm tests."""
import math

import numpy as np

from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import scenes
from embodied.envs.legmm.env import LegMMEnv
from drone_signature import load_config


def env_cfg(**overrides):
  cfg = dict(load_config(['legmm']).env.legmm)
  cfg.update(overrides)
  return cfg


def make_env(split='train', seed=0, wrapped=False, **overrides):
  cfg = env_cfg(split=split, **overrides)
  cfg.pop('use_seed', None)
  if wrapped:
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'dreamerv3'))
    import main as dmain
    config = load_config(['legmm'], [f'--env.legmm.split={split}', f'--seed={seed}',
                                     '--logdir=/tmp/legmm_test'])
    return dmain.make_env(config, 0)
  return LegMMEnv('mm', seed=seed, **cfg)


def zero_action(reset=False, skill=0):
  return {'reset': reset, 'skill': np.int32(skill),
          'nav': np.zeros(I.NAV_DIM, np.float32),
          'pick': np.zeros(I.MANIP_DIM, np.float32),
          'place': np.zeros(I.MANIP_DIM, np.float32)}


def random_action(rng):
  act = zero_action(skill=int(rng.integers(3)))
  for k in ('nav', 'pick', 'place'):
    act[k] = rng.uniform(-1, 1, act[k].shape).astype(np.float32)
  return act


def wall_scene(dist, room=(8.0, 8.0)):
  """Empty room plus one tall box whose near face is `dist` in front of the
  origin along +x. Robot at the origin facing +x."""
  boxes = np.array([[dist, -2.0, 0.0, dist + 0.5, 2.0, 3.0]])
  obj = scenes.SceneObject('box', np.array([3.0, 3.0, 0.05]), np.array([0.03, 0.03, 0.05]), 0.0)
  zone = scenes.PlaceZone(np.array([-3.0, -3.0]), np.array([0.175, 0.175]), 0.0)
  return scenes.Scene('train', 0, 0, np.asarray(room), boxes, ('obstacle',), obj, zone,
                      np.array([0.0, 0.0, 0.0]), np.ones(9, bool))


def height_action(h):
  """Normalized action value for a body height command."""
  lo, hi = I.BASE_HEIGHT_RANGE
  return float(np.clip(2 * (h - lo) / (hi - lo) - 1, -1, 1))
