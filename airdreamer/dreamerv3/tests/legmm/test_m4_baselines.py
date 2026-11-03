import math
import pathlib
import sys

import numpy as np

from embodied.run.eval_legmm import BaselinePolicy
from embodied.envs.legmm import interfaces as I


def test_baseline_manipulation_starts_at_current_base_without_stance_search(monkeypatch):
  root = pathlib.Path(__file__).resolve().parents[2]
  sys.path.insert(0, str(root / 'dreamerv3'))
  from drone_signature import load_config
  import main as dmain
  config = load_config(['legmm', 'sizetest', 'legmm_debug'], [
      '--jax.platform=cpu', '--jax.policy_devices=0', '--jax.train_devices=0',
      '--jax.prealloc=False', '--env.legmm.split=val'])
  env = dmain.make_env(config, 0, split='val')
  env.forced_seed = 10000
  reset = {k: np.zeros(s.shape, s.dtype) for k, s in env.act_space.items()}
  reset['reset'] = np.bool_(True)
  obs = env.step(reset)
  policy = BaselinePolicy('b1', env)
  policy.reset(0)
  monkeypatch.setattr(policy.expert, '_staging',
                      lambda *a, **k: (_ for _ in ()).throw(AssertionError('stance search called')))
  obs = dict(obs)
  obs['goal'] = np.array(obs['goal'], copy=True)
  obs['goal'][0:3] = [1, 0, 0]
  obs['goal'][3] = 0.5
  base = np.array([env.backend.state.x, env.backend.state.y, env.backend.state.yaw])
  action = policy(obs)
  assert policy.ctrl is not None
  np.testing.assert_allclose(policy.ctrl.plan.base, base)
  assert int(action['skill']) == int(I.Skill.PICK)
  env.close()


class _NoEnv:
  pass


def _obs(distance, bearing, holding=False):
  goal = np.zeros(16, np.float32)
  start = 8 if holding else 0
  goal[start:start + 3] = [math.cos(bearing), math.sin(bearing), 0.0]
  goal[start + 3] = distance
  proprio = np.zeros(28, np.float32)
  if holding:
    proprio[27] = 0.8
  return {'goal': goal, 'proprio': proprio}


def test_holding_uses_observed_gripper_and_ignores_label_phase():
  policy = BaselinePolicy('b1', _NoEnv())
  obs = _obs(0.5, 0.0, holding=False)
  obs['label_phase'] = np.int32(I.Phase.HOLDING)
  assert policy._goal_block(obs)[1] is False
  obs = _obs(0.5, 0.0, holding=True)
  obs['label_phase'] = np.int32(I.Phase.TO_PICK)
  assert policy._goal_block(obs)[1] is True


def test_b1_switches_at_first_strict_distance_threshold():
  policy = BaselinePolicy('b1', _NoEnv())
  assert not policy.should_switch(_obs(0.75, 0.0))
  assert policy.should_switch(_obs(0.749, 0.0))


def test_b2_requires_configured_heading_threshold():
  policy = BaselinePolicy('b2', _NoEnv(), heading_deg=30.0)
  assert not policy.should_switch(_obs(0.5, math.radians(40)))
  assert policy.should_switch(_obs(0.5, math.radians(20)))
