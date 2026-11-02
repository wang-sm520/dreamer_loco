import math

import numpy as np
import pytest

from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import kin_backend
from legmm_utils import height_action, make_env, random_action, zero_action


def test_mask_roundtrip_env():
  env = make_env()
  env.step(zero_action(reset=True))
  rng = np.random.default_rng(0)
  for skill in (0, 1, 2, 1, 0):
    act = random_action(rng)
    act['skill'] = np.int32(skill)
    env.step(act)
    ref = I.mask_inactive_np(act)
    for k in ('nav', 'pick', 'place'):
      np.testing.assert_array_equal(env.executed[k], ref[k])
      if k != I.SKILL_ACTION_KEY[I.Skill(skill)]:
        assert not env.executed[k].any()


@pytest.mark.parametrize('split', ['train', 'test_constrained', 'test_unseen_layout'])
def test_random_policy_does_not_crash(split):
  env = make_env(split, seed=1, wrapped=True)   # wrapped: CheckSpaces validates every obs
  rng = np.random.default_rng(1)
  obs = env.step({**zero_action(reset=True)})
  assert obs['is_first']
  episodes, last = 0, False
  for _ in range(1500):
    obs = env.step(random_action(rng))
    if last:
      assert obs['is_first']
    last = bool(obs['is_last'])
    episodes += last
    for k in ('proprio', 'goal'):
      assert np.isfinite(obs[k]).all()
    if obs['is_terminal']:
      assert I.TERM_REASONS[int(obs['log/term_reason'])] not in ('none', 'timeout', 'nan')
  assert episodes >= 1


def test_timeout_is_not_terminal():
  env = make_env(max_episode_steps=5)
  env.step(zero_action(reset=True))
  for i in range(5):
    obs = env.step(zero_action())
  assert obs['is_last'] and not obs['is_terminal']
  assert I.TERM_REASONS[int(obs['log/term_reason'])] == 'timeout'


def test_snapshot_restore_is_exact():
  env = make_env(seed=2)
  env.step(zero_action(reset=True))
  rng = np.random.default_rng(5)
  for _ in range(10):
    env.step(random_action(rng))
  snap = env.snapshot()
  acts = [random_action(rng) for _ in range(8)]
  first = [env.step(a) for a in acts]
  env.restore(snap)
  second = [env.step(a) for a in acts]
  for a, b in zip(first, second):
    for k in a:
      np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


def _teleport_to_grasp(env):
  """Stand 0.45 m from the object along a free direction and put the gripper
  on the object, approaching along that direction."""
  from embodied.envs.legmm import scenes
  b, sc = env.backend, env.scene
  s = b.state
  b._static = np.zeros((0, 6))      # no collisions in this unit test
  i = int(np.where(sc.approach[:scenes.N_DIRS])[0][0])
  d = sc.dirs[i]
  c = s.obj_center
  s.x, s.y = c[0] + 0.45 * d[0], c[1] + 0.45 * d[1]
  s.yaw = math.atan2(-d[1], -d[0])
  s.h = float(np.clip(c[2] - 0.05, *I.BASE_HEIGHT_RANGE))
  R_body = b.R_body()
  a_b = R_body.T @ (-d)
  y_b = np.cross([0, 0, 1.0], a_b)
  y_b /= np.linalg.norm(y_b)
  s.R_ee_b = np.stack([a_b, y_b, np.cross(a_b, y_b)], 1)
  s.ee_b = b.clip_ee(R_body.T @ (c - b.base_pos()), s)
  assert np.linalg.norm(b.ee_world()[0] - c) < 0.02, (b.ee_world()[0], c)
  return s


def _hold_posture(act, h):
  act['pick'][10] = height_action(h)
  return act


def test_grasp_reward_has_no_potential_jump(monkeypatch):
  monkeypatch.setattr(kin_backend, 'GRASP_P', 1.0)
  env = make_env(seed=4, domain_rand={'enabled': False})
  env.step(zero_action(reset=True))
  s = _teleport_to_grasp(env)
  env._prev_state = env.backend.step_state({})
  close = _hold_posture(zero_action(skill=1), s.h)
  close['pick'][6] = -1.0      # grip -> 0 (close)
  obs = env.step(close)
  assert env.backend.state.holding, 'grasp should succeed at the object'
  assert int(obs['label_phase']) == I.Phase.HOLDING
  cfg = env.reward_cfg
  # The phase changed on this step, so no potential difference is added even
  # though the goal jumped from the object to the place zone.
  assert float(obs['reward']) == pytest.approx(cfg.r_time + cfg.r_grasp, abs=1e-5)
  assert obs['proprio'][I.PROPRIO_LAYOUT['grip_effort']][0] > 0.5


def test_failed_grasp_far_from_object_is_not_an_attempt():
  env = make_env(seed=4, domain_rand={'enabled': False})
  env.step(zero_action(reset=True))
  act = _hold_posture(zero_action(skill=1), env.backend.state.h)
  act['pick'][6] = -1.0
  obs = env.step(act)
  assert obs['log/grasp_attempt'] == 0 and env.backend.state.n_failed == 0


def test_potential_shaping_rewards_progress():
  env = make_env(seed=7)
  env.step(zero_action(reset=True))
  b = env.backend
  s = b.state
  # Face the object and drive straight at it for a few steps.
  dx, dy = s.obj_center[0] - s.x, s.obj_center[1] - s.y
  s.yaw = math.atan2(dy, dx)
  env._prev_state = b.step_state({})
  fwd = zero_action(skill=0)
  fwd['nav'][0] = 0.5          # 0.4 m/s
  fwd['nav'][3] = height_action(env.backend.state.h)
  rewards = [float(env.step(fwd)['reward']) for _ in range(5)]
  assert sum(rewards) > 0
