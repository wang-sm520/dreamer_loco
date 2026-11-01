import numpy as np
import pytest

import embodied


def _make_env(**kwargs):
  try:
    from embodied.envs.drone_multi import OmniDronesMulti
  except Exception as e:
    pytest.skip(str(e))
  return OmniDronesMulti(**kwargs)


def test_drone_env_lifecycle():
  env = _make_env(headless=True, max_episode_length=5)
  act = {'reset': True, 'action': np.zeros(env.act_space['action'].shape, np.float32)}
  obs = env.step(act)
  assert obs['is_first'] is True
  assert obs['is_last'] is False
  act['reset'] = False
  for _ in range(3):
    obs = env.step(act)
    assert obs['state'].dtype == np.float32
  env.close()

  env2 = _make_env(headless=True, max_episode_length=5)
  env2.close()


def test_drone_env_driver_integration():
  env = _make_env(headless=True, max_episode_length=5)
  agent = embodied.RandomAgent(
      {k: v for k, v in env.obs_space.items() if not k.startswith('log/')},
      {k: v for k, v in env.act_space.items() if k != 'reset'},
  )
  env.close()

  driver = embodied.Driver([lambda: _make_env(headless=True, max_episode_length=5)], parallel=False)
  driver.reset(agent.init_policy)
  steps = []
  driver.on_step(lambda tran, _: steps.append(tran))
  driver(agent.policy, steps=10)
  assert len(steps) == 10
  driver.close()
