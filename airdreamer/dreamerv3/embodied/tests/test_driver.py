from functools import partial as bind

import embodied
import numpy as np


class _TestSpace:

  def __init__(self, dtype, shape=()):
    self.dtype = np.dtype(dtype)
    self.shape = tuple(shape)


class _FakeSharedArena:

  def __init__(self):
    self._envs = []
    self.steps = 0

  def register(self, env):
    self._envs.append(env)

  def step(self, substeps, headless):
    del headless
    self.steps += int(substeps)
    for env in self._envs:
      env._advance()


class _PassiveResetEnv(embodied.Env):

  def __init__(self, arena, active_on_reset):
    self._arena = arena
    self._arena.register(self)
    self._headless = True
    self._substeps = 1
    self._active_on_reset = bool(active_on_reset)
    self._tick = 0
    self._done = True
    self._cached = None
    self._passive = False

  @property
  def obs_space(self):
    return {
        'value': _TestSpace(np.int32),
        'reward': _TestSpace(np.float32),
        'is_first': _TestSpace(bool),
        'is_last': _TestSpace(bool),
        'is_terminal': _TestSpace(bool),
    }

  @property
  def act_space(self):
    return {
        'reset': _TestSpace(bool),
        'action': _TestSpace(np.float32, (1,)),
    }

  def _make_obs(self, value, is_first):
    return {
        'value': np.asarray(int(value), np.int32),
        'reward': np.asarray(0.0, np.float32),
        'is_first': bool(is_first),
        'is_last': False,
        'is_terminal': False,
    }

  def _advance(self):
    self._tick += 1

  def _omni_pre_step(self, action):
    if bool(action['reset']) or self._done:
      self._tick = 0
      self._done = False
      self._passive = False
      self._cached = self._make_obs(0, True)
      return self._active_on_reset
    return True

  def _omni_passive_step(self):
    self._passive = True

  def _omni_post_step(self, global_step=False, own_needs_step=False):
    if self._cached is not None:
      if global_step and (not own_needs_step) and self._passive:
        obs = self._make_obs(self._tick, True)
      else:
        obs = self._cached
      self._cached = None
      self._passive = False
      return obs
    return self._make_obs(self._tick, False)

  def step(self, action):
    needs_step = self._omni_pre_step(action)
    if needs_step:
      self._arena.step(self._substeps, self._headless)
    return self._omni_post_step(global_step=needs_step, own_needs_step=needs_step)


class _DeferredHoldResetEnv(embodied.Env):

  def __init__(self, arena):
    self._arena = arena
    self._arena.register(self)
    self._headless = True
    self._substeps = 1
    self._tick = 0
    self._done = True
    self._cached = None
    self._hold_pending = False
    self._mode = 'idle'

  @property
  def obs_space(self):
    return {
        'value': _TestSpace(np.int32),
        'reward': _TestSpace(np.float32),
        'is_first': _TestSpace(bool),
        'is_last': _TestSpace(bool),
        'is_terminal': _TestSpace(bool),
    }

  @property
  def act_space(self):
    return {
        'reset': _TestSpace(bool),
        'action': _TestSpace(np.float32, (1,)),
    }

  def _make_obs(self, value, is_first):
    return {
        'value': np.asarray(int(value), np.int32),
        'reward': np.asarray(0.0, np.float32),
        'is_first': bool(is_first),
        'is_last': False,
        'is_terminal': False,
    }

  def _advance(self):
    self._tick += 1 if self._mode == 'hold' else 10
    self._mode = 'idle'

  def _omni_pre_step(self, action):
    if bool(action['reset']) or self._done:
      self._tick = 0
      self._done = False
      self._cached = self._make_obs(0, True)
      self._hold_pending = True
      self._mode = 'idle'
      return False
    if self._hold_pending:
      self._hold_pending = False
      self._mode = 'hold'
      return True
    self._mode = 'act'
    return True

  def _omni_post_step(self, global_step=False, own_needs_step=False):
    del global_step, own_needs_step
    if self._cached is not None:
      obs = self._cached
      self._cached = None
      return obs
    return self._make_obs(self._tick, False)

  def step(self, action):
    needs_step = self._omni_pre_step(action)
    if needs_step:
      self._arena.step(self._substeps, self._headless)
    return self._omni_post_step(global_step=needs_step, own_needs_step=needs_step)


class TestDriver:

  def test_episode_length(self):
    agent = self._make_agent()
    driver = embodied.Driver([self._make_env])
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=1)
    assert len(seq) == 11

  def test_first_step(self):
    agent = self._make_agent()
    driver = embodied.Driver([self._make_env])
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=2)
    for index in [0, 11]:
      assert seq[index]['is_first'].item() is True
      assert seq[index]['is_last'].item() is False
    for index in [1, 10, 12]:
      assert seq[index]['is_first'].item() is False

  def test_last_step(self):
    agent = self._make_agent()
    driver = embodied.Driver([self._make_env])
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=2)
    for index in [10, 21]:
      assert seq[index]['is_last'].item() is True
      assert seq[index]['is_first'].item() is False
    for index in [0, 1, 9, 11, 20]:
      assert seq[index]['is_last'].item() is False

  def test_env_reset(self):
    agent = self._make_agent()
    driver = embodied.Driver([bind(self._make_env, length=5)])
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    action = {'act_disc': np.ones(1, int), 'act_cont': np.zeros((1, 6), float)}
    policy = lambda carry, obs: (carry, action, {})
    driver(policy, episodes=2)
    assert len(seq) == 12
    seq = {k: np.array([seq[i][k] for i in range(len(seq))]) for k in seq[0]}
    assert (seq['is_first'] == [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all()
    assert (seq['is_last']  == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all()
    assert (seq['reset']    == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all()
    assert (seq['act_disc'] == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]).all()

  def test_agent_inputs(self):
    agent = self._make_agent()
    driver = embodied.Driver([self._make_env])
    driver.reset(agent.init_policy)
    inputs = []
    states = []
    def policy(carry, obs, mode='train'):
      inputs.append(obs)
      states.append(carry)
      _, act, _ = agent.policy(carry, obs, mode)
      return 'carry', act, {}
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(policy, episodes=2)
    assert len(seq) == 22
    assert states == ([()] + ['carry'] * 21)
    for index in [0, 11]:
      assert inputs[index]['is_first'].item() is True
    for index in [1, 10, 12, 21]:
      assert inputs[index]['is_first'].item() is False
    for index in [10, 21]:
      assert inputs[index]['is_last'].item() is True
    for index in [0, 1, 9, 11, 20]:
      assert inputs[index]['is_last'].item() is False

  def test_unexpected_reset(self):

    class UnexpectedReset(embodied.Wrapper):
      """Send is_first without preceeding is_last."""
      def __init__(self, env, when):
        super().__init__(env)
        self._when = when
        self._step = 0
      def step(self, action):
        if self._step == self._when:
          action = action.copy()
          action['reset'] = np.ones_like(action['reset'])
        self._step += 1
        return self.env.step(action)

    env = self._make_env(length=4)
    env = UnexpectedReset(env, when=3)
    agent = self._make_agent()
    driver = embodied.Driver([lambda: env])
    driver.reset(agent.init_policy)
    steps = []
    driver.on_step(lambda tran, _: steps.append(tran))
    driver(agent.policy, episodes=1)
    assert len(steps) == 8
    steps = {k: np.array([x[k] for x in steps]) for k in steps[0]}
    assert (steps['reset'] == [0, 0, 0, 0, 0, 0, 0, 1]).all()
    assert (steps['is_first'] == [1, 0, 0, 1, 0, 0, 0, 0]).all()
    assert (steps['is_last'] == [0, 0, 0, 0, 0, 0, 0, 1]).all()

  def test_shared_arena_passive_reset_refreshes_first_obs(self):
    arena = _FakeSharedArena()
    env_a = lambda: _PassiveResetEnv(arena, active_on_reset=False)
    env_b = lambda: _PassiveResetEnv(arena, active_on_reset=True)
    driver = embodied.Driver([env_a, env_b], parallel=False)
    driver.reset(lambda batch_size: None)
    seen = []
    driver.on_step(lambda tran, worker: seen.append((worker, int(tran['value']), bool(tran['is_first']))))
    policy = lambda carry, obs: (
        carry,
        {'action': np.zeros((2, 1), np.float32)},
        {},
    )
    driver(policy, steps=2)
    driver.close()

    first = {(worker, is_first): value for worker, value, is_first in seen[:2]}
    assert first[(0, True)] == 1
    assert first[(1, True)] == 0

  def test_shared_arena_deferred_hold_uses_next_tick(self):
    arena = _FakeSharedArena()
    envs = [lambda: _DeferredHoldResetEnv(arena), lambda: _DeferredHoldResetEnv(arena)]
    driver = embodied.Driver(envs, parallel=False)
    driver.reset(lambda batch_size: None)
    seen = []
    driver.on_step(lambda tran, worker: seen.append((worker, int(tran['value']), bool(tran['is_first']))))
    policy = lambda carry, obs: (
        carry,
        {'action': np.zeros((2, 1), np.float32)},
        {},
    )
    driver(policy, steps=2)
    driver.close()

    first_round = {(worker, is_first): value for worker, value, is_first in seen[:2]}
    second_round = {(worker, is_first): value for worker, value, is_first in seen[2:4]}
    assert first_round[(0, True)] == 0
    assert first_round[(1, True)] == 0
    assert second_round[(0, False)] == 1
    assert second_round[(1, False)] == 1

  def _make_env(self, length=10):
    from embodied.envs import dummy
    return dummy.Dummy('disc', length=length)

  def _make_agent(self):
    env = self._make_env()
    agent = embodied.RandomAgent(env.obs_space, env.act_space)
    env.close()
    return agent
