"""Agent-side switches of legmm_ag: label exclusion, feasibility head, lambda recombination, trigger bias,
smoothing skip, and action semantics across sampling, env, replay and imagination."""
import copy

import elements
import embodied
import jax
import jax.numpy as jnp
import numpy as np

from embodied.envs.legmm import interfaces as I
from embodied.jax import outs
from drone_signature import load_config, signature
from dreamerv3 import agent as agent_lib
from dreamerv3.agent import Agent, compose_reward, imag_loss
from ag_utils import act, good_seed, make_ag_env


import pytest


@pytest.fixture(autouse=True)
def _allow_transfers():
  # A jitted Agent installs a global 'disallow' transfer guard; later tests in this process need it off.
  jax.config.update('jax_transfer_guard', 'allow')
  yield
  jax.config.update('jax_transfer_guard', 'allow')


def make_ag_agent(overrides=(), presets=('legmm_ag', 'sizetest'), replay_context=None, jit=False):
  config = load_config(list(presets), ['--jax.platform=cpu', '--jax.policy_devices=0', '--jax.train_devices=0',
                                       f'--jax.jit={jit}', *overrides])
  config = config.update({'batch_size': 2, 'batch_length': 3})
  if replay_context is not None:
    config = config.update({'replay_context': replay_context})
  obs = {k: v for k, v in I.make_ag_obs_space().items() if not k.startswith('log/')}
  actspace = {k: v for k, v in I.make_ag_act_space().items() if k != 'reset'}
  cfg = elements.Config(
      **config.agent, logdir='/tmp/legmm-ag-agent-tests', seed=0, jax=dict(config.jax, precompile=False),
      batch_size=2, batch_length=3, replay_context=config.replay_context, report_length=config.report_length,
      replica=0, replicas=1)
  agent = Agent(obs, actspace, cfg)
  jax.config.update('jax_transfer_guard', 'allow')
  return agent


def random_batch(agent, seed=0, **fixed):
  rng = np.random.default_rng(seed)
  batch = {'seed': jnp.asarray(rng.integers(0, 2 ** 32, (2,), dtype=np.uint32))}
  length = agent.config.batch_length + agent.config.replay_context
  for key, space in agent.spaces.items():
    shape = (2, length, *space.shape)
    if space.dtype == np.bool_:
      batch[key] = rng.integers(0, 2, shape, dtype=np.uint8).astype(bool)
    elif space.discrete:
      batch[key] = rng.integers(space.low, space.high, shape, dtype=space.dtype)
    elif space.dtype == np.uint8:
      batch[key] = rng.integers(0, 256, shape, dtype=np.uint8)
    else:
      batch[key] = rng.normal(size=shape).astype(space.dtype)
  for k, v in fixed.items():
    batch[k] = np.broadcast_to(np.asarray(v, batch[k].dtype), batch[k].shape).copy()
  return batch


def train_metrics(agent, batch):
  carry = agent.init_train(2)
  agent.train(carry, dict(jax.tree.map(jax.device_put, batch)))
  m = agent._take_outs(agent.pending_mets)
  agent.pending_mets = None
  return m


def test_label_and_shaping_keys_excluded_from_encoder_and_decoder():
  agent = make_ag_agent()
  want = {'depth_head', 'depth_wrist', 'proprio', 'goal'}
  assert set(agent.model.enc.obs_space) == want and set(agent.model.dec.obs_space) == want
  assert set(I.AG_LABEL_KEYS) <= set(agent.obs_space)                # stored in replay, fed to the agent
  assert not set(I.AG_LABEL_KEYS) & set(agent.model.scales)          # no reconstruction loss for them
  assert 'qstar_lam' in agent.model.ext_space


def test_feasibility_head_is_masked_and_not_a_policy_input():
  agent = make_ag_agent()
  assert agent.model.feas is not None and agent.model.scales['feas'] == 1.0
  assert not any(k.startswith('feas/') for k in agent.policy_keys)   # not in the policy parameter set
  batch = random_batch(agent, 1, label_feas_mask=0.0, label_feas=1.0, qstar_lam=1.0)
  assert train_metrics(agent, batch)['loss/feas'] == 0.0             # nothing to learn without labels
  agent2 = make_ag_agent()
  b1 = random_batch(agent2, 1, label_feas_mask=1.0, label_feas=1.0, qstar_lam=1.0)
  m1 = train_metrics(agent2, b1)
  agent3 = make_ag_agent()
  b0 = dict(b1, label_feas=np.zeros_like(b1['label_feas']))
  m0 = train_metrics(agent3, b0)
  assert m1['loss/feas'] > 0 and abs(m1['loss/feas'] - m0['loss/feas']) > 1e-4   # reacts to the labels
  assert m1['feas_label_rate'] == 1.0
  # BCE of the logit at init is ~log 2 for either label, and the two labels' losses sum to ~2 log 2
  assert abs(m1['loss/feas'] + m0['loss/feas'] - 2 * np.log(2)) < 0.3
  # the label is masked per element: half the mask halves nothing but changes the denominator
  mask = np.zeros_like(b1['label_feas_mask'])
  mask[0, :] = 1.0
  half = train_metrics(make_ag_agent(), dict(b1, label_feas_mask=mask))
  assert abs(half['feas_label_rate'] - 0.5) < 1e-6


def test_feasibility_gradient_reaches_the_world_model():
  agent = make_ag_agent()
  keys = [k for k in agent.params if k.startswith('enc/') or k.startswith('dyn/')]
  before = {k: np.asarray(agent.params[k]).copy() for k in keys}
  # isolate the feas loss: no labels -> no feas gradient; labels -> feas gradient (compare params of two runs)
  b = random_batch(agent, 3, label_feas_mask=1.0, label_feas=1.0, qstar_lam=1.0)
  a1, a2 = make_ag_agent(('--agent.opt.warmup=0',)), make_ag_agent(('--agent.opt.warmup=0',))
  b_nolabel = dict(b, label_feas_mask=np.zeros_like(b['label_feas_mask']))
  for a, bb in ((a1, b), (a2, b_nolabel)):
    a.train(a.init_train(2), dict(jax.tree.map(jax.device_put, bb)))
  diff = max(float(np.abs(np.asarray(a1.params[k]) - np.asarray(a2.params[k])).max()) for k in keys)
  assert diff > 0                                                    # the feas loss changes encoder/dynamics


def test_reward_is_recombined_per_batch_from_task_and_qshape():
  agent = make_ag_agent(replay_context=0)
  rng = np.random.default_rng(0)
  base = random_batch(agent, 2)
  rt, rq = rng.normal(size=base['reward'].shape).astype(np.float32), rng.normal(size=base['reward'].shape).astype(np.float32)
  for lam in (0.0, 0.25, 1.0):
    data = dict(base, rew_task=rt, rew_qshape=rq, qstar_lam=np.full(rt.shape, lam, np.float32),
                reward=np.full(rt.shape, 99.0, np.float32))
    carry = (None, None, None, jax.tree.map(lambda x: x[:, 0], {k: base[k] for k in agent.act_space}))
    _, obs, _, _ = agent.model._apply_replay_context(carry, data)
    np.testing.assert_allclose(np.asarray(obs['reward']), rt + lam * rq, atol=1e-6)
  assert np.allclose(np.asarray(compose_reward(dict(rew_task=rt, rew_qshape=rq, qstar_lam=rt * 0 + 0.5))), rt + 0.5 * rq, atol=1e-6)


def test_qstar_off_leaves_reward_untouched_and_unknown_mode_is_rejected():
  mm = make_ag_agent(('--agent.qstar.mode=off',))
  assert 'qstar_lam' not in mm.model.ext_space
  base = random_batch(mm, 2)
  carry = (None, None, None, {k: base[k][:, 0] for k in mm.act_space})
  mm2 = make_ag_agent(('--agent.qstar.mode=off',), replay_context=0)
  b = random_batch(mm2, 2)
  _, obs, _, _ = mm2.model._apply_replay_context((None, None, None, {k: b[k][:, 0] for k in mm2.act_space}), b)
  np.testing.assert_array_equal(np.asarray(obs['reward']), b['reward'])
  try:
    make_ag_agent(('--agent.qstar.mode=bogus',))
    raise AssertionError('accepted')
  except ValueError:
    pass


def test_trigger_bias_initial_logits_and_default_off():
  off = make_ag_agent(('--agent.trigger_bias.enabled=False',))        # built first: jitted agents change global state
  np.testing.assert_allclose(np.asarray(off.params['pol/head/trigger/logits/bias']), [0.0, 0.0])
  agent = make_ag_agent(jit=True)
  np.testing.assert_allclose(np.asarray(agent.params['pol/head/trigger/logits/bias']), [0.0, -3.0])
  assert np.abs(np.asarray(agent.params['pol/head/nav/mean/bias'])).max() == 0.0
  # initial policy mostly continues (30 single-env policy calls)
  trig = []
  for i in range(30):
    carry = agent.init_policy(1)
    obs = {k: np.zeros((1, *v.shape), v.dtype) for k, v in agent.obs_space.items()}
    obs['is_first'][:] = True
    _, acts, _ = agent.policy(carry, obs)
    trig.append(int(acts['trigger'][0]))
  assert np.mean(trig) < 0.3, trig


def test_other_presets_unchanged_by_the_new_switches():
  for preset in ('drone_nav', 'drone_multi'):
    c = load_config([preset]).agent
    assert not c.feas_head.enabled and c.qstar.mode == 'off' and not c.trigger_bias.enabled
    assert not c.smooth_skip_discrete
  c = load_config(['legmm']).agent
  assert not c.feas_head.enabled and c.qstar.mode == 'off' and c.skill.enabled and c.switch_cost == 0.2
  ag = load_config(['legmm_ag']).agent
  assert not ag.skill.enabled and not ag.aff_head.enabled and not ag.phase_head.enabled
  assert ag.bc.scale == 0 and ag.bc.scale_end == 0 and ag.skill_prior.scale == 0 and ag.switch_cost == 0
  assert ag.feas_head.enabled and ag.smooth_skip_discrete and ag.trigger_bias.enabled
  agent = make_ag_agent()
  assert agent.model._smooth_skip == ('trigger',) and not agent.model._bc_enabled


class _Norm:
  def stats(self):
    return 0.0, 1.0
  def __call__(self, x, update):
    return 0.0, 1.0


class _Val:
  def __init__(self, v):
    self.v = v
  def pred(self):
    return self.v
  def loss(self, target):
    return (self.v - target) ** 2


def test_smoothing_skips_discrete_heads_only_when_asked():
  B, H = 2, 6
  mean = jnp.tile(jnp.linspace(0, 1, H)[None, :, None], (B, 1, 3)) * 0.5
  def run(logits, skip):
    policy = {'nav': outs.Agg(outs.Normal(mean, jnp.ones_like(mean)), 1, jnp.sum), 'trigger': outs.Categorical(logits)}
    actions = {'nav': mean, 'trigger': jnp.argmax(logits, -1)}
    rew = jnp.zeros((B, H))
    val = _Val(jnp.zeros((B, H)))
    los, _, _ = imag_loss(actions, rew, jnp.ones((B, H)), policy, val, val, _Norm(), _Norm(), _Norm(),
                          update=False, actsmooth=1.0, actent=0.0, smooth_skip=skip)
    return np.asarray(los['policy'])
  steady = jnp.tile(jnp.asarray([5.0, 0.0])[None, None], (B, H, 1))
  flip = steady.at[:, ::2].set(jnp.asarray([0.0, 5.0]))                # the argmax flips every step
  assert np.abs(run(flip, ()) - run(steady, ())).max() > 1e-3          # default: flips are penalised
  np.testing.assert_allclose(run(flip, ('trigger',)), run(steady, ('trigger',)), atol=1e-6)


def test_action_semantics_sampling_env_replay_imagination():
  agent = make_ag_agent(jit=True)
  env = make_ag_env()
  env_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  assert set(env_space) == set(agent.act_space) == {'nav', 'trigger'}
  for k, v in env_space.items():                                     # same dtype/shape/bounds everywhere
    s = agent.act_space[k]
    assert (s.dtype, s.shape) == (v.dtype, v.shape) and np.array_equal(s.low, v.low) and np.array_equal(s.high, v.high)
  assert set(agent.model.dyn.act_space) == {'nav', 'trigger'}        # the RSSM embeds the same two heads
  # sampling -> env: the executed action is exactly the sampled one
  env.forced_seed = good_seed()
  obs = env.step(act(reset=True))
  carry = agent.init_policy(1)
  pobs = {k: np.asarray(obs[k])[None] for k in agent.obs_space}
  carry, acts, _ = agent.policy(carry, pobs)
  assert acts['nav'].shape == (1, 3) and acts['trigger'].dtype == np.int32 and acts['trigger'][0] in (0, 1)
  # the sample is unbounded (bounded-normal head); embodied's ClipAction and the env clip it to [-1, 1]
  env.step({'reset': False, 'nav': acts['nav'][0], 'trigger': acts['trigger'][0]})
  np.testing.assert_array_equal(env.executed['nav'], np.clip(acts['nav'][0], -1, 1))
  assert int(env.executed['trigger']) == int(acts['trigger'][0])
  # replay: what the driver stores is what comes back out
  replay = embodied.replay.Replay(length=4, capacity=64, online=False)
  stored = []
  for t in range(4):
    a = {'nav': np.full(3, 0.1 * t - 0.15, np.float32), 'trigger': np.int32(t % 2)}
    tran = {**{k: obs[k] for k in obs if not k.startswith('log/')}, **a}
    tran['is_first'] = t == 0
    replay.add(tran)
    stored.append(a)
  batch = replay.sample(1, 'train')
  np.testing.assert_array_equal(batch['trigger'][0], [0, 1, 0, 1])
  np.testing.assert_allclose(batch['nav'][0], [s['nav'] for s in stored], atol=0)
  # imagination: policy outputs used there are over the same keys with the same dtypes (training loss logs them)
  m = train_metrics(agent, random_batch(agent, 5, qstar_lam=1.0))
  assert 'ent/nav' in m and 'ent/trigger' in m and 'rand/trigger' in m
  dyn_in = agent.model.dyn.act_space
  assert dyn_in['trigger'].discrete and dyn_in['nav'].shape == (3,)
