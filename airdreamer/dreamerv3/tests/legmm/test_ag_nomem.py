"""Policy history, transition gradients, and parameter parity for legmm_ag_nomem."""
import math

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from drone_signature import load_config
from dreamerv3 import agent as agent_lib
from dreamerv3 import main
from embodied.jax import nets


@pytest.fixture(autouse=True)
def _allow_transfers():
  jax.config.update('jax_transfer_guard', 'allow')
  yield
  jax.config.update('jax_transfer_guard', 'allow')


def _make_agent(
    tmp_path, markov_only, *, full_size=False, float_depth=False, overrides=()):
  presets = ['legmm_ag']
  if markov_only:
    presets.append('legmm_ag_nomem')
  if not full_size:
    presets.append('sizetest')
  config = load_config(presets, [
      '--jax.platform=cpu', '--jax.prealloc=False',
      '--env.legmm_ag.spaces_only=True', f'--logdir={tmp_path}',
      '--jax.jit=True', '--batch_size=1', '--batch_length=4',
      '--report_length=4', '--replay_context=1', '--agent.imag_length=3',
      *(['--env.legmm_ag.obs_depth_dtype=float32'] if float_depth else []),
      *overrides,
  ])
  # Build the same Agent as training, without precompiling unused test paths.
  agent = main.make_agent(elements.Config(
      dict(config, jax=dict(config.jax, precompile=False))))
  jax.config.update('jax_transfer_guard', 'allow')
  return agent.model, agent.params


def _inputs(model, seed, length):
  rng = np.random.default_rng(seed)
  obs = {}
  for key, space in model.obs_space.items():
    shape = (1, length, *space.shape)
    if space.dtype == np.uint8:
      value = rng.integers(0, 256, shape, dtype=np.uint8)
    elif space.dtype == bool:
      value = np.zeros(shape, dtype=bool)
    elif key.startswith('depth_'):
      value = rng.uniform(0, 1, shape).astype(space.dtype)
    else:
      value = rng.normal(size=shape).astype(space.dtype)
    obs[key] = jnp.asarray(value)
  obs['is_first'] = obs['is_first'].at[:, 0].set(True)
  obs['label_feas'] = jnp.ones_like(obs['label_feas'])
  obs['label_feas_mask'] = jnp.ones_like(obs['label_feas_mask'])
  acts = dict(
      nav=jnp.asarray(rng.uniform(-1, 1, (1, length, 3)), jnp.float32),
      trigger=jnp.asarray(rng.integers(0, 2, (1, length)), jnp.int32))
  return obs, acts


def _initial(model, params):
  _, carry = nj.pure(model.init_policy)(params, 1, seed=0)
  return carry


def _at(tree, index):
  return jax.tree.map(lambda value: value[:, index], tree)


def _diff(left, right):
  return float(np.max(np.abs(np.asarray(left, np.float32) -
                             np.asarray(right, np.float32))))


def _zero(value):
  np.testing.assert_array_equal(value, np.zeros_like(value))


@pytest.mark.parametrize('markov_only', [True, False], ids=['markov', 'recurrent'])
def test_policy_history_invariance(tmp_path, monkeypatch, markov_only):
  # The tiny CPU model's near-zero initial output scale can obscure the
  # recurrent control. Use the same stronger policy output scale in both modes.
  model, params = _make_agent(
      tmp_path, markov_only, overrides=('--agent.policy.outscale=0.1',))
  policy = nj.pure(model.policy)
  histories = [_inputs(model, seed, 6) for seed in (17, 33)]
  carried = []
  for obs, acts in histories:
    carry = _initial(model, params)
    for t in range(6):
      _, (carry, _, _) = policy(params, carry, _at(obs, t), seed=t + 1)
      # Intervene on the executed action so both observation and action
      # histories differ, irrespective of the policy's initial bias.
      carry = (*carry[:3], _at(acts, t))
    carried.append(carry)
  assert _diff(carried[0][-1]['nav'], carried[1][-1]['nav']) > 1e-4

  features, distributions = [], []
  observe = type(model.dyn).observe
  sample = agent_lib.sample

  def record_features(dyn, *args, **kwargs):
    result = observe(dyn, *args, **kwargs)
    features.append(result[2])
    return result

  def record_distributions(outputs):
    distributions.append(dict(
        nav_mean=outputs['nav'].output.mean,
        nav_stddev=outputs['nav'].output.stddev,
        trigger_logits=outputs['trigger'].logits))
    return sample(outputs)

  monkeypatch.setattr(type(model.dyn), 'observe', record_features)
  monkeypatch.setattr(agent_lib, 'sample', record_distributions)
  obs, _ = _inputs(model, 91, 1)
  common = dict(_at(obs, 0), is_first=jnp.array([False]))
  actions = []
  for carry in carried:
    _, (carry, action, _) = policy(params, carry, common, seed=24)
    actions.append(action)
    np.testing.assert_array_equal(carry[1]['stoch'], features[-1]['stoch'])

  feat_diff = {key: _diff(features[0][key], features[1][key])
               for key in features[0]}
  pol_diff = {key: _diff(distributions[0][key], distributions[1][key])
              for key in distributions[0]}
  act_diff = {key: _diff(actions[0][key], actions[1][key]) for key in actions[0]}
  print(f'markov_only={markov_only}: feature={feat_diff}, '
        f'policy={pol_diff}, action={act_diff}')
  if markov_only:
    for feat in features:
      _zero(feat['deter'])
    assert all(value == 0.0 for value in feat_diff.values())
    assert all(value == 0.0 for value in pol_diff.values())
    assert all(value == 0.0 for value in act_diff.values())
  else:
    assert feat_diff['deter'] > 1e-4
    assert max(pol_diff.values()) > 1e-4


def test_policy_gradients_are_current_frame_only(tmp_path):
  # Depth pixels are continuous here through the supported float32 depth
  # encoder. The preceding test exercises the regular uint8 execution path.
  model, params = _make_agent(tmp_path, True, float_depth=True)
  carry = _initial(model, params)
  obs, acts = _inputs(model, 12, 4)
  inputs = {key: obs[key] for key in model.enc.obs_space}

  def policy_score(frames, nav):
    reset = obs['is_first']
    _, _, tokens = model.enc(carry[0], dict(obs, **frames), reset, False)
    _, _, feat = model.dyn.observe(
        carry[1], tokens, dict(acts, nav=nav), reset, False)
    policy = model.pol(model.feat2tensor(_at(feat, -1)), bdims=1)
    return policy['nav'].pred()[0, 0]

  pure_score = nj.pure(policy_score)
  grad_frames, grad_nav = jax.grad(
      lambda frames, nav: pure_score(params, frames, nav, seed=4)[1],
      argnums=(0, 1))(inputs, acts['nav'])
  for key, grad in grad_frames.items():
    _zero(grad[:, :-1])
    current = float(np.linalg.norm(np.asarray(grad[:, -1])))
    print(f'{key}: past_grad_norm=0.0, current_grad_norm={current}')
    assert current > 0.0 and math.isfinite(current)
  _zero(grad_nav)
  print(f'prevact/nav: grad_norm={float(np.linalg.norm(np.asarray(grad_nav)))}')


def test_transition_kl_and_action_responsive_imagination(tmp_path):
  model, params = _make_agent(tmp_path, True)
  carry = _initial(model, params)[:3]
  obs, acts = _inputs(model, 42, 4)
  agent_loss = nj.pure(model.loss)

  @jax.jit
  def kl_means(dyn_weights):
    weights = dict(params, **dyn_weights)
    _, (_, (carry_out, entries, out, _)) = agent_loss(
        weights, carry, obs, acts, True, seed=8)
    values = jnp.stack([out['losses'][key].mean() for key in ('dyn', 'rep')])
    return values, (carry_out[1], entries[1], out['repfeat'])

  dyn_weights = {key: value for key, value in params.items() if key.startswith('dyn/')}
  means, pullback, (new_carry, entries, posterior) = jax.vjp(
      kl_means, dyn_weights, has_aux=True)
  assert np.isfinite(means).all()
  dyn_grad, = pullback(jnp.array([1.0, 0.0]))
  rep_grad, = pullback(jnp.array([0.0, 1.0]))
  print(f'KL means: dyn={float(means[0])}, rep={float(means[1])}')

  def norm(grad, prefix):
    weight_grads = [np.asarray(value, np.float32)
                    for key, value in grad.items()
                    if key.startswith(prefix) and value.ndim >= 2]
    assert weight_grads, prefix
    return math.sqrt(sum(float(np.vdot(x, x)) for x in weight_grads))

  for prefix in ('dyn/dynin1/', 'dyn/dynin2/', 'dyn/dyngru/', 'dyn/priorlogit/'):
    dyn_norm, rep_norm = norm(dyn_grad, prefix), norm(rep_grad, prefix)
    print(f'{prefix}: dyn_grad_norm={dyn_norm}, rep_grad_norm={rep_norm}')
    assert dyn_norm > 0.0 and math.isfinite(dyn_norm)
    assert rep_norm > 0.0 and math.isfinite(rep_norm)
  rep_norm = norm(rep_grad, 'dyn/obslogit/')
  print(f'dyn/obslogit/: rep_grad_norm={rep_norm}')
  assert rep_norm > 0.0 and math.isfinite(rep_norm)

  _zero(new_carry['deter'])
  _zero(entries['deter'])
  _zero(posterior['deter'])
  start = _at(entries, -1)
  dirty_start = dict(start, deter=jnp.full_like(start['deter'], 2))
  actions = jax.tree.map(lambda value: value[:, :3], acts)
  changed = dict(actions, nav=actions['nav'].at[:, 0].set(0.9))
  imagine = nj.pure(model.dyn.imagine)
  _, (end, feat, used_actions) = imagine(params, start, actions, 3, False, seed=2)
  _, (_, different, _) = imagine(params, start, changed, 3, False, seed=2)
  _, (_, cleaned, _) = imagine(params, dirty_start, actions, 3, False, seed=2)
  _zero(end['deter'])
  for step_feats in (feat, different, cleaned):
    _zero(step_feats['deter'])
  for key in feat:
    np.testing.assert_array_equal(feat[key], cleaned[key])
  for key, expected in nets.cast(actions).items():
    np.testing.assert_array_equal(expected, used_actions[key])
  action_effect = _diff(feat['logit'][:, 0], different['logit'][:, 0])
  print(f'imagination first-step logit max_diff={action_effect}')
  assert action_effect > 1e-4

  def read_deter(state):
    return dict(
        nav=jnp.broadcast_to(state['deter'][:, :1], (1, 3)),
        trigger=jnp.zeros((1,), jnp.int32))

  _, (_, _, policy_actions) = imagine(
      params, dirty_start, read_deter, 3, False, seed=2)
  _zero(policy_actions['nav'])


def test_preset_delta_and_full_size_parameter_parity(tmp_path):
  for preset in ('drone_multi', 'drone_nav', 'legmm_ag'):
    assert load_config([preset]).agent.dyn.rssm.markov_only is False
  for sizes in ((), ('size1m', 'legmm_debug')):
    base = load_config(['legmm_ag', *sizes]).flat
    nomem = load_config(['legmm_ag', 'legmm_ag_nomem', *sizes]).flat
    assert base.keys() == nomem.keys()
    assert {key: nomem[key] for key in base if base[key] != nomem[key]} == {
        'agent.dyn.rssm.markov_only': True}

  def shapes(markov_only):
    model, params = _make_agent(tmp_path, markov_only, full_size=True)
    prefixes = tuple(module.path + '/' for module in model.modules)
    return {key: (value.shape, value.dtype) for key, value in params.items()
            if key.startswith(prefixes)}

  recurrent, markov = shapes(False), shapes(True)
  assert recurrent and recurrent == markov
  count = sum(math.prod(shape) for shape, _ in markov.values())
  print(f'full-size trainable parameters in both modes: {count}, '
        f'tensors={len(markov)}')
