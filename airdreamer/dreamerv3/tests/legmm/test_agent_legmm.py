import numpy as np
import elements
import jax

from embodied.envs.legmm import interfaces
from drone_signature import load_config
from dreamerv3.agent import Agent


def make_legmm_agent(rec_scales=None, skill_enabled=None,
                     rec_per_pixel_norm=True, train_phase=None, bc=None,
                     batch_length=3, jax_jit=False):
  config = load_config(['legmm', 'sizetest'], [
      '--jax.platform=cpu',
      '--jax.policy_devices=0', '--jax.train_devices=0', f'--jax.jit={jax_jit}'])
  config = config.update({'agent': {
      'rec_per_pixel_norm': rec_per_pixel_norm},
      'batch_size': 2, 'batch_length': batch_length})
  obs = {k: v for k, v in interfaces.make_obs_space().items()
         if not k.startswith('log/')}
  act = {k: v for k, v in interfaces.make_act_space().items() if k != 'reset'}
  if rec_scales is not None:
    config = config.update({'agent': {'rec_scales': rec_scales}})
  if skill_enabled is not None:
    config = config.update({'agent': {'skill': {'enabled': skill_enabled}}})
  if train_phase is not None:
    config = config.update({'agent': {'train_phase': train_phase}})
  if bc is not None:
    config = config.update({'agent': {'bc': bc}})
  cfg = elements.Config(
      **config.agent, logdir='/tmp/legmm-agent-tests', seed=config.seed,
      jax=dict(config.jax, precompile=False), batch_size=config.batch_size,
      batch_length=config.batch_length, replay_context=config.replay_context,
      report_length=config.report_length, replica=config.replica,
      replicas=config.replicas)
  return Agent(obs, act, cfg)


def _random_batch(agent, seed=0):
  rng = np.random.default_rng(seed)
  batch = {}
  batch['seed'] = jax.numpy.asarray(
      rng.integers(0, 2**32, (2,), dtype=np.uint32))
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
  return batch


def _reconstruction_losses(agent, batch):
  carry = agent.init_train(2)
  batch = jax.tree.map(jax.device_put, batch)
  agent.train(carry, dict(batch))
  # The first public train() returns before its async metrics are fetched.
  # Fetch that first loss call explicitly; do not perform a second optimizer update.
  metrics = agent._take_outs(agent.pending_mets)
  agent.pending_mets = None
  return {key: metrics[f'loss/{key}'] for key in
          ('depth_head', 'depth_wrist', 'proprio', 'goal')}


def test_label_exclusion():
  agent = make_legmm_agent()
  for space in (agent.model.enc.obs_space, agent.model.dec.obs_space):
    assert not set(interfaces.LABEL_KEYS) & set(space)
    assert {'depth_head', 'depth_wrist', 'proprio', 'goal'} <= set(space)


def test_rec_scales():
  agent = make_legmm_agent()
  assert agent.model.scales['proprio'] == 2.0
  assert agent.model.scales['goal'] == 2.0
  assert agent.model.scales['depth_head'] == 1.0
  default = make_legmm_agent('')
  for key in default.model.dec.obs_space:
    assert default.model.scales[key] == default.config.loss_scales.rec
  assert agent.config.rec_per_pixel_norm
  assert agent.model.dec.obs_space['depth_head'].shape[0:2] == (48, 80)


def test_rec_per_pixel_normalisation_uses_loss_path():
  normalized = make_legmm_agent(rec_per_pixel_norm=True)
  raw = make_legmm_agent(rec_per_pixel_norm=False)
  for key in normalized.params:
    np.testing.assert_array_equal(np.asarray(normalized.params[key]),
                                  np.asarray(raw.params[key]))
  batch = _random_batch(normalized, seed=7)
  norm = _reconstruction_losses(normalized, batch)
  unnorm = _reconstruction_losses(raw, batch)
  for key in ('depth_head', 'depth_wrist'):
    pixels = np.prod(normalized.model.dec.obs_space[key].shape[:2])
    np.testing.assert_allclose(norm[key] * pixels, unnorm[key], rtol=1e-5)
  for key in ('proprio', 'goal'):
    np.testing.assert_allclose(norm[key], unnorm[key], rtol=1e-5)


def test_mask_jax_matches_numpy():
  agent = make_legmm_agent()
  rng = np.random.default_rng(0)
  act = {
      'skill': rng.integers(0, 3, (7,), dtype=np.int32),
      'nav': rng.normal(size=(7, 4)).astype(np.float32),
      'pick': rng.normal(size=(7, 11)).astype(np.float32),
      'place': rng.normal(size=(7, 11)).astype(np.float32),
  }
  expected = interfaces.mask_inactive_np(act)
  mask = jax.jit(agent.model.mask_inactive)
  actual = mask({k: jax.numpy.asarray(v) for k, v in act.items()})
  for key in act:
    np.testing.assert_array_equal(np.asarray(actual[key]), expected[key])
  disabled = make_legmm_agent(skill_enabled=False)
  identity = jax.jit(disabled.model.mask_inactive)(
      {k: jax.numpy.asarray(v) for k, v in act.items()})
  for key in act:
    np.testing.assert_array_equal(np.asarray(identity[key]), act[key])
