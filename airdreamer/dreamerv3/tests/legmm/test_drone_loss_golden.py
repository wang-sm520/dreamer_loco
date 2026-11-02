import hashlib
import json

import jax
import ninjax as nj
import numpy as np
import pytest

import elements
from drone_signature import drone_spaces, load_config
from dreamerv3.agent import Agent


GOLDEN = json.loads((__import__('pathlib').Path(__file__).with_name(
    'drone_loss_golden.json')).read_text())


@pytest.fixture(autouse=True)
def _allow_cpu_transfers():
  jax.config.update('jax_transfer_guard', 'allow')


def _make_agent_and_batch():
  overrides = [
      '--jax.platform=cpu', '--jax.policy_devices=0', '--jax.train_devices=0',
      '--jax.prealloc=False', '--jax.jit=False', '--batch_size=2',
      '--batch_length=5', '--replay_context=2', '--logdir=/tmp/drone-golden-head']
  config = load_config(['drone_nav'], overrides)
  obs, act = drone_spaces('drone_nav')
  agent_cfg = elements.Config(
      **config.agent, logdir=config.logdir, seed=config.seed,
      jax=dict(config.jax, precompile=False), batch_size=config.batch_size,
      batch_length=config.batch_length, replay_context=config.replay_context,
      report_length=config.report_length, replica=config.replica,
      replicas=config.replicas)
  agent = Agent(obs, act, agent_cfg)
  rng = np.random.default_rng(GOLDEN['seed'])
  batch_size = GOLDEN['batch_size']
  length = GOLDEN['batch_length'] + GOLDEN['replay_context']
  batch = {}
  for key, space in agent.spaces.items():
    shape = (batch_size, length, *space.shape)
    if space.dtype == np.bool_:
      value = rng.integers(0, 2, shape, dtype=np.uint8).astype(bool)
    elif space.discrete:
      value = rng.integers(space.low, space.high, shape, dtype=space.dtype)
    else:
      value = rng.normal(size=shape).astype(space.dtype)
    batch[key] = value
  batch['consec'][:] = 0
  batch['is_first'][:] = False
  batch['is_first'][:, 0] = True
  batch['is_last'][:] = False
  batch['is_terminal'][:] = False
  return agent, batch


def _param_tree_sha256(params):
  digest = hashlib.sha256()
  for key in sorted(params):
    value = params[key].addressable_data(0) if hasattr(params[key], 'addressable_data') else params[key]
    digest.update(key.encode())
    digest.update(np.asarray(jax.device_get(value)).tobytes())
  return digest.hexdigest()


def test_drone_loss_matches_golden_and_parameter_tree():
  agent, batch = _make_agent_and_batch()
  params = {k: (v.addressable_data(0) if hasattr(v, 'addressable_data') else v)
            for k, v in agent.params.items()}
  assert _param_tree_sha256(params) == GOLDEN['param_tree_sha256']
  carry = agent.init_train(GOLDEN['batch_size'])
  apply = nj.pure(agent.model._apply_replay_context)
  state, (carry3, obs, prevact, _) = apply(params, carry, batch, seed=0)
  lossfn = nj.pure(agent.model.loss)
  state, result = lossfn(state, carry3, obs, prevact, False, seed=0)
  losses = result[1][2]['losses']
  means = {key: float(np.asarray(value).mean()) for key, value in losses.items()}
  total = sum(means[key] * float(agent.model.scales[key]) for key in means)
  assert means.keys() == GOLDEN['loss_means'].keys()
  for key in means:
    np.testing.assert_allclose(means[key], GOLDEN['loss_means'][key], rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(total, GOLDEN['total_loss'], rtol=1e-5, atol=1e-5)
