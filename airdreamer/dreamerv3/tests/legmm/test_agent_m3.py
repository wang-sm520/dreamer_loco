import copy

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3.agent import anneal_scale, bc_policy_input, soft_bce_logits
from test_agent_legmm import make_legmm_agent, _random_batch


def _pure_losses(agent, batch, gradients=False, neutralize_actions=False):
  """Run _apply_replay_context and loss with one unchanged parameter tree."""
  data = jax.tree.map(jax.device_put, batch)
  carry = agent.init_train(2)
  params = {
      k: (v.addressable_data(0) if hasattr(v, 'addressable_data') else v)
      for k, v in agent.params.items()}
  if neutralize_actions:
    params = {
        k: (jnp.zeros_like(v) if 'dyn/dynin' in k else v)
        for k, v in params.items()}
  apply = nj.pure(agent.model._apply_replay_context)
  state, (carry3, obs, prevact, _) = apply(params, carry, data, seed=0)
  loss = nj.pure(agent.model.loss)
  state, result = loss(state, carry3, obs, prevact, False, seed=0)
  losses = result[1][2]['losses']
  if not gradients:
    return losses, obs

  def gradloss(*args):
    return nj.grad(agent.model.loss, agent.model.pol, has_aux=True)(*args)
  gradfn = nj.pure(gradloss)
  _, (_, _, grads, _) = gradfn(state, carry3, obs, prevact, False, seed=0)
  return losses, obs, grads


def _raw_index(agent, t):
  return agent.config.replay_context + t


def _set_clean_sequence(agent, batch):
  batch = {k: np.array(v, copy=True) for k, v in batch.items()}
  batch['is_first'][:] = False
  batch['is_last'][:] = False
  batch['is_terminal'][:] = False
  batch['skill'][:] = 0
  return batch


def test_anneal_schedule_and_counter():
  np.testing.assert_allclose(
      [anneal_scale(x, 1.0, 0.1, 100) for x in (0, 50, 100, 200)],
      [1.0, 0.55, 0.1, 0.1], rtol=1e-6)
  agent = make_legmm_agent(
      train_phase='bc', bc={'scale': 1.0, 'scale_end': 0.1, 'anneal_steps': 100})
  assert float(agent.params['agent_step/value']) == 0
  batch = jax.tree.map(jax.device_put, _random_batch(agent))
  carry = agent.init_train(2)
  agent.train(carry, batch)
  assert float(agent.params['agent_step/value']) == 1


def test_bc_alignment_and_inactive_heads():
  agent = make_legmm_agent(
      train_phase='bc', bc={'scale': 1.0, 'scale_end': 1.0, 'anneal_steps': 1},
      batch_length=5)
  batch = _set_clean_sequence(agent, _random_batch(agent, seed=8))
  t0, t1 = 1, 3
  i0, i1 = _raw_index(agent, t0), _raw_index(agent, t1)
  batch['nav'][:, i0] = 0.0
  kw = {'neutralize_actions': True}
  base, _ = _pure_losses(agent, batch, **kw)

  active = copy.deepcopy(batch)
  active['nav'][:, i0] = 1.0
  changed, _ = _pure_losses(agent, active, **kw)
  np.testing.assert_allclose(base['bc'][:, :t0], changed['bc'][:, :t0], rtol=1e-6, atol=0)
  assert not np.allclose(base['bc'][:, t0], changed['bc'][:, t0])

  feature_probe = jnp.arange(2 * 5 * 3, dtype=jnp.float32).reshape(2, 5, 3)
  np.testing.assert_array_equal(np.asarray(bc_policy_input(feature_probe)),
                               np.asarray(feature_probe[:, :-1]))

  inactive = copy.deepcopy(batch)
  inactive['pick'][:, i0] = 1.0
  inactive_losses, _ = _pure_losses(agent, inactive, **kw)
  for key in base:
    np.testing.assert_array_equal(base[key], inactive_losses[key])

  terminal = copy.deepcopy(batch)
  terminal['is_last'][:, i1] = True
  terminal_base, _ = _pure_losses(agent, terminal, **kw)
  assert np.all(terminal_base['bc'][:, t1] == 0)
  terminal_action = copy.deepcopy(terminal)
  terminal_action['nav'][:, i1] = 1.0
  terminal_changed, _ = _pure_losses(agent, terminal_action, **kw)
  np.testing.assert_array_equal(
      terminal_base['bc'][:, :t1 + 1], terminal_changed['bc'][:, :t1 + 1])


def test_aff_loss_masked():
  logits = np.asarray([-30.0, 0.0, 30.0], np.float32)
  labels = np.asarray([0.25, 0.5, 0.75], np.float32)
  ref = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
  np.testing.assert_allclose(soft_bce_logits(logits, labels), ref, rtol=1e-6)

  agent = make_legmm_agent(train_phase='wm')
  batch = _set_clean_sequence(agent, _random_batch(agent, seed=4))
  batch['label_aff'][:] = 0.25
  batch['label_aff_mask'][:] = 1
  tmask = _raw_index(agent, 1)
  batch['label_aff_mask'][:, tmask] = 0
  base, _ = _pure_losses(agent, batch)
  masked = copy.deepcopy(batch)
  masked['label_aff'][:, tmask] = 1.0
  masked_losses, _ = _pure_losses(agent, masked)
  np.testing.assert_array_equal(base['aff'], masked_losses['aff'])
  valid = copy.deepcopy(batch)
  valid['label_aff'][:, _raw_index(agent, 0)] = 1.0
  valid_losses, _ = _pure_losses(agent, valid)
  assert not np.allclose(base['aff'], valid_losses['aff'])


def test_phase_loss():
  agent = make_legmm_agent(train_phase='wm')
  batch = _set_clean_sequence(agent, _random_batch(agent, seed=5))
  base, _ = _pure_losses(agent, batch)
  changed = copy.deepcopy(batch)
  changed['label_phase'][:, _raw_index(agent, 1)] = (
      changed['label_phase'][:, _raw_index(agent, 1)] + 1) % 3
  other, _ = _pure_losses(agent, changed)
  assert not np.allclose(base['phase'], other['phase'])


def test_train_phase_gating():
  expected = {
      'wm': {'aff', 'phase'},
      'bc': {'aff', 'phase', 'bc'},
      'rl': {'aff', 'phase', 'bc', 'policy', 'value', 'repval'},
      'online': {'aff', 'phase', 'policy', 'value', 'repval'},
  }
  base = {'con', 'dyn', 'rep', 'rew', 'depth_head', 'depth_wrist', 'proprio', 'goal'}
  for phase, extra in expected.items():
    kw = {'scale': 1.0, 'scale_end': 1.0, 'anneal_steps': 1} if phase in ('bc', 'rl') else None
    agent = make_legmm_agent(train_phase=phase, bc=kw)
    losses, _, grads = _pure_losses(
        agent, _random_batch(agent, seed={'wm': 1, 'bc': 2, 'rl': 3, 'online': 4}[phase]), gradients=True)
    assert set(losses) == base | extra
    pol_norm = sum(float(np.abs(np.asarray(x)).sum()) for x in grads.values())
    if phase == 'wm':
      assert pol_norm == 0.0
    if phase == 'rl':
      assert pol_norm > 0.0
