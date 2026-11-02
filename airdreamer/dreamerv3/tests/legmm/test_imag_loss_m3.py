import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The CPU-only test process may have already initialized a JAX agent with the
# repository's transfer guard enabled. Pure-function tests intentionally pass
# host arrays, so allow those transfers for this module.
jax.config.update('jax_transfer_guard', 'allow')


@pytest.fixture(autouse=True)
def _allow_cpu_transfers():
  jax.config.update('jax_transfer_guard', 'allow')

from embodied.envs.legmm import interfaces as I
from embodied.jax.outs import Categorical, Normal
from dreamerv3.agent import imag_loss, skill_prior_jax


class _UnitNorm:

  def stats(self):
    return jnp.array(0.0), jnp.array(1.0)

  def __call__(self, value, update):
    del value, update
    return jnp.array(0.0), jnp.array(1.0)


def _inputs(skill=(0, 0, 1, 2, 0)):
  skill = jnp.asarray(skill, jnp.int32)[None]
  n, length = skill.shape
  zeros = jnp.zeros((n, length))
  act = {
      'skill': skill,
      'nav': jnp.ones((n, length)),
      'pick': jnp.ones((n, length)) * 2,
      'place': jnp.ones((n, length)) * 3,
  }
  policy = {
      'skill': Categorical(jnp.zeros((n, length, 3))),
      'nav': Normal(zeros), 'pick': Normal(zeros), 'place': Normal(zeros),
  }
  return act, policy, zeros, jnp.ones_like(zeros)


def _loss(policy, act, rew, *, actent=0.0, actsmooth=0.0):
  unit = _UnitNorm()
  return imag_loss(
      act, rew, jnp.ones_like(rew), policy,
      Normal(jnp.zeros_like(rew)), Normal(jnp.zeros_like(rew)),
      unit, unit, unit, False, horizon=10, actent=actent,
      actsmooth=actsmooth, skill_key='skill',
      action_keys=('nav', 'pick', 'place'))[0]['policy'].sum()


def test_imag_mask_logp_entropy_and_smoothness():
  act, policy, zeros, _ = _inputs()
  rew = jnp.array([[0.0, 1.0, 2.0, 1.0, 1.0]])

  def logp_grad(key):
    def fn(x):
      p = dict(policy)
      p[key] = Normal(x) if key != 'skill' else Categorical(x)
      return _loss(p, act, rew)
    return jax.grad(fn)(jnp.zeros((1, 5, 3)) if key == 'skill' else zeros)

  for i, (key, grad) in enumerate(zip(
      ('nav', 'pick', 'place'), (logp_grad('nav'), logp_grad('pick'), logp_grad('place')))):
    assert np.linalg.norm(np.asarray(grad)) > 0
    grad = np.asarray(grad)[:, :-1]
    inactive = np.asarray(act['skill'][:, :-1] != i)
    assert np.all(grad[inactive] == 0)
  skill_grad = logp_grad('skill')
  assert np.linalg.norm(np.asarray(skill_grad)) > 0

  # Entropy is independently active even when the advantage is zero.
  zero_rew = jnp.zeros_like(rew)
  for key in ('nav', 'pick', 'place'):
    def ent_fn(x, key=key):
      p = dict(policy)
      p[key] = Normal(zeros, jnp.exp(x))
      return _loss(p, act, zero_rew, actent=1.0)
    grad = jax.grad(ent_fn)(zeros)
    grad = grad[:, :-1]
    inactive = act['skill'][:, :-1] != ('nav', 'pick', 'place').index(key)
    active = ~inactive
    assert np.linalg.norm(np.asarray(grad)[np.asarray(active)]) > 0
    assert np.all(np.asarray(grad)[np.asarray(inactive)] == 0)

  def smooth_nav_fn(x):
    p = dict(policy)
    p['nav'] = Normal(x)
    return _loss(p, act, zero_rew, actsmooth=1.0)
  nav_means = jnp.arange(5, dtype=jnp.float32)[None] * 0.1
  smooth_grad = jax.grad(smooth_nav_fn)(nav_means)
  assert np.linalg.norm(np.asarray(smooth_grad)) > 0
  smooth_pairs = np.asarray((act['skill'][:, :-1] == 0) &
                            (act['skill'][:, 1:] == 0))
  smooth_active = np.pad(smooth_pairs, ((0, 0), (0, 1)), constant_values=False)
  smooth_active[:, 1:] |= smooth_pairs
  assert np.all(np.asarray(smooth_grad)[~smooth_active] == 0)

  def smooth_skill_fn(x):
    p = dict(policy)
    p['skill'] = Categorical(x)
    return _loss(p, act, zero_rew, actsmooth=1.0)
  skill_logits = jnp.arange(15, dtype=jnp.float32).reshape(1, 5, 3) * 0.1
  smooth_skill_grad = jax.grad(smooth_skill_fn)(skill_logits)
  # The discrete skill head is never included in smoothness.
  assert np.all(np.asarray(smooth_skill_grad) == 0)


def test_switch_cost_alignment():
  act, policy, _, _ = _inputs(skill=(0, 0, 1, 1, 2))
  rew = jnp.zeros((1, 5))
  out = imag_loss(
      act, rew, jnp.ones_like(rew), policy,
      Normal(jnp.zeros_like(rew)), Normal(jnp.zeros_like(rew)),
      _UnitNorm(), _UnitNorm(), _UnitNorm(), False, horizon=10,
      skill_key='skill', action_keys=('nav', 'pick', 'place'),
      switch_cost=0.25)[1]
  np.testing.assert_allclose(out['rew'], [[0.0, 0.0, 0.0, -0.25, 0.0]])


def test_skill_prior_parity_and_stop_gradient():
  rng = np.random.default_rng(7)
  aff_pick, aff_place, holding = rng.random((3, 2, 4)).astype(np.float32)
  expected = I.skill_prior(aff_pick, aff_place, holding, eps=0.05)
  actual = skill_prior_jax(aff_pick, aff_place, holding, eps=0.05)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  n, length = 1, 4
  aff = jnp.zeros((n, length, 2))
  phase = jnp.zeros((n, length, 3))
  prior = skill_prior_jax(
      jax.nn.sigmoid(aff)[..., 0], jax.nn.sigmoid(aff)[..., 1],
      jax.nn.softmax(phase, -1)[..., 1])
  logits = jnp.log(prior)
  act = {'skill': jnp.zeros((n, length), jnp.int32)}
  policy = {'skill': Categorical(logits)}
  zeros = jnp.zeros((n, length))

  def fn(logits, aff, phase):
    losses = imag_loss(
        act, zeros, jnp.ones_like(zeros), {'skill': Categorical(logits)},
        Normal(zeros), Normal(zeros), _UnitNorm(), _UnitNorm(), _UnitNorm(),
        False, horizon=10, skill_key='skill', skill_prior_scale=1.0,
        aff=aff, phase=phase)[0]
    return losses['skill_prior'].sum()

  probe_logits = jnp.broadcast_to(jnp.array([2.0, 0.0, -1.0]), logits.shape)
  probe_pi = jax.nn.softmax(probe_logits, -1)
  expected_kl = (probe_pi * (jnp.log(probe_pi) - jnp.log(prior))).sum(-1)[:, :-1].sum()
  np.testing.assert_allclose(float(fn(probe_logits, aff, phase)), float(expected_kl), rtol=1e-5)
  probe_grads = jax.grad(fn, argnums=(1, 2))(probe_logits, aff, phase)
  assert np.all(np.asarray(probe_grads[0]) == 0)
  assert np.all(np.asarray(probe_grads[1]) == 0)
  value, grads = jax.value_and_grad(fn, argnums=(0, 1, 2))(logits, aff, phase)
  assert float(value) < 1e-6
  assert np.linalg.norm(np.asarray(grads[0])) < 1e-5
  assert np.all(np.asarray(grads[1]) == 0)
  assert np.all(np.asarray(grads[2]) == 0)
