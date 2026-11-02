import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embodied.jax.outs import Categorical, Normal
from dreamerv3.agent import imag_loss
from imag_loss_reference import imag_loss as reference_imag_loss


@pytest.fixture(autouse=True)
def _allow_cpu_transfers():
  jax.config.update('jax_transfer_guard', 'allow')


class _UnitNorm:

  def stats(self):
    return jnp.array(0.0), jnp.array(1.0)

  def __call__(self, value, update):
    del value, update
    return jnp.array(0.0), jnp.array(1.0)


def test_disabled_skill_imag_loss_matches_commit_reference():
  rng = np.random.default_rng(314159)
  batch, length = 3, 7
  act = {
      'continuous': jnp.asarray(rng.normal(size=(batch, length)), jnp.float32),
      'discrete': jnp.asarray(rng.integers(0, 3, size=(batch, length)), jnp.int32),
  }
  policy = {
      'continuous': Normal(jnp.asarray(rng.normal(size=(batch, length)), jnp.float32)),
      'discrete': Categorical(jnp.asarray(rng.normal(size=(batch, length, 3)), jnp.float32)),
  }
  rew = jnp.asarray(rng.normal(size=(batch, length)), jnp.float32)
  con = jnp.asarray(rng.uniform(0.7, 1.0, size=(batch, length)), jnp.float32)
  value = Normal(jnp.asarray(rng.normal(size=(batch, length)), jnp.float32))
  slowvalue = Normal(jnp.asarray(rng.normal(size=(batch, length)), jnp.float32))
  norms = (_UnitNorm(), _UnitNorm(), _UnitNorm())
  kwargs = dict(
      update=False, contdisc=True, slowtar=False, horizon=17, lam=0.95,
      actent=3e-4, actsmooth=5e-2, slowreg=1.0)

  current = imag_loss(
      act, rew, con, policy, value, slowvalue, *norms, skill_key=None,
      **kwargs)
  reference = reference_imag_loss(
      act, rew, con, policy, value, slowvalue, *norms, **kwargs)

  for key in reference[0]:
    np.testing.assert_allclose(current[0][key], reference[0][key], rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(current[1]['ret'], reference[1]['ret'], rtol=1e-6, atol=1e-6)
  assert current[2].keys() == reference[2].keys()
  for key in reference[2]:
    np.testing.assert_allclose(current[2][key], reference[2][key], rtol=1e-6, atol=1e-6)
