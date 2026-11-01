"""drone_nav / drone_multi agents must match the pre-legmm baseline exactly.

tests/legmm/drone_baseline.json was written by drone_signature.py on the
unmodified repository (commit a57b89f).
"""
import json
import pathlib

import pytest

from drone_signature import signature

BASELINE = json.loads(
    (pathlib.Path(__file__).parent / 'drone_baseline.json').read_text())


@pytest.mark.parametrize('preset', ['drone_nav', 'drone_multi'])
def test_backward_compat(preset, tmp_path):
  sig = signature(preset, logdir=str(tmp_path))
  ref = BASELINE[preset]
  assert sig['n_params'] == ref['n_params']
  assert sig['param_keys_sha'] == ref['param_keys_sha']
  assert sig['loss_keys'] == ref['loss_keys']
  assert sig['loss_scales'] == ref['loss_scales']
  assert sig['enc_keys'] == ref['enc_keys']
  assert sig['dec_keys'] == ref['dec_keys']
