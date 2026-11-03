import ast
import contextlib
import io

import numpy as np
import pytest

from test_agent_legmm import make_legmm_agent


def test_bc_checkpoint_loads_into_rl_with_exact_normalizers_and_strict_model_keys():
  bc = make_legmm_agent(train_phase='bc', jax_jit=True)
  rl = make_legmm_agent(train_phase='rl', jax_jit=True)
  checkpoint = bc.save()
  source = {key: np.asarray(value) for key, value in checkpoint['params'].items()}
  expected = sorted(set(rl.params) - set(source))
  assert expected == ['retnorm/hi/value', 'retnorm/lo/value']

  output = io.StringIO()
  with contextlib.redirect_stdout(output):
    rl.load(checkpoint)
  lines = [line for line in output.getvalue().splitlines()
           if line.startswith('Checkpoint phase differences:')]
  assert len(lines) == 1
  missing_text = lines[0].split('missing=', 1)[1].split(', unexpected=', 1)[0]
  assert ast.literal_eval(missing_text) == expected
  for key, value in source.items():
    np.testing.assert_array_equal(np.asarray(rl.params[key]), value)

  for prefix in ('enc/', 'pol/'):
    broken = dict(checkpoint)
    broken['params'] = dict(source)
    broken['params'].pop(next(key for key in source if key.startswith(prefix)))
    with pytest.raises(KeyError):
      rl.load(broken)
