#!/bin/bash
# M0 smoke test: dummy task, debug preset, CPU. Run from airdreamer/dreamerv3.
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
LOGDIR=${1:-~/logdir/legmm/m0_smoke}
python dreamerv3/main.py --configs defaults debug --task dummy_disc --script train \
  --logdir "$LOGDIR" --run.steps 3000 --jax.policy_devices 0 --jax.train_devices 0
python -m pytest tests/legmm -q -p no:warnings
