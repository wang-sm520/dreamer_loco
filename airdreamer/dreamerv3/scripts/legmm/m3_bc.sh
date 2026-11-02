#!/usr/bin/env bash
set -euo pipefail
exec python dreamerv3/main.py \
  --configs legmm legmm_bc \
  --script train_offline \
  --offline.dir ~/data/legmm_v4/train \
  --run.from_checkpoint ~/logdir/legmm/m3/wm/ckpt \
  --logdir ~/logdir/legmm/m3/bc
