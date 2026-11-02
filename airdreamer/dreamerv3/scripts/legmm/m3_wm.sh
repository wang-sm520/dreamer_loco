#!/usr/bin/env bash
set -euo pipefail
exec python dreamerv3/main.py \
  --configs legmm legmm_wm \
  --script train_offline \
  --offline.dir ~/data/legmm_v4/train \
  --logdir ~/logdir/legmm/m3/wm
