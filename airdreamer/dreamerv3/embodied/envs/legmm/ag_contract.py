"""Evaluation-only true-state skill envelope; legacy callers never enter this module's checks.

Reach knots are measured by scan_reach.py, reach_by_height.json in
/home/user/logdir/legmm/v4/contract_true_state_20261110/ (new STOW, twist +-15 deg).
Between measured heights use linear interpolation; outside the height range the
height gate rejects, while interpolation clamps only for diagnostics/noisy estimates.
Footprint clearance uses the existing search's inflated rectangle convention.
"""
import math
import numpy as np
from . import ag_shaping as shp
from . import scenes as sc

BODY_MARGIN = 0.03
# Filled from the completed measurement, not from the old geometry audit.
REACH_HEIGHTS = (0.50, 0.55, 0.60, 0.65)
REACH_RADII = (0.80, 0.80, 0.79, 0.78)


def reach_limit(support_z):
  return float(np.interp(support_z, REACH_HEIGHTS, REACH_RADII))


def gate_cfg(support_z, cfg=None):
  g = shp.gate_cfg(cfg)
  return dict(g, dist=(g['dist'][0], reach_limit(support_z)))


def radius_grid(support_z, margin=0.0):
  """Original 5 cm grid plus the measured/interpolated endpoint; same margin clamping."""
  hi = reach_limit(support_z)
  rs = [i / 100 for i in range(45, int(math.floor(hi * 100 + 1e-9)) + 1, 5)]
  rs.append(hi)
  return tuple(dict.fromkeys(min(max(r, .45 + margin), hi - margin) for r in rs))


def report(backend, base=None, cfg=None):
  s = backend.state
  x, y, yaw = (s.x, s.y, s.yaw) if base is None else base
  dx, dy = backend.shoulder_b[:2]
  sh = (x + math.cos(yaw)*dx - math.sin(yaw)*dy,
        y + math.sin(yaw)*dx + math.cos(yaw)*dy)
  support = float(backend.scene.obj.support_z)
  g = gate_cfg(support, cfg)
  rep = shp.gate_report(sh, yaw, s.obj_center, support, 0., g)
  # Same footprint inflation as feasible_stances; tiny epsilon admits the exact boundary.
  rep['clearance_ok'] = bool(sc.footprint_free(backend._static, x, y, yaw, BODY_MARGIN - 1e-9))
  rep['rmax'] = g['dist'][1]
  rep['body_margin'] = BODY_MARGIN
  rep['ok'] = all(rep[k] for k in ('dist_ok', 'bearing_ok', 'height_ok', 'clearance_ok'))
  return rep
