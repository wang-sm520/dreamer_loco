"""Stronger affordance oracle used to audit labels and judges (not for training).

Same motion envelope as the labeler (base_adj only: |dx|,|dy| <= 0.15/0.12 m,
|dyaw| <= 0.3 rad, no re-navigation) but a much wider search: 5x5x5 base
offsets, 4 approach tilts, gripper rolls 0/+-90 deg, up to `max_plans`
distinct plans x k rollouts, longer horizon. Returns the best plan's success
rate.
"""
import math

import numpy as np

from . import expert as ex
from . import geometry as geo
from . import interfaces as I
from . import labeler as lb
from . import scenes

TILTS = (0.45, 0.30, 0.15, 0.0)
ROLLS = (0.0, math.pi / 2, -math.pi / 2)


def fine_bases(state, boxes):
  offs = [(dx, dy, dyaw) for dx in np.linspace(-0.15, 0.15, 5) for dy in np.linspace(-0.12, 0.12, 5)
          for dyaw in np.linspace(-0.3, 0.3, 5)]
  return ex.local_bases(state, boxes, offs)()


def pick_plans(backend, scene, bases):
  s = backend.state
  out = []
  for i in range(scenes.N_DIRS + 1):
    if not scene.approach[i]:
      continue
    top = i == scenes.N_DIRS
    if top:
      axes = [(0.0, np.array([0.0, 0.0, -1.0]))]
    else:
      d = scenes.APPROACH_DIRS[i][:2]
      axes = [(t, np.array([-d[0] * math.cos(t), -d[1] * math.cos(t), -math.sin(t)])) for t in TILTS]
    for tilt, a in axes:
      pre, goal, post = ex.pick_targets(s.obj_center, a, top)
      for base in bases:
        for roll in ROLLS:
          R_w = ex.grasp_frame(a, base[2]) @ geo.rot_x(roll)
          if ex.height_range(backend, base, (pre, goal), R_w) is None:
            break   # reach does not depend on roll
          h = ex.plan_at_base(backend, 'pick', base, R_w, pre, goal, post)
          if h is not None:
            out.append(ex.ManipPlan('pick', np.asarray(base, np.float64), h, R_w, pre, goal, post,
                                    np.zeros(2), i, score=0.3 * (ex.H_MAX - h) + 0.2 * tilt + 0.1 * abs(roll)))
            break
  return out


def oracle_pick_rate(env, k=4, max_plans=6, horizon=45, seed=0):
  b, s = env.backend, env.backend.state
  plans = pick_plans(b, env.scene, fine_bases(s, env.scene.boxes))
  plans.sort(key=lambda p: p.score + lb._offset_cost(p, s))
  lab = lb.AffordanceLabeler(k_rollouts=k, horizon=horizon, seed=seed)
  tried, best, seen = 0, 0.0, set()
  snap = b.snapshot()
  for p in plans:
    key = (p.dir_index, round(p.base[0], 2), round(p.base[1], 2), round(p.base[2], 2))
    if key in seen:
      continue
    way = ex.transit_waypoints(b, p, s.ee_b, s.R_ee_b)
    if way is None:
      continue
    seen.add(key)
    p.way_w = way
    wins = 0
    for _ in range(k):
      b.restore(snap)
      b.rng.bit_generator.state = np.random.default_rng(lab.rng.integers(2 ** 63)).bit_generator.state
      wins += lab._rollout(env, p, 0)
    b.restore(snap)
    best = max(best, wins / k)
    tried += 1
    if best == 1.0 or tried >= max_plans:
      break
  return best, dict(plans=len(plans), tried=tried)
