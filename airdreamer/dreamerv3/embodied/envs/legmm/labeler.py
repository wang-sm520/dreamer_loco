"""Affordance labels from privileged rollouts (brief §7.2).

On TO_PICK (or HOLDING) states within max_dist of the current target, every
every_n_steps steps: snapshot, plan a local manipulation that may only use
base_adj (no re-navigation), run k scripted rollouts with small action noise,
and use the success rate as label_aff[0] (or [1]); the matching mask entry is
set to 1. A geometric heuristic aff_geom is computed alongside for QA and
baselines; it is logged, never used as a training label.
"""
import math

import numpy as np

from . import expert as ex
from . import geometry as geo
from . import interfaces as I
from . import scenes


class AffordanceLabeler:

  def __init__(self, k_rollouts=4, every_n_steps=2, max_dist=2.0, horizon=30,
               noise=0.05, seed=0, enabled=True, **unused):
    self.k = int(k_rollouts)
    self.every = int(every_n_steps)
    self.max_dist = float(max_dist)
    self.horizon = int(horizon)
    self.noise = float(noise)
    self.rng = np.random.default_rng(seed)
    self.enabled = enabled
    self.last = None
    self.calls = 0
    self.rollouts = 0

  def __call__(self, env):
    b, s = env.backend, env.backend.state
    phase = b.phase()
    aff = np.zeros(2, np.float32)
    mask = np.zeros(2, np.float32)
    geom = np.array([aff_geom_pick(b, env.scene), aff_geom_place(b, env.scene)], np.float32)
    self.last = dict(geom=geom, labeled=False)
    if not self.enabled or phase == I.Phase.PLACED or env._step % self.every:
      return aff, mask
    i = int(phase)       # TO_PICK -> 0, HOLDING -> 1
    target = s.obj_center if i == 0 else np.array([*env.scene.place.center, 0.0])
    if math.hypot(target[0] - s.x, target[1] - s.y) > self.max_dist:
      return aff, mask
    self.calls += 1
    aff[i] = self.rate(env, i)
    mask[i] = 1.0
    self.last['labeled'] = True
    return aff, mask

  def rate(self, env, which):
    b = env.backend
    s = b.state
    bases = ex.local_bases(s, env.scene.boxes)
    if which == 0:
      plans = ex.pick_candidates(b, env.scene, bases)
    else:
      plans = ex.place_candidates(b, env.scene, lambda sd, c: bases(sd))
    if not plans:
      return 0.0
    plans.sort(key=lambda p: p.score + _offset_cost(p, s))
    plan = None
    for cand in plans[:8]:
      way = ex.transit_waypoints(b, cand, s.ee_b, s.R_ee_b, s.holding, s.obj_offset)
      if way is not None:
        cand.way_w = way
        plan = cand
        break
    if plan is None:
      return 0.0
    snap = b.snapshot()
    wins = 0
    for _ in range(self.k):
      b.restore(snap)
      # Same Generator object (shared with the goal teacher), fresh state.
      b.rng.bit_generator.state = np.random.default_rng(
          self.rng.integers(2 ** 63)).bit_generator.state
      wins += self._rollout(env, plan, which)
      self.rollouts += 1
    b.restore(snap)
    return wins / self.k

  def _rollout(self, env, plan, which):
    b = env.backend
    ctrl = ex.ManipController(plan)
    for _ in range(self.horizon):
      act, status = ctrl.step(b)
      if status != 'running':
        return status == 'done' or (which == 1 and b.state.placed)
      skill = I.Skill(int(act['skill']))
      key = I.SKILL_ACTION_KEY[skill]
      vec = np.clip(act[key] + self.rng.normal(0, self.noise, act[key].shape), -1, 1)
      ev = b.step(env._command(skill, vec.astype(np.float32)))
      if ev['grasp_success'] and which == 0:
        return True
      if ev['place_success']:
        return which == 1
      if ev['dropped'] or ev['collision']:
        return False
      if b.state.n_failed > 0 and which == 0 and not b.state.holding and ev['failed_grasp']:
        return False
    return False


def _offset_cost(plan, s):
  return math.hypot(plan.base[0] - s.x, plan.base[1] - s.y) + 0.3 * abs(
      math.atan2(math.sin(plan.base[2] - s.yaw), math.cos(plan.base[2] - s.yaw)))


def _line_free(p0, p1, boxes, margin=0.02, skip_end=0.06):
  d = np.linalg.norm(p1 - p0)
  if d < 1e-6:
    return True
  n = max(2, int(d / 0.03))
  ts = np.linspace(0, max(0.0, 1 - skip_end / d), n)[:, None]
  pts = p0[None] * (1 - ts) + p1[None] * ts
  return not geo.points_in_boxes(pts, boxes, margin).any()


def aff_geom_pick(backend, scene, max_angle=math.radians(60.0)):
  """Heuristic: the object is inside the reach shell for some body height, a
  free approach direction points within 60 deg of the robot, and the straight
  line from the shoulder to the object is collision-free."""
  s = backend.state
  if s.holding or s.placed:
    return 0.0
  obj = s.obj_center
  sh_w = backend.to_world(backend.shoulder_b)
  to_robot = sh_w[:2] - obj[:2]
  n = np.linalg.norm(to_robot)
  if n < 1e-6:
    return 0.0
  to_robot /= n
  free = np.where(scene.approach[:scenes.N_DIRS])[0]
  ok_dir = any(scenes.APPROACH_DIRS[i][:2] @ to_robot > math.cos(max_angle) for i in free)
  ok_dir = ok_dir or bool(scene.approach[scenes.N_DIRS])
  L = backend.L1 + backend.L2 + backend.tool
  best = None
  for h in np.linspace(*I.BASE_HEIGHT_RANGE, 6):
    sh = sh_w.copy()
    sh[2] += h - s.h
    if np.linalg.norm(obj - sh) <= L:
      best = sh
      break
  if best is None or not ok_dir:
    return 0.0
  return float(_line_free(best, obj, scene.boxes))


def aff_geom_place(backend, scene):
  s = backend.state
  if not s.holding:
    return 0.0
  zone = scene.place
  target = np.array([*zone.center, zone.support_z + scene.obj.half[2] + 0.01])
  sh_w = backend.to_world(backend.shoulder_b)
  L = backend.L1 + backend.L2 + backend.tool
  for h in np.linspace(*I.BASE_HEIGHT_RANGE, 6):
    sh = sh_w.copy()
    sh[2] += h - s.h
    if np.linalg.norm(target - sh) <= L:
      return float(_line_free(sh, target, scene.boxes))
  return 0.0
