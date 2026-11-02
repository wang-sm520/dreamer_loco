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

  def __init__(self, k_rollouts=4, every_n_steps=2, max_dist=2.0, horizon=40,
               noise=0.05, seed=0, enabled=True, max_plans=3, grid=(5, 3, 3), **unused):
    self.max_plans = int(max_plans)
    self.grid = tuple(grid)
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
    """Label = best success rate over up to `max_plans` distinct local plans
    (base_adj only: |dx| <= 0.15, |dy| <= 0.12 m, |dyaw| <= 0.3 rad). Searching
    several plans (with gripper rolls and tilts) instead of one keeps labels
    close to the stronger oracle (see tools/legmm_judge_eval.py)."""
    from . import oracle
    bases = oracle.fine_bases(env.backend.state, env.scene.boxes, self.grid)
    rate, info = oracle.best_rate(env, which, bases, self.k, self.max_plans, self.horizon,
                                  int(self.rng.integers(2 ** 31)))
    self.rollouts += 4 * info['tried']
    return rate

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


GEOM_SLACK = 0.10   # matches the base_adj offsets the rollouts may use


def aff_geom_pick(backend, scene, max_angle=math.radians(60.0)):
  """Geometric heuristic (no rollouts): some free approach direction within
  60 deg of the robot admits a body height that puts the wrist center inside
  the reach shell for pre-grasp and grasp, and neither the TCP nor the wrist
  center lies inside an obstacle."""
  s = backend.state
  if s.holding or s.placed:
    return 0.0
  obj = s.obj_center
  base = (s.x, s.y, s.yaw)
  sh_w = backend.to_world(backend.shoulder_b)
  to_robot = sh_w[:2] - obj[:2]
  to_robot /= max(np.linalg.norm(to_robot), 1e-6)
  for i, _, axes in ex.approach_options(scene, s.yaw):
    if i < scenes.N_DIRS and scene.dirs[i][:2] @ to_robot < math.cos(max_angle):
      continue
    for _, a in axes:
      R_w = ex.grasp_frame(a, s.yaw)
      pre, goal, _ = ex.pick_targets(obj, a, i == scenes.N_DIRS)
      if ex.height_range(backend, base, (pre, goal), R_w, GEOM_SLACK) is None:
        continue
      pts = np.stack([goal, goal - backend.tool * a])
      if not geo.points_in_boxes(pts, scene.boxes, 0.02).any():
        return 1.0
  return 0.0


def aff_geom_place(backend, scene):
  s = backend.state
  if not s.holding:
    return 0.0
  zone = scene.place
  sh_w = backend.to_world(backend.shoulder_b)
  # Release point shifted 0.10 m towards the robot, inside the zone (as planned).
  toward = sh_w[:2] - zone.center
  toward /= max(np.linalg.norm(toward), 1e-6)
  r_obj = float(scene.obj.half[0])
  xy = np.clip(zone.center + 0.10 * toward, zone.center - zone.half + r_obj,
               zone.center + zone.half - r_obj)
  target = np.array([*xy, zone.support_z + scene.obj.half[2] + 0.015])
  base = (s.x, s.y, s.yaw)
  heading = math.atan2(target[1] - sh_w[1], target[0] - sh_w[0])
  if abs(math.atan2(math.sin(heading - s.yaw), math.cos(heading - s.yaw))) > math.radians(90):
    return 0.0
  for pitch in ex.PLACE_PITCHES:
    R_w = geo.rot_z(heading) @ geo.rot_pitch(-pitch)
    goal = target - R_w @ s.obj_offset
    if ex.height_range(backend, base, (goal + [0, 0, 0.08], goal), R_w, GEOM_SLACK) is None:
      continue
    pts = np.stack([goal, goal - backend.tool * R_w[:, 0]])
    if not geo.points_in_boxes(pts, scene.boxes, 0.02).any():
      return 1.0
  return 0.0
