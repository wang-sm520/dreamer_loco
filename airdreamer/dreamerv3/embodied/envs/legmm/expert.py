"""Scripted expert with privileged state (brief §7.1).

NAV: Dijkstra/A*-equivalent shortest path on the inflated 5 cm grid to a
staging point, pure pursuit, align, straight approach to the pre-manipulation
base pose. Pre-manipulation poses are sampled along approach directions and
scored by reachability (arm reach shell, collisions) and path length.
PICK / PLACE: pre-grasp (10 cm back along the approach axis) -> advance ->
close -> lift; pre-place -> lower -> open. Diversity injection: bad poses,
early switches, DART noise, random exploration segments, and recovery after
failures. Actions are masked and normalized to [-1, 1].
"""
import collections
import copy
import dataclasses
import math

import numpy as np

from . import geometry as geo
from . import interfaces as I
from . import kin_backend as kb
from . import planning
from . import scenes

H_MIN, H_MAX = I.BASE_HEIGHT_RANGE
STAND_RADII = (0.45, 0.55, 0.65, 0.72, 0.8, 0.88, 0.96)
TILTS = (0.30, 0.15, 0.0)          # approach axis tilt below horizontal (rad)
PRE_BACK = 0.10                    # pre-grasp distance along the approach axis
ARM_GAIN = 0.6                     # proportional gain on TCP / orientation errors
PLAN_MARGIN = 0.01                 # extra clearance in planning (x3 for self collision)
EXPLORE_SIGMA = 0.3                # noise of random exploration segments (normalized units)
LIFT = 0.10
STAGE_BACK = (0.4, 0.25, 0.6)
LOCAL_OFFSETS = [(dx, dy, dyaw) for dx in (0.0, -0.15, 0.15) for dy in (0.0, -0.12, 0.12)
                 for dyaw in (0.0, -0.3, 0.3)]


@dataclasses.dataclass
class ManipPlan:
  kind: str                 # 'pick' | 'place'
  base: np.ndarray          # (3,) x, y, yaw
  h: float
  R_w: np.ndarray           # gripper orientation (world)
  pre_w: np.ndarray         # TCP pre-pose
  goal_w: np.ndarray        # TCP grasp / release pose
  post_w: np.ndarray        # TCP after closing / opening
  stand_dir: np.ndarray     # (2,) unit, object -> robot
  dir_index: int            # approach bin (8 = top)
  score: float = 0.0
  way_w: tuple = ()         # TCP waypoints (world) before pre_w
  raise_z: float = 0.0      # body-frame TCP height reached straight up after a pick
  turn_first: bool = False  # rotate to the carry orientation before retracting


@dataclasses.dataclass
class ExpertConfig:
  diversity: bool = True
  p_bad_pose: float = 0.3
  p_early_switch: float = 0.2
  p_dart: float = 0.5
  dart_sigma: float = 0.1
  explore_frac: float = 0.1
  p_other_side: float = 0.5
  max_replans: int = 8


# ---------------------------------------------------------------------------
# Kinematic feasibility helpers (shared with the labeler)
# ---------------------------------------------------------------------------

def grasp_frame(a, yaw):
  a = np.asarray(a, np.float64)
  a = a / np.linalg.norm(a)
  y = np.cross([0.0, 0.0, 1.0], a)
  if np.linalg.norm(y) < 1e-3:
    y = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
  y /= np.linalg.norm(y)
  return np.stack([a, y, np.cross(a, y)], 1)


def _state_at(backend, base, h):
  s = copy.copy(backend.state)
  s.x, s.y, s.yaw = float(base[0]), float(base[1]), float(base[2])
  s.h, s.pitch = float(h), 0.0
  return s


def tcp_feasible(backend, base, h, tcp_w, R_w, holding=False, obj_offset=None):
  s = _state_at(backend, base, h)
  Rb = backend.R_body(s)
  R_ee_b = Rb.T @ R_w
  ee_b = Rb.T @ (np.asarray(tcp_w) - backend.base_pos(s))
  if np.linalg.norm(backend.clip_ee(ee_b, s, R_ee_b) - ee_b) > 0.005:
    return False
  s.ee_b, s.R_ee_b = ee_b, R_ee_b
  s.holding = holding
  if holding:
    s.obj_center = np.asarray(tcp_w) + R_w @ obj_offset
  return not backend.arm_collides(s, PLAN_MARGIN)


def height_range(backend, base, pts_w, R_w, slack=0.0):
  """Body heights for which the wrist centers of all TCP poses are in reach
  (reach extended by slack, e.g. to account for small base adjustments)."""
  x, y, yaw = base
  a = R_w[:, 0]
  sh = backend.shoulder_b
  c, sn = math.cos(yaw), math.sin(yaw)
  sx, sy = x + c * sh[0] - sn * sh[1], y + sn * sh[0] + c * sh[1]
  L = backend.L1 + backend.L2 - 0.01 + slack
  lo, hi = H_MIN, H_MAX
  for p in pts_w:
    w = np.asarray(p) - backend.tool * a
    dh = math.hypot(w[0] - sx, w[1] - sy)
    if dh >= L:
      return None
    r = math.sqrt(L * L - dh * dh)
    lo = max(lo, w[2] - sh[2] - r)
    hi = min(hi, w[2] - sh[2] + r)
  return (lo, hi) if lo <= hi else None


def plan_at_base(backend, kind, base, R_w, pre_w, goal_w, post_w, holding=False, obj_offset=None):
  """Highest feasible body height for executing pre -> goal -> post from base."""
  hr = height_range(backend, base, (pre_w, goal_w), R_w)
  if hr is None:
    return None
  lo, hi = hr
  for h in (hi - 0.03, hi - 0.10, hi - 0.18, lo + 0.02):
    if h < lo or h > hi or h < H_MIN or h > H_MAX:
      continue
    if all(tcp_feasible(backend, base, h, p, R_w, holding, obj_offset) for p in (pre_w, goal_w)):
      return h
  return None


def approach_options(scene, yaw_hint, include_blocked=False):
  """[(dir_index, stand_dirs (k, 2), approach axes [(tilt, a)])]."""
  opts = []
  for i in range(scenes.N_DIRS):
    if not (scene.approach[i] or include_blocked):
      continue
    d = scenes.APPROACH_DIRS[i][:2]
    axes = [(t, np.array([-d[0] * math.cos(t), -d[1] * math.cos(t), -math.sin(t)])) for t in TILTS]
    rot = lambda v, q: np.array([v[0] * math.cos(q) - v[1] * math.sin(q), v[0] * math.sin(q) + v[1] * math.cos(q)])
    stands = np.stack([d, rot(d, 0.5), rot(d, -0.5)])
    opts.append((i, stands, axes))
  if scene.approach[scenes.N_DIRS] or include_blocked:
    stands = scenes.APPROACH_DIRS[:scenes.N_DIRS, :2]
    opts.append((scenes.N_DIRS, stands, [(0.0, np.array([0.0, 0.0, -1.0]))]))
  return opts


def pick_targets(obj_center, a, top):
  goal = np.asarray(obj_center, np.float64) + ([0, 0, 0.02] if top else [0, 0, 0])
  pre = goal - PRE_BACK * a
  post = goal + [0, 0, LIFT]
  return pre, goal, post


def pick_candidates(backend, scene, bases_fn, exclude=(), include_blocked=False):
  """Feasible pick plans. bases_fn(stand_dir) -> list of (x, y, yaw)."""
  s = backend.state
  out = []
  for i, stands, axes in approach_options(scene, s.yaw, include_blocked):
    if i in exclude:
      continue
    top = i == scenes.N_DIRS
    for tilt, a in axes:
      for sd in stands:
        for base in bases_fn(sd):
          R_w = grasp_frame(a, base[2])
          pre, goal, post = pick_targets(s.obj_center, a, top)
          h = plan_at_base(backend, 'pick', base, R_w, pre, goal, post)
          if h is not None:
            out.append(ManipPlan('pick', np.asarray(base, np.float64), h, R_w, pre, goal, post,
                                 np.asarray(sd), i, score=0.3 * (H_MAX - h) + 0.2 * tilt))
  return out


def place_targets(scene, center_obj, R_w, obj_offset):
  goal = center_obj - R_w @ obj_offset
  return goal + [0, 0, 0.08], goal, goal + [0, 0, LIFT]


PLACE_PITCHES = (math.radians(30.0), 0.0, math.radians(60.0), math.radians(89.0))


def place_candidates(backend, scene, bases_fn, jitter=np.zeros(2)):
  """Release poses on the place zone. The release point is shifted towards the
  robot (staying inside the zone) and the gripper may pitch 0-90 deg down."""
  s = backend.state
  zone = scene.place
  hz = scene.obj.half[2]
  r_obj = float(scene.obj.half[0])
  out = []
  for k in range(scenes.N_DIRS):
    sd = scenes.APPROACH_DIRS[k][:2]
    shift = np.clip(zone.center + 0.10 * sd + jitter, zone.center - zone.half + r_obj,
                    zone.center + zone.half - r_obj)
    # Release with the bottom 1.5 cm above the support: middle of the env's
    # [-0.005, 0.03] window; the closed-loop hold during 'act' keeps it there.
    center = np.array([*shift, zone.support_z + hz + 0.015])
    for base in bases_fn(sd, center):
      for pitch in PLACE_PITCHES:
        R_w = geo.rot_z(base[2]) @ geo.rot_pitch(-pitch)
        pre, goal, post = place_targets(scene, center, R_w, s.obj_offset)
        h = plan_at_base(backend, 'place', base, R_w, pre, goal, post, True, s.obj_offset)
        if h is not None:
          out.append(ManipPlan('place', np.asarray(base, np.float64), h, R_w, pre, goal, post,
                               np.asarray(sd), k, score=0.3 * (H_MAX - h) + 0.1 * pitch))
          break
  return out


def radial_bases(target_xy, boxes, radii=STAND_RADII, laterals=(0.0, 0.1, -0.1, 0.2, -0.2)):
  """Base poses around a target: along stand direction sd at radius r, shifted
  sideways, either facing the target or facing along -sd."""
  def fn(sd, *_):
    out = []
    perp = np.array([-sd[1], sd[0]])
    for r in radii:
      for lat in laterals:
        x, y = target_xy[0] + r * sd[0] + lat * perp[0], target_xy[1] + r * sd[1] + lat * perp[1]
        yaws = {math.atan2(target_xy[1] - y, target_xy[0] - x), math.atan2(-sd[1], -sd[0])}
        for yaw in yaws:
          if scenes.footprint_free(boxes, x, y, yaw, 0.03):
            out.append((x, y, yaw))
    return out
  return fn


def local_bases(state, boxes, offsets=LOCAL_OFFSETS):
  c, s = math.cos(state.yaw), math.sin(state.yaw)
  bases = []
  for dx, dy, dyaw in offsets:
    x = state.x + c * dx - s * dy
    y = state.y + s * dx + c * dy
    yaw = planning.wrap(state.yaw + dyaw)
    if scenes.footprint_free(boxes, x, y, yaw, 0.02):
      bases.append((x, y, yaw))
  return lambda *_: bases


def body_feasible(backend, base, h, ee_b, R_ee_b, holding=False, obj_offset=None, tol=0.01):
  s = _state_at(backend, base, h)
  if np.linalg.norm(backend.clip_ee(ee_b, s, R_ee_b) - ee_b) > tol:
    return False
  s.ee_b, s.R_ee_b = np.asarray(ee_b, np.float64), R_ee_b
  s.holding = holding
  if holding:
    p_w = backend.to_world(s.ee_b, s)
    s.obj_center = p_w + backend.R_body(s) @ R_ee_b @ obj_offset
  return not backend.arm_collides(s, PLAN_MARGIN)


def _interp_R(R0, R1, t):
  return geo.axis_angle(t * geo.log_rot(R1 @ R0.T)) @ R0


def _leg_ok(backend, base, h, p0, p1, R0, R1, holding=False, offset=None, n=5):
  for t in np.linspace(0, 1, n)[1:]:
    if not body_feasible(backend, base, h, p0 * (1 - t) + p1 * t, _interp_R(R0, R1, t),
                         holding, offset):
      return False
  return True


def transit_waypoints(backend, plan, start_b, R_start_b, holding=False, obj_offset=None):
  """Collision-free TCP route start -> (raise) -> above pre -> pre -> goal,
  plus goal -> post -> carry for picks. Returns world waypoints before pre_w,
  or None if no route among the tried clearances is feasible."""
  s = _state_at(backend, plan.base, plan.h)
  Rb = backend.R_body(s)
  bp = backend.base_pos(s)
  to_b = lambda p: Rb.T @ (np.asarray(p) - bp)
  R_plan_b = Rb.T @ plan.R_w
  pre_b, goal_b, post_b = to_b(plan.pre_w), to_b(plan.goal_w), to_b(plan.post_w)
  start_b = np.asarray(start_b, np.float64)
  if plan.kind == 'pick':
    offset = plan.R_w.T @ (backend.state.obj_center - plan.goal_w)
    carry_b = backend.shoulder_b + kb.CARRY_REL
    if not _leg_ok(backend, plan.base, plan.h, goal_b, post_b, R_plan_b, R_plan_b, True, offset):
      return None
    found = False
    for z in (max(post_b[2], carry_b[2]), post_b[2] + 0.20, post_b[2] + 0.10):
      up_b = np.array([post_b[0], post_b[1], z])
      if not _leg_ok(backend, plan.base, plan.h, post_b, up_b, R_plan_b, R_plan_b, True, offset, 8):
        continue
      # A: rotate while retracting. B: rotate in place, then retract.
      if _leg_ok(backend, plan.base, plan.h, up_b, carry_b, R_plan_b, kb.R_DEFAULT, True, offset, 8):
        plan.raise_z, plan.turn_first, found = float(z), False, True
        break
      if (_leg_ok(backend, plan.base, plan.h, up_b, up_b, R_plan_b, kb.R_DEFAULT, True, offset, 6)
          and _leg_ok(backend, plan.base, plan.h, up_b, carry_b, kb.R_DEFAULT, kb.R_DEFAULT, True,
                      offset, 8)):
        plan.raise_z, plan.turn_first, found = float(z), True, True
        break
    if not found:
      return None
  for over in (0.10, 0.05, 0.0):
    w1 = start_b.copy()
    w1[2] = max(start_b[2], pre_b[2] + over)
    w2 = pre_b + [0.0, 0.0, over]
    legs = [(start_b, w1, R_start_b, R_start_b), (w1, w2, R_start_b, R_plan_b),
            (w2, pre_b, R_plan_b, R_plan_b), (pre_b, goal_b, R_plan_b, R_plan_b)]
    if all(_leg_ok(backend, plan.base, plan.h, a, b, Ra, Rb_, holding, obj_offset)
           for a, b, Ra, Rb_ in legs):
      way = [w for w in (w1, w2) if np.linalg.norm(w - pre_b) > 0.02]
      return tuple(bp + Rb @ w for w in way)
  return None


# ---------------------------------------------------------------------------
# Low-level action builders
# ---------------------------------------------------------------------------

def nav_action(vx=0.0, vy=0.0, wz=0.0, h=H_MAX):
  vec = I.unscale_action({'vx': vx, 'vy': vy, 'wz': wz, 'height': h}, I.NAV_LAYOUT, I.NAV_DIM)
  return _dict(I.Skill.NAV, vec)


def manip_action(skill, dee_pos, dee_rot, grip, base_adj=(0, 0, 0), h=H_MAX):
  base_adj = np.asarray(base_adj, np.float64).copy()
  base_adj[2] /= 2.0          # the env doubles wz
  vec = I.unscale_action({'dee_pos': dee_pos, 'dee_rot': dee_rot, 'grip': grip,
                          'base_adj': base_adj, 'height': h}, I.MANIP_LAYOUT, I.MANIP_DIM)
  return _dict(skill, vec)


def _dict(skill, vec):
  act = {'skill': np.int32(skill), 'nav': np.zeros(I.NAV_DIM, np.float32),
         'pick': np.zeros(I.MANIP_DIM, np.float32), 'place': np.zeros(I.MANIP_DIM, np.float32)}
  act[I.SKILL_ACTION_KEY[I.Skill(int(skill))]] = vec.astype(np.float32)
  return I.mask_inactive_np(act)


class ManipController:
  """Closed-loop tracking of a ManipPlan: waypoints -> pre -> goal -> toggle
  gripper -> post (-> retract to the carry pose for picks)."""

  def __init__(self, plan, cap=0.15):
    self.plan = plan
    self.cap = cap
    self.subs = [f'way{i}' for i in range(len(plan.way_w))] + ['pre', 'goal', 'act', 'post']
    if plan.kind == 'pick':
      self.subs += ['raise', 'turn', 'retract'] if plan.turn_first else ['raise', 'retract']
    self.i = 0
    self.t = 0
    self.why = ''
    self.limit = None

  @property
  def sub(self):
    return self.subs[self.i]

  def targets(self, backend=None):
    p, sub = self.plan, self.sub
    if sub.startswith('way'):
      return p.way_w[int(sub[3:])]
    if sub == 'retract':
      return backend.to_world(backend.shoulder_b + kb.CARRY_REL)
    if sub in ('raise', 'turn'):
      post_b = backend.R_body().T @ (p.post_w - backend.base_pos())
      return backend.to_world(np.array([post_b[0], post_b[1], p.raise_z]))
    return {'pre': p.pre_w, 'goal': p.goal_w, 'act': p.goal_w, 'post': p.post_w}[sub]

  def _advance(self):
    self.i += 1
    self.t = 0
    self.limit = None

  def step(self, backend):
    """Returns (action, status) with status in running | done | failed."""
    p, s = self.plan, backend.state
    self.t += 1
    tcp_w, R_cur_w = backend.ee_world()
    err = float(np.linalg.norm(self.targets(backend) - tcp_w))
    rot_err = float(np.linalg.norm(geo.log_rot(p.R_w @ R_cur_w.T)))
    if self.limit is None:   # step budget from the distance at entry (0.05 m/step max)
      self.limit = 6 + int(math.ceil(err / 0.035))
    sub = self.sub
    if sub.startswith('way'):
      if err < 0.03:
        self._advance()
      elif self.t > self.limit:
        self.why = 'way_timeout'
        return None, 'failed'
    elif sub == 'pre':
      if err < 0.015 and rot_err < 0.08:
        self._advance()
      elif self.t > self.limit + 6:
        self.why = 'pre_timeout'
        return None, 'failed'
    elif sub == 'goal':
      if err < (0.008 if p.kind == 'place' and self.t < 6 else 0.012):
        self._advance()
      elif self.t > self.limit + 4:
        self.why = 'goal_timeout'
        return None, 'failed'
    elif sub == 'act':
      # Hold still; the gripper toggles (after one settling step for places),
      # then check the outcome once the arm latency has passed.
      if self.t > (4 if p.kind == 'place' else 2):
        ok = s.holding if p.kind == 'pick' else s.placed
        if not ok:
          self.why = 'act_failed'
          return None, 'failed'
        self._advance()
    elif sub == 'post':
      if err < 0.02 or self.t > 6:
        if self.i == len(self.subs) - 1:
          return None, 'done'
        self._advance()
    elif sub == 'raise':
      if err < 0.03 or self.t > self.limit:
        self._advance()
    elif sub == 'turn':
      turn_err = float(np.linalg.norm(geo.log_rot(
          backend.R_body() @ kb.R_DEFAULT @ R_cur_w.T)))
      if turn_err < 0.1 or self.t > 12:
        self._advance()
    elif sub == 'retract':
      if err < 0.03 or self.t > self.limit:
        return None, 'done'
    sub = self.sub
    tgt = self.targets(backend)
    Rb = backend.R_body()
    tcp_b = Rb.T @ (tgt - backend.base_pos())
    speed = 0.03 if sub == 'goal' else 0.05
    # Gain < 1 so that the loop also converges with one step of arm latency.
    dee = ARM_GAIN * (tcp_b - s.ee_b)
    n = float(np.linalg.norm(dee))
    if n > speed:   # scale, not per-axis clip, so the TCP path stays straight
      dee *= speed / n
    # During 'act' the loop keeps holding the goal pose (closed loop), which
    # cancels latent commands and noise while the gripper toggles.
    R_tgt_b = kb.R_DEFAULT if sub in ('turn', 'retract') else Rb.T @ p.R_w
    drot = ARM_GAIN * geo.log_rot(R_tgt_b @ s.R_ee_b.T)
    n = float(np.linalg.norm(drot))
    if n > 0.3:
      drot *= 0.3 / n
    # Base correction towards the planned base pose.
    c, sn = math.cos(s.yaw), math.sin(s.yaw)
    dxy = p.base[:2] - [s.x, s.y]
    bx, by = c * dxy[0] + sn * dxy[1], -sn * dxy[0] + c * dxy[1]
    dyaw = planning.wrap(p.base[2] - s.yaw)
    base_adj = np.clip([1.5 * bx, 1.5 * by, 1.5 * dyaw], [-self.cap, -self.cap, -0.3],
                       [self.cap, self.cap, 0.3])
    if p.kind == 'pick':
      grip = 0.0 if sub in ('act', 'post', 'raise', 'turn', 'retract') else 1.0
      skill = I.Skill.PICK
    else:
      grip = 1.0 if (sub == 'act' and self.t > 1) or sub == 'post' else 0.0
      skill = I.Skill.PLACE
    return manip_action(skill, dee, drot, grip, base_adj, p.h), 'running'


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class ScriptedExpert:

  def __init__(self, cfg=None, rng=None):
    self.cfg = cfg or ExpertConfig()
    self.rng = rng or np.random.default_rng(0)

  def reset(self, scene, rng, backend):
    self.scene, self.rng, self.backend = scene, rng, backend
    c = self.cfg
    self.dart = c.diversity and rng.random() < c.p_dart
    self.mode = 'plan'
    self.plan = None
    self.path = None
    self.ctrl = None
    self.n_plans = 0
    self.n_pick_plans = 0
    self.excluded = set()
    self.explore_left = 0
    self.explore_skill = None
    self.early_at = None
    self.timer = 0
    self.progress = []
    self.planner = planning.GridPlanner(scene, 0.45)
    self.planner_tight = planning.GridPlanner(scene, scenes.NAV_INFLATE)
    self.stats = collections.Counter(bad_pose=0, early_switch=0, explore_steps=0, replans=0, failures=0)

  # ------------------------------------------------------------- planning

  def _plan(self):
    s = self.backend.state
    self.n_plans += 1
    if self.n_plans > 1:
      self.stats['replans'] += 1
    if s.holding:
      jitter = self.rng.uniform(-0.05, 0.05, 2)
      cands = place_candidates(self.backend, self.scene,
                               lambda sd, c: radial_bases(c[:2], self.scene.boxes)(sd), jitter)
    else:
      self.n_pick_plans += 1
      cands = pick_candidates(self.backend, self.scene,
                              radial_bases(s.obj_center[:2], self.scene.boxes), self.excluded)
      if not cands and self.excluded:
        self.excluded = set()
        cands = pick_candidates(self.backend, self.scene,
                                radial_bases(s.obj_center[:2], self.scene.boxes))
    if not cands:
      return False
    start = np.array([s.x, s.y])
    best = None
    for planner in (self.planner, self.planner_tight):
      planner.solve_from(start)
      scored = []
      for c in cands:
        stage = self._staging(c, planner)
        if stage is None:
          continue
        d = planner.dist_to(stage)
        if np.isfinite(d):
          noise = self.rng.uniform(0, 0.3) if self.cfg.diversity else 0.0
          scored.append((d + c.score + noise, c, stage))
      scored.sort(key=lambda x: x[0])
      for _, c, stage in scored[:25]:
        way = self._transit(c)
        if way is not None:
          c.way_w = way
          best = (0.0, c, stage)
          break
      if best is not None:
        break
    if best is None:
      return False
    _, plan, stage = best
    if not s.holding and self.cfg.diversity and self.n_pick_plans == 1:
      if self.rng.random() < self.cfg.p_bad_pose:
        plan = self._make_bad(plan, cands)
        plan.way_w = self._transit(plan) or plan.way_w
        bad_stage = self._staging(plan, planner)
        stage = bad_stage if bad_stage is not None else stage
      elif self.rng.random() < self.cfg.p_early_switch:
        self.early_at = self.rng.uniform(0.4, 1.2)
        self.stats['early_switch'] += 1
    self.plan = plan
    path = planner.path_to(stage)
    self.path = path if path is not None else np.array([start, stage])
    self.mode = 'nav'
    self.timer = 0
    self.progress = []
    return True

  def _transit(self, plan):
    s = self.backend.state
    if s.holding:
      return transit_waypoints(self.backend, plan, self.backend.shoulder_b + kb.CARRY_REL,
                               kb.R_DEFAULT, True, s.obj_offset)
    return transit_waypoints(self.backend, plan, self.backend.shoulder_b + kb.STOW_REL, kb.R_STOW)

  def _staging(self, plan, planner):
    for sweep in (True, False):
      for back in STAGE_BACK:
        st = plan.base[:2] + back * plan.stand_dir
        if not (planner.free(st) and self._line_free(st, plan.base[:2], plan.base[2])):
          continue
        if sweep and not all(scenes.footprint_free(self.scene.boxes, st[0], st[1], yaw, 0.02)
                             for yaw in np.linspace(-math.pi, math.pi, 13)):
          continue
        return st
    if planner.free(plan.base[:2]):
      return plan.base[:2].copy()
    return None

  def _line_free(self, a, b, yaw):
    for t in np.linspace(0, 1, 6):
      p = a * (1 - t) + b * t
      if not scenes.footprint_free(self.scene.boxes, p[0], p[1], yaw, 0.02):
        return False
    return True

  def _make_bad(self, plan, cands):
    self.stats['bad_pose'] += 1
    kind = self.rng.integers(4)
    plan = copy.deepcopy(plan)
    if kind == 0:                        # wrong side: approach from a blocked bin
      blocked = [i for i in range(scenes.N_DIRS) if not self.scene.approach[i]]
      if blocked:
        i = int(self.rng.choice(blocked))
        d = scenes.APPROACH_DIRS[i][:2]
        plan.stand_dir = d
        base = self.backend.state.obj_center[:2] + 0.7 * d
        plan.base = np.array([*base, math.atan2(-d[1], -d[0])])
        a = np.array([-d[0], -d[1], 0.0])
        plan.R_w = grasp_frame(a, plan.base[2])
        plan.pre_w, plan.goal_w, plan.post_w = pick_targets(self.backend.state.obj_center, a, False)
        plan.dir_index = i
        return plan
    if kind <= 1:                         # too far
      plan.base[:2] += 0.3 * plan.stand_dir
    elif kind == 2:                       # too near
      near = plan.base[:2] - 0.12 * plan.stand_dir
      if scenes.footprint_free(self.scene.boxes, near[0], near[1], plan.base[2], 0.0):
        plan.base[:2] = near
    else:                                 # heading off
      plan.base[2] = planning.wrap(plan.base[2] + self.rng.choice([-0.6, 0.6]))
    return plan

  # ---------------------------------------------------------------- acting

  def act(self, obs=None, priv=None):
    act = self._act()
    act = self._diversify(act)
    return I.mask_inactive_np(act)

  def _act(self):
    s = self.backend.state
    if s.placed:
      return nav_action()
    if self.mode == 'plan':
      if self.n_plans >= self.cfg.max_replans:
        self.mode = 'idle'
        self.stats['fail_max_replans'] += 1
      elif not self._plan():
        self.mode = 'idle'
        self.stats['fail_plan_' + ('place' if s.holding else 'pick')] += 1
    if self.mode == 'idle':
      return nav_action()
    if self.mode == 'backoff':
      self.timer += 1
      if self.timer > 4:
        self.mode = 'plan'
        return nav_action()
      vx, _, _ = self._safe(-0.3, 0.0, 0.0)
      return nav_action(vx=vx)
    if self.mode in ('nav', 'align', 'line'):
      a = self._nav_step()
      if a is not None:
        return a
    if self.mode == 'manip':
      a, status = self.ctrl.step(self.backend)
      if status == 'running':
        return a
      if status == 'done':
        self.mode = 'plan'
        return nav_action()
      self.stats[f'fail_{self.plan.kind}_{self.ctrl.why}'] += 1
      self._on_failure()
      return nav_action()
    return nav_action()

  def _nav_step(self):
    s = self.backend.state
    p = self.plan
    pos = np.array([s.x, s.y])
    self.timer += 1
    remaining = float(np.linalg.norm(p.base[:2] - pos))
    if self.early_at is not None and remaining < self.early_at:
      self.early_at = None
      return self._start_manip()
    if self.mode == 'nav':
      self.progress.append(remaining)
      if len(self.progress) > 30 and self.progress[-30] - remaining < 0.05:
        self.stats['fail_nav_stuck'] += 1
        self._on_failure(nav=True)
        return nav_action()
      vx, wz, rem = planning.pure_pursuit(pos, s.yaw, self.path, lookahead=0.3, vmax=0.5)
      if rem < 0.1:
        self.mode = 'align'
      else:
        vx, wz = self._safe(vx, 0.0, wz)[::2]
        return nav_action(vx=vx, wz=wz)
    if self.mode == 'align':
      err = planning.wrap(p.base[2] - s.yaw)
      if abs(err) < 0.08:
        self.mode = 'line'
      else:
        _, _, wz = self._safe(0.0, 0.0, float(np.clip(2.0 * err, -1, 1)))
        return nav_action(wz=wz, h=H_MAX)
    if self.mode == 'line':
      c, sn = math.cos(s.yaw), math.sin(s.yaw)
      d = p.base[:2] - pos
      fx, fy = c * d[0] + sn * d[1], -sn * d[0] + c * d[1]
      err = planning.wrap(p.base[2] - s.yaw)
      # Loose tolerance: the manipulation controller keeps correcting the base.
      if math.hypot(fx, fy) < 0.05 and abs(err) < 0.1:
        return self._start_manip()
      if self.timer > 80:
        self.stats['fail_line_timeout'] += 1
        self._on_failure(nav=True)
        return nav_action()
      return nav_action(vx=float(np.clip(1.2 * fx, -0.3, 0.3)), vy=float(np.clip(1.2 * fy, -0.2, 0.2)),
                        wz=float(np.clip(2.0 * err, -0.6, 0.6)), h=p.h)
    return None

  def _safe(self, vx, vy, wz, horizon=0.3):
    """Predictive footprint check of a base command; degrades to turning
    only, then to backing up, then to stopping."""
    s = self.backend.state
    for cand in ((vx, vy, wz), (0.0, 0.0, wz), (-0.15, 0.0, 0.0), (0.0, 0.0, 0.0)):
      cvx, cvy, cwz = cand
      ok = True
      for frac in (0.5, 1.0):
        t = horizon * frac
        yaw = s.yaw + cwz * t
        c, sn = math.cos(s.yaw + 0.5 * cwz * t), math.sin(s.yaw + 0.5 * cwz * t)
        x = s.x + (c * cvx - sn * cvy) * t
        y = s.y + (sn * cvx + c * cvy) * t
        if not scenes.footprint_free(self.scene.boxes, x, y, yaw, 0.02):
          ok = False
          break
      if ok:
        return cand
    return (0.0, 0.0, 0.0)

  def _start_manip(self):
    self.mode = 'manip'
    self.ctrl = ManipController(self.plan)
    a, _ = self.ctrl.step(self.backend)
    return a if a is not None else nav_action()

  def _on_failure(self, nav=False):
    self.stats['failures'] += 1
    if not nav and self.plan is not None and self.plan.kind == 'pick':
      if self.rng.random() < self.cfg.p_other_side or not self.cfg.diversity:
        self.excluded.add(self.plan.dir_index)
    s = self.backend.state
    back = np.array([s.x - 0.3 * math.cos(s.yaw), s.y - 0.3 * math.sin(s.yaw)])
    if scenes.footprint_free(self.scene.boxes, back[0], back[1], s.yaw, 0.02):
      self.mode, self.timer = 'backoff', 0
    else:
      self.mode = 'plan'

  def _diversify(self, act):
    """DART noise and exploration segments. Noise never touches the body
    height or gripper dimensions (height jitter moves a held object by
    centimetres; gripper noise toggles grasps), exploration never switches
    skills while holding, and noisy NAV commands go through the footprint
    safety filter."""
    c = self.cfg
    if not c.diversity:
      return act
    key = I.SKILL_ACTION_KEY[I.Skill(int(act['skill']))]
    holding = self.backend.state.holding
    if self.explore_left == 0 and self.rng.random() < c.explore_frac / 10.0:
      self.explore_left = int(self.rng.integers(5, 16))
      self.explore_skill = None
      if not holding and self.rng.random() < 0.3:
        self.explore_skill = int(self.rng.integers(3))
    sigma = 0.0
    if self.explore_left > 0:
      self.explore_left -= 1
      self.stats['explore_steps'] += 1
      if self.explore_skill is not None:
        act = _dict(self.explore_skill, act[I.SKILL_ACTION_KEY[I.Skill(self.explore_skill)]])
        key = I.SKILL_ACTION_KEY[I.Skill(self.explore_skill)]
      sigma = EXPLORE_SIGMA
      if self.explore_left == 0 and self.mode != 'idle':
        self.mode = 'plan'
        self.n_plans = max(0, self.n_plans - 1)
    elif self.dart:
      sigma = c.dart_sigma
    if sigma == 0.0:
      return act
    noise = self.rng.normal(0, sigma, act[key].shape)
    layout = I.NAV_LAYOUT if key == 'nav' else I.MANIP_LAYOUT
    for name in ('height', 'grip'):
      if name in layout:
        noise[layout[name][0]] = 0.0
    vec = np.clip(act[key] + noise, -1, 1).astype(np.float32)
    if key == 'nav':
      ph = I.scale_action(vec, I.NAV_LAYOUT)
      vx, vy, wz = self._safe(float(ph['vx'][0]), float(ph['vy'][0]), float(ph['wz'][0]))
      vec = I.unscale_action({'vx': vx, 'vy': vy, 'wz': wz, 'height': ph['height']},
                             I.NAV_LAYOUT, I.NAV_DIM)
    return dict(act, **{key: vec})
