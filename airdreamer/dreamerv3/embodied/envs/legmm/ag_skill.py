"""Fixed-base grasp skill surrogate for legmm_ag (brief §4, §5, §7). Kinematic stand-in, NOT a real success rate.

From the settled real pose, at the fixed standing height, with the arm at STOW (shoulder + 0.36 m), the
skill replans pre / goal / lift / hold for this exact base pose (wrist twist within +-15 deg, the same
check as the geometry audit: STOW -> raise -> above pre -> pre -> goal -> close -> lift -> hold), then
executes it closed loop on the kin backend. The base cannot move: every command built here has a zero
base velocity and the body height held at its current value, the controller has no base term, and the
skill refuses to finish if the base pose changed. A plan alone is never a success; success requires
execution: lifted >= 5 cm, held >= 2 s, no drop, no arm/body collision, within 15 s. kin's own stochastic
grasp check (GRASP_P = 0.92) stays available through the backend config value grasp_p.

The planning and path checks are copied from tools/legmm_v4_geometry_audit.py (worktree
legmm-v4-geometry); nothing is imported from tools/.
"""
import collections
import dataclasses
import math
import time

import numpy as np

from . import expert as ex
from . import geometry as geo
from . import interfaces as I
from . import kin_backend as kb
from . import scenes as sc
from .ag_shaping import beta0, width_angle
from . import ag_contract as contract

GRIP_LIMIT = kb.GRIP_MAX_WIDTH - 0.005     # 7.5 cm, as kin._try_grasp and scenes.graspable
BODY_MARGIN = contract.BODY_MARGIN                         # footprint clearance of a stance (expert.radial_bases)
REASONS = ('body', 'width', 'reach', 'limit', 'self', 'table', 'path')
STANCE_RHOS = (0.55, 0.50, 0.60, 0.45, 0.65, 0.70)
STANCE_BEARINGS = (0, 15, -15, 30, -30, 45, -45)


def skill_cfg(cfg=None):
  c = dict(twist_deg=15.0, lift=0.06, hold_s=2.0, max_s=15.0, grasp_p=1.0, tilts=tuple(ex.TILTS))
  c.update(dict(cfg or {}))
  return c


def twist_set(limit_deg):
  """Wrist twists in 5 deg steps, ascending |theta| (the planner stops at the first feasible one)."""
  limit = int(round(limit_deg))
  return tuple(sorted(range(-limit, limit + 1, 5), key=abs))


# ---------------------------------------------------------------------------------------------
# Planning (copied from the geometry audit, parameterised by the actual height)
# ---------------------------------------------------------------------------------------------

def closing_width(half, R, shape, heading):
  """Opening needed for a horizontal closing axis perpendicular to the approach heading."""
  c = np.array([-math.sin(heading), math.cos(heading), 0.0])
  if shape == 'cyl':
    return 2 * float(half[0])
  return 2 * float(half[0] * abs(c @ R[:, 0]) + half[1] * abs(c @ R[:, 1]))


def approach_axis(heading, tilt):
  return np.array([math.cos(heading) * math.cos(tilt), math.sin(heading) * math.cos(tilt), -math.sin(tilt)])


def shoulder_world(backend, base):
  x, y, yaw = base
  sh = backend.shoulder_b
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([x + c * sh[0] - s * sh[1], y + s * sh[0] + c * sh[1]])


def _dense_clear(backend, s):
  pts = backend.arm_points(s, n=8)
  return not geo.points_in_boxes(pts, backend._static, kb.LINK_RADIUS + ex.PLAN_MARGIN).any()


def _pose_reason(backend, base, h, tcp_w, R_w):
  """'' if the TCP pose is feasible from this exact base, else why not (audit.pose_reason, dense)."""
  s = ex._state_at(backend, base, h)
  Rb = backend.R_body(s)
  R_ee_b = Rb.T @ R_w
  ee_b = Rb.T @ (np.asarray(tcp_w) - backend.base_pos(s))
  if np.linalg.norm(backend.clip_ee(ee_b, s, R_ee_b) - ee_b) > 0.005:
    return 'reach'
  s.ee_b, s.R_ee_b = ee_b, R_ee_b
  s.holding = False
  if not backend.arm_collides(s, ex.PLAN_MARGIN):
    return '' if _dense_clear(backend, s) else 'table'
  if not backend.joint_ok(ee_b, R_ee_b):
    return 'limit'
  pts = backend.arm_points(s)
  if geo.points_in_boxes(pts, backend._static, kb.LINK_RADIUS + ex.PLAN_MARGIN).any():
    return 'table'
  return 'self'


def _leg_ok(backend, base, h, p0, p1, R0, R1, holding=False, offset=None, n=5):
  for t in np.linspace(0, 1, n)[1:]:
    p, R = p0 * (1 - t) + p1 * t, ex._interp_R(R0, R1, t)
    if not ex.body_feasible(backend, base, h, p, R, holding, offset):
      return False
    if not holding:
      s = ex._state_at(backend, base, h)
      s.ee_b, s.R_ee_b = p, R
      if not _dense_clear(backend, s):
        return False
  return True


@dataclasses.dataclass
class SkillPlan:
  base: tuple              # (x, y, yaw) the plan was made for
  h: float
  theta_deg: float
  tilt: float
  heading: float
  R_b: np.ndarray          # gripper orientation, body frame
  R_start: np.ndarray
  way_b: tuple             # body-frame waypoints before pre
  pre_b: np.ndarray
  goal_b: np.ndarray
  post_b: np.ndarray
  lift: float


def _route(backend, base, h, R_w, goal_w, R_b, R_start, start_b, pre_b, goal_b, post_b):
  """Body-frame waypoints (w1, w2) or None. Mirrors the audit's skill_path_ok."""
  if not ex.body_feasible(backend, base, h, start_b, R_start):
    return None
  s = backend.state
  saved = s.obj_R_rel
  s.obj_R_rel = R_w.T @ s.obj_R
  try:
    offset = R_w.T @ (s.obj_center - goal_w)
    if not _leg_ok(backend, base, h, goal_b, post_b, R_b, R_b, True, offset, 5):
      return None
    for over in (0.10, 0.05, 0.0):
      w1 = start_b.copy()
      w1[2] = max(start_b[2], pre_b[2] + over)
      w2 = pre_b + [0.0, 0.0, over]
      legs = ((start_b, w1, R_start, R_start), (w1, w2, R_start, R_b),
              (w2, pre_b, R_b, R_b), (pre_b, goal_b, R_b, R_b))
      if all(_leg_ok(backend, base, h, a, b, Ra, Rb_) for a, b, Ra, Rb_ in legs):
        return (w1, w2)
    return None
  finally:
    s.obj_R_rel = saved


def plan_skill(backend, cfg=None, base=None, h=None):
  """Plans from a base pose (default: the backend's current pose and height) for the object at its
  current position. Returns (SkillPlan or None, reason of the furthest failure)."""
  cfg = skill_cfg(cfg)
  s = backend.state
  base = (s.x, s.y, s.yaw) if base is None else tuple(float(v) for v in base)
  h = float(s.h if h is None else h)
  obj = backend.scene.obj
  center, R_obj = s.obj_center, s.obj_R
  sh = shoulder_world(backend, base)
  to_obj = math.atan2(center[1] - sh[1], center[0] - sh[0])
  st = ex._state_at(backend, base, h)
  Rb, bp = backend.R_body(st), backend.base_pos(st)
  start_b = np.asarray(s.ee_b, np.float64).copy()
  R_start = np.asarray(s.R_ee_b, np.float64).copy()
  stage = 1
  for theta in twist_set(cfg['twist_deg']):
    heading = to_obj + math.radians(theta)
    if closing_width(obj.half, R_obj, obj.shape, heading) >= GRIP_LIMIT:
      continue
    stage = max(stage, 2)
    for tilt in cfg['tilts']:
      a = approach_axis(heading, tilt)
      R_w = ex.grasp_frame(a, base[2])
      goal = np.asarray(center, np.float64).copy()
      pre = goal - ex.PRE_BACK * a
      post = goal + [0.0, 0.0, cfg['lift']]
      why = _pose_reason(backend, base, h, pre, R_w) or _pose_reason(backend, base, h, goal, R_w)
      if why:
        stage = max(stage, REASONS.index(why))
        continue
      to_b = lambda p: Rb.T @ (np.asarray(p) - bp)
      R_b = Rb.T @ R_w
      pre_b, goal_b, post_b = to_b(pre), to_b(goal), to_b(post)
      way = _route(backend, base, h, R_w, goal, R_b, R_start, start_b, pre_b, goal_b, post_b)
      if way is None:
        stage = max(stage, REASONS.index('path'))
        continue
      return SkillPlan(base, h, float(theta), float(tilt), float(heading), R_b, R_start, way,
                       pre_b, goal_b, post_b, float(cfg['lift'])), ''
  return None, REASONS[stage]


# ---------------------------------------------------------------------------------------------
# Execution (closed loop, base locked)
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class SkillResult:
  success: bool = False
  reason: str = ''            # '' on success
  steps: int = 0              # high-level (0.2 s) steps of execution
  sim_s: float = 0.0
  plan_wall_s: float = 0.0
  wall_s: float = 0.0         # planning + execution wall clock
  theta_deg: float = 0.0
  tilt: float = 0.0
  lift_m: float = 0.0
  hold_s: float = 0.0
  planned: bool = False
  executed: bool = False
  detail: str = ''
  contract: dict = None       # true_state only; diagnostics, never policy input


class SkillExecutor:
  """Closed-loop tracker of a SkillPlan. It never commands the base (zero velocity, constant height)."""

  ARM_GAIN = 0.4          # < 0.5 so that the loop stays damped with one step of arm latency

  def __init__(self, backend, plan, cfg=None):
    self.b, self.plan, self.cfg = backend, plan, skill_cfg(cfg)
    self.subs = [f'way{i}' for i in range(len(plan.way_b))] + ['pre', 'goal', 'act', 'lift', 'hold']
    self.i, self.t, self.limit = 0, 0, None
    self.hold_steps = 0
    self.pose0 = self._pose()

  def _pose(self):
    s = self.b.state
    return (s.x, s.y, s.yaw, s.h)

  def base_fixed(self, tol=1e-9):
    x, y, yaw, h = self._pose()
    x0, y0, yaw0, h0 = self.pose0
    return (abs(x - x0) < tol and abs(y - y0) < tol and abs(h - h0) < tol
            and abs(math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))) < tol)

  @property
  def sub(self):
    return self.subs[self.i]

  def _target(self):
    p, sub = self.plan, self.sub
    if sub.startswith('way'):
      return p.way_b[int(sub[3:])]
    return {'pre': p.pre_b, 'goal': p.goal_b, 'act': p.goal_b, 'lift': p.post_b, 'hold': p.post_b}[sub]

  def _R_target(self):
    return self.plan.R_start if self.sub == 'way0' else self.plan.R_b

  def _advance(self):
    self.i += 1
    self.t = 0
    self.limit = None

  def lifted(self):
    """Height of the lowest object corner above its support surface."""
    s = self.b.state
    return float(self.b.obj_corners(s)[:, 2].min() - self.b.scene.obj.support_z)

  def resting_contact(self):
    """kin flags 'held_bottom' (the grasped object still rests on its support, so the arm cannot push it
    down) as an arm collision. That is object-support contact, not a collision of the arm."""
    s = self.b.state
    return bool(s.holding and self.lifted() <= 1e-6)

  def command(self):
    """The command for the next high-level step: arm only, base velocity zero, height unchanged."""
    s = self.b.state
    tgt = self._target()
    speed = 0.03 if self.sub == 'goal' else 0.05
    gain = 0.25 if self.sub in ('pre', 'goal', 'act', 'lift', 'hold') else self.ARM_GAIN   # gentle near the table
    dee = gain * (tgt - s.ee_b)
    n = float(np.linalg.norm(dee))
    if n > speed:
      dee *= speed / n
    drot = gain * geo.log_rot(self._R_target() @ s.R_ee_b.T)
    n = float(np.linalg.norm(drot))
    if n > 0.3:
      drot *= 0.3 / n
    grip = 0.0 if self.sub in ('act', 'lift', 'hold') else 1.0
    return dict(skill=int(I.Skill.PICK), base=np.zeros(3), height=float(self.pose0[3]),
                dee_pos=np.clip(dee, -0.05, 0.05), dee_rot=np.clip(drot, -0.3, 0.3), grip=grip)

  def run(self):
    b = self.b
    s = b.state
    res = SkillResult(planned=True, executed=True, theta_deg=self.plan.theta_deg, tilt=self.plan.tilt)
    max_steps = int(round(self.cfg['max_s'] / b.hl_dt))
    hold_need = int(math.ceil(self.cfg['hold_s'] / b.hl_dt - 1e-9))
    s.v = np.zeros(3)                                   # the skill starts from rest
    for step in range(1, max_steps + 1):
      s = b.state                                      # kin may replace the state object when it undoes a step
      sub = self.sub
      self.t += 1
      err = float(np.linalg.norm(self._target() - s.ee_b))
      rot_err = float(np.linalg.norm(geo.log_rot(self._R_target() @ s.R_ee_b.T)))
      if self.limit is None:
        self.limit = 6 + int(math.ceil(err / 0.035))
      fail = ''
      if sub.startswith('way'):
        if err < 0.03:
          self._advance()
        elif self.t > self.limit:
          fail = 'way_timeout'
      elif sub == 'pre':
        if err < 0.015 and rot_err < 0.08:
          self._advance()
        elif self.t > self.limit + 6:
          fail = 'pre_timeout'
      elif sub == 'goal':
        if err < 0.012:
          self._advance()
        elif self.t > self.limit + 4:
          fail = 'goal_timeout'
      elif sub == 'act':
        if self.t > 2:
          if not s.holding:
            fail = 'grasp'
          else:
            self._advance()
      elif sub == 'lift':
        if err < 0.008:
          self._advance()
        elif self.t > self.limit + 4:
          fail = 'lift_timeout'
      if not fail:
        ev = b.step(self.command())
        res.steps = step
        if not self.base_fixed():
          raise RuntimeError('base pose changed inside the skill')
        if ev['collision'] or (ev['arm_collision'] and not self.resting_contact()):
          fail = 'collision'
        elif ev['dropped']:
          fail = 'drop'
        elif ev['failed_grasp']:
          fail = 'grasp'
        elif sub == 'hold':
          if not s.holding:
            fail = 'lost'
          elif self.lifted() < 0.05 - 1e-6:
            fail = 'lift_short'
          else:
            self.hold_steps += 1
            if self.hold_steps >= hold_need:
              res.success = True
              res.lift_m = self.lifted()
              res.hold_s = self.hold_steps * b.hl_dt
              break
      if fail:
        res.reason = fail
        break
    else:
      res.reason = 'timeout'
    res.sim_s = res.steps * b.hl_dt
    res.lift_m = max(res.lift_m, self.lifted()) if b.state.holding else res.lift_m
    res.hold_s = self.hold_steps * b.hl_dt
    return res


def run_skill(backend, cfg=None):
  """Plan from the current pose, then execute. Returns a SkillResult; a plan without execution is a failure."""
  cfg = skill_cfg(cfg)
  t0 = time.perf_counter()
  rep = None
  if cfg.get('contract', 'legacy') == 'true_state':
    rep = contract.report(backend, cfg=cfg.get('contract_gate'))
    if not rep['ok']:
      return SkillResult(False, 'contract_violation', contract=rep, wall_s=time.perf_counter() - t0)
  plan, why = plan_skill(backend, cfg)
  t1 = time.perf_counter()
  if plan is None:
    return SkillResult(False, 'no_plan', plan_wall_s=t1 - t0, wall_s=t1 - t0, detail=why, contract=rep)
  res = SkillExecutor(backend, plan, cfg).run()
  res.contract = rep
  res.plan_wall_s, res.wall_s = t1 - t0, time.perf_counter() - t0
  return res


# ---------------------------------------------------------------------------------------------
# Solvability of a scene under the fixed-base skill (filters and calibration)
# ---------------------------------------------------------------------------------------------

def stance_pose(backend, obj_xy, rho, az_deg, bearing_deg):
  """Base pose with the shoulder at rho from the object along azimuth az and the object at the given
  bearing in the body frame (geometry audit's `stance`)."""
  az = math.radians(az_deg)
  sx, sy = obj_xy[0] + rho * math.cos(az), obj_xy[1] + rho * math.sin(az)
  yaw = az + math.pi - math.radians(bearing_deg)
  dx, dy = backend.shoulder_b[:2]
  x = sx - math.cos(yaw) * dx + math.sin(yaw) * dy
  y = sy - math.sin(yaw) * dx - math.cos(yaw) * dy
  return np.array([x, y, math.atan2(math.sin(yaw), math.cos(yaw))])


def stance_candidates(backend, scene, twist_deg=15.0, az_step=5, rhos=STANCE_RHOS,
                      bearings=STANCE_BEARINGS, margin=0.0, dist_max=0.70):
  """Stances inside the §4 envelope, most promising first (facing a narrow face, mid range, head-on).
  Flat boxes skip azimuths whose heading cannot fit the opening for any twist (exact prefilter).
  margin > 0 shrinks the envelope (rho by margin metres, bearing by 300 * margin degrees): stances on the
  gate boundary cannot be reached to the millimetre, so the margin counts the 'robustly solvable' scenes."""
  rhos = tuple(dict.fromkeys(min(max(r, 0.45 + margin), dist_max - margin) for r in rhos))
  blim = 45.0 - 300.0 * margin
  bearings = tuple(dict.fromkeys(int(math.copysign(min(abs(b), blim), b)) if b else 0 for b in bearings))
  obj = scene.obj
  lim = None
  if obj.shape == 'flat':
    lim = beta0(obj.half) + math.radians(twist_deg) + math.radians(1.0)
  azs = np.arange(0, 360, az_step)
  keyed = []
  for az in azs:
    heading = math.radians(az) + math.pi
    ang = width_angle(obj, heading) if obj.shape == 'flat' else 0.0
    if lim is not None and ang > lim:
      continue
    keyed.append((ang, int(az)))
  keyed.sort()
  for _, az in keyed:
    for bearing in bearings:
      for rho in rhos:
        yield rho, az, bearing, stance_pose(backend, scene.obj.center[:2], rho, az, bearing)


def inside_room(backend, base, margin=0.0):
  """The footprint lies inside the room (stances behind a room wall are not stances)."""
  x, y, yaw = base
  hx, hy = backend.foot_half
  c, s = math.cos(yaw), math.sin(yaw)
  ex = abs(c) * hx + abs(s) * hy
  ey = abs(s) * hx + abs(c) * hy
  W, L = backend.scene.room
  return bool(abs(x) + ex <= W / 2 - margin and abs(y) + ey <= L / 2 - margin)


def feasible_stances(backend, scene, cfg=None, first=False, az_step=5, margin=0.0):
  """List of dicts (base, rho, az, bearing, theta, tilt) of stances with a skill plan. The backend must
  hold the scene (backend.reset(scene, rng)); the robot pose in its state is not used."""
  cfg = skill_cfg(cfg)
  out = []
  options = {}
  true_state = cfg.get('contract', 'legacy') == 'true_state'
  if true_state:
    options = dict(rhos=contract.radius_grid(scene.obj.support_z),
                   dist_max=contract.reach_limit(scene.obj.support_z))
  for rho, az, bearing, base in stance_candidates(backend, scene, cfg['twist_deg'], az_step, margin=margin, **options):
    if not inside_room(backend, base) or not sc.footprint_free(
        backend._static, base[0], base[1], base[2], BODY_MARGIN):
      continue
    if true_state and not contract.report(backend, base, cfg.get('contract_gate'))['ok']:
      continue
    plan, _ = plan_skill(backend, cfg, base=base, h=backend.state.h)
    if plan is not None:
      out.append(dict(base=base, rho=rho, az=az, bearing=bearing, theta=plan.theta_deg, tilt=plan.tilt))
      if first:
        break
  return out


def solvable_scene(backend, scene, cfg=None, rng=None, margin=0.0):
  """True iff some stance in the §4 envelope (+-15 deg twist, fixed base and height) has a skill plan."""
  backend.reset(scene, rng or np.random.default_rng(0))
  return bool(feasible_stances(backend, scene, cfg, first=True, margin=margin))
