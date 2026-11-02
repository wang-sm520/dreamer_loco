"""Kinematic stand-in backend (brief §6.2): pure numpy, no physics engine.

Base: planar pose (x, y, yaw) plus body height; velocities and height track
their commands with first-order lags (pitch is not commanded and stays 0).
Arm: AIRBOT Play chain. The TCP pose is commanded in the body frame; the wrist
center sits TOOL behind the TCP along the approach axis and is clipped to the
shoulder's reach shell; a two-link vertical-plane IK places the elbow.
Depth: vectorized ray casting against boxes, vertical cylinders, the floor,
and (for the head camera) the arm and the wrist camera stand.
"""
import copy
import dataclasses
import math

import numpy as np

from . import geometry as geo
from . import interfaces as I
from . import scenes

# AIRBOT Play (~/wang-sm/depth-mani/airbot_rl/assets/airbot_play.xml, which
# matches the DISCOVERSE model plus the wrist camera stand):
SHOULDER_Z = 0.1172          # arm_base -> joint2 axis
UPPER = 0.27009              # joint2 -> joint3
FORE = 0.29015               # joint3 -> wrist (joint4/5)
TOOL = 0.241                 # wrist (joint5) -> TCP between the finger pads
GRIP_MAX_WIDTH = 0.08        # two fingers x 0.04 m stroke
LINK_RADIUS = 0.03
GRIP_RATE = 0.5              # normalized opening change per high-level step
ARM_SPEED_NAV = 0.05         # m per step when NAV moves the arm to stow/carry
ARM_ROT_NAV = 0.3            # rad per step
GRASP_TOL = 0.03             # brief §6.2
GRASP_ANGLE = math.radians(35.0)
GRASP_P = 0.92
GRASP_ATTEMPT_RADIUS = 0.15  # closing farther than this from the object is not an attempt
PLACE_TOL = 0.03
KIN_SUBSTEPS = 8             # base integration substeps per high-level step
# A2 base_link mesh bounds (unitree_a2/xmls/assets/base_link.STL), incl. head.
BODY_BOX = np.array([-0.33, -0.145, -0.08, 0.39, 0.145, 0.13])
REACH_MIN = 0.08             # min shoulder-to-wrist distance

# TCP poses relative to the shoulder (joint2), body frame.
STOW_REL = np.array([0.33, 0.0, 0.0])      # folded, gripper forward over the head
CARRY_REL = np.array([0.30, 0.0, 0.30])      # high: the object clears table tops while turning
R_STOW = np.eye(3)                          # approach axis forward, horizontal
R_DEFAULT = geo.rot_pitch(math.radians(-30.0))   # carry: 30 deg down


@dataclasses.dataclass
class DomainRand:
  vel_scale: np.ndarray = dataclasses.field(default_factory=lambda: np.ones(3))
  arm_latency: int = 0
  head_extr: tuple = (np.zeros(3), np.eye(3))
  wrist_extr: tuple = (np.zeros(3), np.eye(3))
  noise_a: tuple = (0.002, 0.002)       # head, wrist
  hole_frac: tuple = (0.0, 0.0)
  friction: float = 1.0


@dataclasses.dataclass
class KinState:
  x: float
  y: float
  yaw: float
  h: float
  pitch: float               # not commanded; kept for the proprio layout
  v: np.ndarray               # actual body vx, vy, wz
  ee_b: np.ndarray            # TCP (between the finger pads), body frame
  R_ee_b: np.ndarray          # gripper orientation, body frame (col 0 = approach axis)
  grip: float                 # actual opening 0..1
  grip_closed: bool           # hysteresis state of the command
  holding: bool = False
  obj_center: np.ndarray = None
  obj_offset: np.ndarray = None   # object center in the gripper frame while held
  placed: bool = False
  n_failed: int = 0
  picked: bool = False
  arm_queue: list = dataclasses.field(default_factory=list)


def make_dr(cfg, rng):
  dr = DomainRand()
  dcfg = cfg.get('domain_rand', {})
  if not dcfg.get('enabled', False):
    return dr
  err = float(dcfg.get('base_vel_tracking_err', 0.0))
  dr.vel_scale = 1.0 + rng.uniform(-err, err, 3)
  lat = dcfg.get('arm_latency_steps', (0, 0))
  dr.arm_latency = int(rng.integers(lat[0], lat[1] + 1))
  ext = dcfg.get('cam_extrinsic_sigma', {})
  ps, rs = float(ext.get('pos', 0.0)), math.radians(float(ext.get('rot_deg', 0.0)))
  dr.head_extr = (rng.normal(0, ps, 3), geo.axis_angle(rng.normal(0, rs, 3)))
  dr.wrist_extr = (rng.normal(0, ps, 3), geo.axis_angle(rng.normal(0, rs, 3)))
  na = dcfg.get('cam_noise_a', (0.001, 0.004))
  dr.noise_a = (float(rng.uniform(*na)), float(rng.uniform(*na)))
  hf = dcfg.get('cam_hole_frac', (0.0, 0.03))
  dr.hole_frac = (float(rng.uniform(*hf)), float(rng.uniform(*hf)))
  fr = dcfg.get('object_friction', (1.0, 1.0))
  dr.friction = float(rng.uniform(*fr))   # recorded only; the kin backend has no contact model
  return dr


def _cam_model(ccfg):
  return I.D435iModel(min_z=float(ccfg.get('min_z', 0.28)), max_z=float(ccfg.get('max_z', 6.0)))


class KinBackend:

  def __init__(self, cfg):
    self.cfg = cfg
    base = cfg.get('base', {})
    self.vel_tau = float(base.get('vel_tau', 0.15))
    self.posture_tau = float(base.get('posture_tau', 0.4))
    self.foot_half = tuple(0.5 * float(x) for x in base.get('footprint', (0.82, 0.44)))
    arm = cfg.get('arm', {})
    mount = np.asarray(arm.get('mount_xyz', (0.03, 0.0, 0.095)), np.float64)
    self.shoulder_b = mount + [0.0, 0.0, float(arm.get('shoulder_z', SHOULDER_Z))]
    self.L1 = float(arm.get('upper', UPPER))
    self.L2 = float(arm.get('fore', FORE))
    self.tool = float(arm.get('tool', TOOL))
    self.cap = float(cfg.get('manip_base_speed_cap', 0.15))
    self.hl_dt = 1.0 / float(cfg.get('hl_hz', I.HL_HZ))
    self.cams = {}
    for name in ('head', 'wrist'):
      ccfg = cfg.get(f'camera_{name}', {})
      self.cams[name] = dict(
          model=_cam_model(ccfg),
          mount=np.asarray(ccfg.get('mount_xyz', (0, 0, 0)), np.float64),
          R_mount=geo.rot_pitch(math.radians(float(ccfg.get('mount_pitch_deg', 0.0)))),
      )
    hfov = self.cams['head']['model'].hfov_deg
    vfov = self.cams['head']['model'].vfov_deg
    self.rays = geo.camera_rays(I.DEPTH_W, I.DEPTH_H, hfov, vfov)
    self.h_range = I.NAV_LAYOUT['height'][1:]
    self.scene = None
    self.state = None
    self.dr = DomainRand()
    self.rng = np.random.default_rng(0)

  # ------------------------------------------------------------------ setup

  def reset(self, scene, rng, dr=None):
    self.scene = scene
    self.rng = rng
    self.dr = dr or DomainRand()
    x, y, yaw = scene.start
    self.state = KinState(
        x=float(x), y=float(y), yaw=float(yaw), h=I.BASE_HEIGHT_RANGE[1], pitch=0.0,
        v=np.zeros(3), ee_b=self.shoulder_b + STOW_REL, R_ee_b=R_STOW.copy(), grip=1.0, grip_closed=False,
        obj_center=scene.obj.center.copy())
    self._static = scene.boxes
    return self.state

  def snapshot(self):
    return copy.deepcopy((self.state, self.rng.bit_generator.state))

  def restore(self, snap):
    state, rng_state = copy.deepcopy(snap)
    self.state = state
    self.rng.bit_generator.state = rng_state

  # ------------------------------------------------------------- kinematics

  def R_body(self, s=None):
    s = s or self.state
    return geo.rot_z(s.yaw) @ geo.rot_pitch(s.pitch)

  def base_pos(self, s=None):
    s = s or self.state
    return np.array([s.x, s.y, s.h])

  def to_world(self, p_b, s=None):
    return self.base_pos(s) + self.R_body(s) @ p_b

  def ee_world(self, s=None):
    s = s or self.state
    return self.to_world(s.ee_b, s), self.R_body(s) @ s.R_ee_b

  def wrist_b(self, ee_b, R_ee_b):
    return ee_b - self.tool * R_ee_b[:, 0]

  def elbow_b(self, ee_b, R_ee_b=None):
    """Elbow position (body frame) and (yaw, shoulder pitch, elbow angle)."""
    R_ee_b = self.state.R_ee_b if R_ee_b is None else R_ee_b
    v = self.wrist_b(ee_b, R_ee_b) - self.shoulder_b
    psi = math.atan2(v[1], v[0])
    r = math.hypot(v[0], v[1])
    L1, L2 = self.L1, self.L2
    D = float(np.clip(np.linalg.norm(v), abs(L1 - L2) + 1e-3, L1 + L2 - 1e-6))
    phi = math.atan2(v[2], r)
    beta = math.acos(np.clip((L1 ** 2 + D ** 2 - L2 ** 2) / (2 * L1 * D), -1, 1))
    alpha = phi + beta                       # elbow up
    elbow = self.shoulder_b + L1 * np.array(
        [math.cos(alpha) * math.cos(psi), math.cos(alpha) * math.sin(psi), math.sin(alpha)])
    gamma = math.pi - math.acos(np.clip((L1 ** 2 + L2 ** 2 - D ** 2) / (2 * L1 * L2), -1, 1))
    return elbow, (psi, alpha, gamma)

  def clip_ee(self, ee_b, s, R_ee_b=None):
    """Clips a TCP target so that the wrist stays in the shoulder's reach shell,
    the TCP stays 2 cm above the ground, and wrist/TCP stay out of the body."""
    R_ee_b = s.R_ee_b if R_ee_b is None else R_ee_b
    a = R_ee_b[:, 0]
    lmax = self.L1 + self.L2 - 1e-3
    ee_b = np.asarray(ee_b, np.float64).copy()
    for _ in range(2):
      v = ee_b - self.tool * a - self.shoulder_b
      n = float(np.linalg.norm(v))
      if n > lmax:
        v *= lmax / n
      elif n < REACH_MIN:
        v = v / n * REACH_MIN if n > 1e-6 else np.array([REACH_MIN, 0.0, 0.0])
      ee_b = self.shoulder_b + v + self.tool * a
      for p in (ee_b, ee_b - self.tool * a):
        lo, hi = BODY_BOX[:3] - LINK_RADIUS, BODY_BOX[3:] + LINK_RADIUS
        if np.all(p >= lo) and np.all(p <= hi):
          ee_b[2] += hi[2] - p[2]
      ee_w = self.to_world(ee_b, s)
      if ee_w[2] < 0.02:
        ee_w[2] = 0.02
        ee_b = self.R_body(s).T @ (ee_w - self.base_pos(s))
    return ee_b

  def wrist_cam_b(self, s=None):
    s = s or self.state
    return s.ee_b + s.R_ee_b @ self.cams['wrist']['mount']

  def arm_points(self, s=None, n=3):
    """World points along upper arm, forearm, and wrist-to-TCP, plus the
    fingertip and the wrist camera."""
    s = s or self.state
    elbow, _ = self.elbow_b(s.ee_b, s.R_ee_b)
    wrist = self.wrist_b(s.ee_b, s.R_ee_b)
    ts = np.linspace(0.0, 1.0, n + 1)[1:, None]
    seg = lambda p, q: p[None] * (1 - ts) + q[None] * ts
    tip = s.ee_b + 0.02 * s.R_ee_b[:, 0]
    pts_b = np.concatenate([
        seg(self.shoulder_b, elbow), seg(elbow, wrist), seg(wrist, s.ee_b),
        tip[None], self.wrist_cam_b(s)[None]], 0)
    return self.base_pos(s)[None] + pts_b @ self.R_body(s).T

  def held_points(self, s=None):
    s = s or self.state
    c = s.obj_center
    hz = self.scene.obj.half[2]
    return np.array([c, c - [0, 0, hz - 0.005]])

  def arm_collides(self, s=None, margin=0.0):
    """margin > 0 adds clearance (used by planners to leave room for tracking
    errors); the environment itself always checks with margin 0."""
    s = s or self.state
    pts = self.arm_points(s)
    if geo.points_in_boxes(pts, self._static, LINK_RADIUS + margin).any():
      return True
    # Self collision with the dog body (skip the first upper-arm sample).
    pts_b = (pts - self.base_pos(s)) @ self.R_body(s)
    if geo.points_in_boxes(pts_b[1:], BODY_BOX[None], 0.01 + 3 * margin).any():
      return True
    if s.holding:
      hp = self.held_points(s)
      r = float(self.scene.obj.half[0])
      if geo.points_in_boxes(hp[:1], self._static, r).any():
        return True
      if geo.points_in_boxes(hp[1:], self._static, 0.0).any() or hp[1, 2] < 0.0:
        return True
    return False

  def base_collides(self, s=None):
    s = s or self.state
    return bool(geo.obb_hits_boxes((s.x, s.y), s.yaw, self.foot_half, self._static).any())

  # ------------------------------------------------------------------- step

  def step(self, cmd):
    """cmd: dict with skill (int), base (3,) vx vy wz, height and, for
    manipulation skills, dee_pos (3,), dee_rot (3,), grip (scalar 0..1).

    Returns a dict of events for this high-level step."""
    s = self.state
    ev = dict(collision=False, arm_collision=False, grasp_success=False,
              failed_grasp=False, grasp_attempt=False, place_success=False,
              dropped=False, base_disp=0.0)
    prev = copy.deepcopy(s)
    manip = int(cmd['skill']) != int(I.Skill.NAV)

    # Arm command latency (domain randomization).
    arm_cmd = (np.asarray(cmd.get('dee_pos', np.zeros(3)), np.float64),
               np.asarray(cmd.get('dee_rot', np.zeros(3)), np.float64),
               cmd.get('grip', None), manip)
    s.arm_queue.append(arm_cmd)
    if len(s.arm_queue) > self.dr.arm_latency:
      dee_pos, dee_rot, grip, arm_manip = s.arm_queue.pop(0)
    else:
      dee_pos, dee_rot, grip, arm_manip = np.zeros(3), np.zeros(3), None, manip

    # Base and posture.
    base_cmd = np.asarray(cmd['base'], np.float64) * self.dr.vel_scale
    h_cmd = float(np.clip(cmd['height'], *self.h_range))
    dt = self.hl_dt / KIN_SUBSTEPS
    av = 1 - math.exp(-dt / self.vel_tau)
    ap = 1 - math.exp(-dt / self.posture_tau)
    for _ in range(KIN_SUBSTEPS):
      px, py, pyaw = s.x, s.y, s.yaw
      s.v = s.v + (base_cmd - s.v) * av
      c, sn = math.cos(s.yaw), math.sin(s.yaw)
      s.x += (c * s.v[0] - sn * s.v[1]) * dt
      s.y += (sn * s.v[0] + c * s.v[1]) * dt
      s.yaw = (s.yaw + s.v[2] * dt + math.pi) % (2 * math.pi) - math.pi
      s.h += (h_cmd - s.h) * ap
      if self.base_collides(s):
        s.x, s.y, s.yaw = px, py, pyaw
        s.v = np.zeros(3)
        ev['collision'] = True
        break
    ev['base_disp'] = math.hypot(s.x - prev.x, s.y - prev.y)

    # Arm.
    if arm_manip:
      R_new = geo.axis_angle(dee_rot) @ s.R_ee_b
      s.R_ee_b = geo.orthonormalize(R_new)
      s.ee_b = self.clip_ee(s.ee_b + dee_pos, s)
    else:
      target = self.shoulder_b + (CARRY_REL if s.holding else STOW_REL)
      d = target - s.ee_b
      n = float(np.linalg.norm(d))
      s.ee_b = s.ee_b + (d if n <= ARM_SPEED_NAV else d * ARM_SPEED_NAV / n)
      R_target = R_DEFAULT if s.holding else R_STOW
      rv = geo.log_rot(R_target @ s.R_ee_b.T)
      ang = float(np.linalg.norm(rv))
      if ang > ARM_ROT_NAV:
        rv *= ARM_ROT_NAV / ang
      s.R_ee_b = geo.orthonormalize(geo.axis_angle(rv) @ s.R_ee_b)
      s.ee_b = self.clip_ee(s.ee_b, s)
    if s.holding:
      self._carry(s)
    if self.arm_collides(s):
      if s.holding:
        s.holding = False
        ev['dropped'] = True
      else:
        ev['arm_collision'] = True
        s.ee_b, s.R_ee_b = prev.ee_b.copy(), prev.R_ee_b.copy()
        if self.arm_collides(s):
          # Posture or base motion pushed the arm into geometry: undo the step.
          arm_queue = s.arm_queue
          self.state = s = copy.deepcopy(prev)
          s.arm_queue = arm_queue
          s.v = np.zeros(3)

    # Gripper.
    if grip is not None and arm_manip:
      g = float(grip)
      was_closed = s.grip_closed
      if g < 0.4:
        s.grip_closed = True
      elif g > 0.6:
        s.grip_closed = False
      if s.grip_closed and not was_closed and not s.holding:
        self._try_grasp(s, ev)
      elif was_closed and not s.grip_closed and s.holding:
        self._release(s, ev)
    target = 0.0 if s.grip_closed else 1.0
    if s.holding:
      target = 2 * self.scene.obj.half[0] / GRIP_MAX_WIDTH
    s.grip = float(np.clip(s.grip + np.clip(target - s.grip, -GRIP_RATE, GRIP_RATE), 0, 1))
    return ev

  def _carry(self, s):
    p, R = self.ee_world(s)
    s.obj_center = p + R @ s.obj_offset

  def _try_grasp(self, s, ev):
    p, R = self.ee_world(s)
    obj = self.scene.obj
    dist = float(np.linalg.norm(p - s.obj_center))
    if dist > GRASP_ATTEMPT_RADIUS:
      return
    ev['grasp_attempt'] = True
    approach = R[:, 0]
    ok_dist = dist < GRASP_TOL + float(obj.half[0])
    ok_width = 2 * float(obj.half[0]) < GRIP_MAX_WIDTH - 0.005
    cos = scenes.APPROACH_DIRS[self.scene.approach] @ (-approach)
    ok_dir = bool((cos > math.cos(GRASP_ANGLE)).any())
    ok_free = not self.arm_collides(s)
    if ok_dist and ok_width and ok_dir and ok_free and self.rng.random() < GRASP_P:
      s.holding = s.picked = True
      s.obj_offset = R.T @ (s.obj_center - p)
      ev['grasp_success'] = True
    else:
      s.n_failed += 1
      ev['failed_grasp'] = True

  def _release(self, s, ev):
    s.holding = False
    zone = self.scene.place
    bottom = s.obj_center[2] - self.scene.obj.half[2]
    inside = np.all(np.abs(s.obj_center[:2] - zone.center) <= zone.half)
    if inside and -0.005 <= bottom - zone.support_z <= PLACE_TOL:
      s.placed = True
      s.obj_center = np.array([*s.obj_center[:2], zone.support_z + self.scene.obj.half[2]])
      ev['place_success'] = True
    else:
      ev['dropped'] = True

  # ---------------------------------------------------------------- sensors

  def object_geometry(self, s=None):
    s = s or self.state
    obj = self.scene.obj
    if obj.shape == 'cyl':
      c = s.obj_center
      cyl = np.array([[c[0], c[1], obj.half[0], c[2] - obj.half[2], c[2] + obj.half[2]]])
      return geo.EMPTY_BOXES, cyl
    box = np.concatenate([s.obj_center - obj.half, s.obj_center + obj.half])[None]
    return box, geo.EMPTY_CYLS

  def camera_pose(self, name, s=None):
    s = s or self.state
    cam = self.cams[name]
    dpos, dR = self.dr.head_extr if name == 'head' else self.dr.wrist_extr
    if name == 'head':
      R_parent, origin = self.R_body(s), self.base_pos(s)
    else:
      origin, R_parent = self.ee_world(s)
    pos = origin + R_parent @ (cam['mount'] + dpos)
    R = R_parent @ cam['R_mount'] @ dR
    return pos, R

  def render_true(self, name, s=None):
    """Noise-free z-depth (H, W) in meters, inf where nothing is hit."""
    s = s or self.state
    pos, R = self.camera_pose(name, s)
    dirs = self.rays @ R.T
    obox, ocyl = self.object_geometry(s)
    boxes = np.concatenate([self._static, obox], 0)
    if name == 'head':
      pts = self.arm_points(s)
      r = 0.04
      boxes = np.concatenate([boxes, np.concatenate([pts - r, pts + r], 1)], 0)
    boxes = boxes[geo.boxes_in_front(pos, R[:, 0], boxes)]
    t = np.minimum(geo.ray_boxes(pos, dirs, boxes), geo.ray_floor(pos, dirs))
    t = np.minimum(t, geo.ray_cylinders(pos, dirs, ocyl))
    return t.reshape(I.DEPTH_H, I.DEPTH_W)

  def render(self, name, s=None, noise=True):
    cam = self.cams[name]['model']
    z = self.render_true(name, s)
    if noise:
      i = 0 if name == 'head' else 1
      cam = dataclasses.replace(cam, hole_frac=(self.dr.hole_frac[i],) * 2)
      z = I.apply_depth_noise(z, self.rng, cam, a=self.dr.noise_a[i])
    else:
      far = ~np.isfinite(z) | (z > cam.max_z)
      z = np.where(far, cam.max_z, z)
      z[z < cam.min_z] = 0.0
    return I.depth_to_uint8(z, cam)

  def proprio(self, s=None):
    s = s or self.state
    out = np.zeros(I.PROPRIO_DIM, np.float32)
    L = I.PROPRIO_LAYOUT
    R = self.R_body(s)
    out[L['v_body']] = [s.v[0], s.v[1], 0.0]
    out[L['w_body']] = [0.0, 0.0, s.v[2]]
    out[L['g_proj']] = R.T @ np.array([0.0, 0.0, -1.0])
    out[L['base_height']] = s.h
    out[L['base_pitch']] = s.pitch
    _, (psi, alpha, gamma) = self.elbow_b(s.ee_b, s.R_ee_b)
    out[L['q_arm']] = [psi, alpha, gamma, *geo.log_rot(s.R_ee_b @ R_STOW.T)]
    out[L['ee_pos_b']] = s.ee_b
    out[L['ee_rot6d_b']] = s.R_ee_b[:, :2].T.reshape(-1)
    out[L['grip_width']] = s.grip
    effort = 0.0
    if s.holding:
      effort = 0.8
    elif s.grip_closed:
      effort = 0.2
    out[L['grip_effort']] = effort
    return out

  def phase(self, s=None):
    s = s or self.state
    if s.placed:
      return int(I.Phase.PLACED)
    return int(I.Phase.HOLDING) if s.holding else int(I.Phase.TO_PICK)

  def step_state(self, ev, active_delta_sq=0.0, s=None):
    s = s or self.state
    phase = self.phase(s)
    p_ee, _ = self.ee_world(s)
    if phase == I.Phase.TO_PICK:
      goal = self.scene.grasp_point
      d_ee = float(np.linalg.norm(p_ee - s.obj_center))
    else:
      goal = self.scene.place_point(self.scene.obj.half[2])
      d_ee = float(np.linalg.norm(s.obj_center - goal))
    d_xy = float(math.hypot(goal[0] - s.x, goal[1] - s.y))
    return I.StepState(
        phase=phase, d_xy_goal=d_xy, d_ee_goal=d_ee, roll=0.0,
        grasp_success=ev.get('grasp_success', False),
        place_success=ev.get('place_success', False),
        failed_grasp=ev.get('failed_grasp', False),
        dropped=ev.get('dropped', False),
        collision=ev.get('collision', False) or ev.get('arm_collision', False),
        active_action_delta_sq=float(active_delta_sq))
