"""Kinematic stand-in backend (brief §6.2): pure numpy, no physics engine.

Base: planar pose (x, y, yaw) plus posture (height, pitch); body velocities
and posture track their commands with first-order lags. Arm: the end-effector
pose is commanded in the body frame, clipped to a reach shell around the
shoulder, with a two-link vertical-plane IK for the elbow. Depth: vectorized
ray casting against boxes, vertical cylinders, the floor, and (for the head
camera) the arm links.
"""
import copy
import dataclasses
import math

import numpy as np

from . import geometry as geo
from . import interfaces as I
from . import scenes

# TODO(calibrate): AIRBOT Play link lengths (sum = reach) and gripper stroke.
LINK1 = 0.30
LINK2 = 0.347
GRIP_MAX_WIDTH = 0.10
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
BODY_CLEAR = np.array([0.45, 0.25, 0.10])   # ee may not enter |x|<.45, |y|<.25, z<.10 (body frame)

STOW_B = np.array([0.25, 0.0, 0.30])
CARRY_B = np.array([0.40, 0.0, 0.30])
R_DEFAULT = geo.rot_pitch(math.radians(-30.0))   # approach axis forward, 30 deg down


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
  pitch: float
  v: np.ndarray               # actual body vx, vy, wz
  ee_b: np.ndarray            # gripper center, body frame
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
    self.shoulder_b = np.asarray(arm.get('mount_xyz', (0.10, 0.0, 0.12)), np.float64)
    self.reach = float(arm.get('reach', 0.647))
    self.min_reach = float(arm.get('min_reach', 0.15))
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
    self.p_range = I.NAV_LAYOUT['pitch'][1:]
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
        x=float(x), y=float(y), yaw=float(yaw), h=0.45, pitch=0.0, v=np.zeros(3),
        ee_b=STOW_B.copy(), R_ee_b=R_DEFAULT.copy(), grip=1.0, grip_closed=False,
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

  def elbow_b(self, ee_b):
    v = ee_b - self.shoulder_b
    psi = math.atan2(v[1], v[0])
    r = math.hypot(v[0], v[1])
    D = float(np.clip(np.linalg.norm(v), abs(LINK1 - LINK2) + 1e-3, LINK1 + LINK2 - 1e-6))
    phi = math.atan2(v[2], r)
    beta = math.acos(np.clip((LINK1 ** 2 + D ** 2 - LINK2 ** 2) / (2 * LINK1 * D), -1, 1))
    alpha = phi + beta
    elbow = self.shoulder_b + LINK1 * np.array(
        [math.cos(alpha) * math.cos(psi), math.cos(alpha) * math.sin(psi), math.sin(alpha)])
    gamma = math.pi - math.acos(np.clip((LINK1 ** 2 + LINK2 ** 2 - D ** 2) / (2 * LINK1 * LINK2), -1, 1))
    return elbow, (psi, alpha, gamma)

  def clip_ee(self, ee_b, s):
    v = ee_b - self.shoulder_b
    n = float(np.linalg.norm(v))
    if n > self.reach:
      v *= self.reach / n
    elif n < self.min_reach:
      v = v / max(n, 1e-6) * self.min_reach if n > 1e-6 else np.array([self.min_reach, 0, 0])
    ee_b = self.shoulder_b + v
    if abs(ee_b[0]) < BODY_CLEAR[0] and abs(ee_b[1]) < BODY_CLEAR[1] and ee_b[2] < BODY_CLEAR[2]:
      ee_b = ee_b.copy()
      ee_b[2] = BODY_CLEAR[2]
    ee_w = self.to_world(ee_b, s)
    if ee_w[2] < 0.02:
      ee_w[2] = 0.02
      ee_b = self.R_body(s).T @ (ee_w - self.base_pos(s))
      v = ee_b - self.shoulder_b
      n = float(np.linalg.norm(v))
      if n > self.reach:
        ee_b = self.shoulder_b + v * self.reach / n
    return ee_b

  def arm_points(self, s=None, n=4):
    """World points along both links plus the fingertip."""
    s = s or self.state
    elbow, _ = self.elbow_b(s.ee_b)
    ts = np.linspace(0.0, 1.0, n + 1)[1:, None]
    seg1 = self.shoulder_b[None] * (1 - ts) + elbow[None] * ts
    seg2 = elbow[None] * (1 - ts) + s.ee_b[None] * ts
    tip = s.ee_b + 0.04 * s.R_ee_b[:, 0]
    pts_b = np.concatenate([seg1, seg2, tip[None]], 0)
    return self.base_pos(s)[None] + pts_b @ self.R_body(s).T

  def held_points(self, s=None):
    s = s or self.state
    c = s.obj_center
    hz = self.scene.obj.half[2]
    return np.array([c, c - [0, 0, hz - 0.005]])

  def arm_collides(self, s=None):
    s = s or self.state
    if geo.points_in_boxes(self.arm_points(s), self._static, LINK_RADIUS).any():
      return True
    if s.holding:
      pts = self.held_points(s)
      r = float(self.scene.obj.half[0])
      if geo.points_in_boxes(pts[:1], self._static, r).any():
        return True
      if geo.points_in_boxes(pts[1:], self._static, 0.0).any() or pts[1, 2] < 0.0:
        return True
    return False

  def base_collides(self, s=None):
    s = s or self.state
    return bool(geo.obb_hits_boxes((s.x, s.y), s.yaw, self.foot_half, self._static).any())

  # ------------------------------------------------------------------- step

  def step(self, cmd):
    """cmd: dict with skill (int), base (3,) vx vy wz, height, pitch and, for
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
    p_cmd = float(np.clip(cmd['pitch'], *self.p_range))
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
      s.pitch += (p_cmd - s.pitch) * ap
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
      target = CARRY_B if s.holding else STOW_B
      d = target - s.ee_b
      n = float(np.linalg.norm(d))
      s.ee_b = s.ee_b + (d if n <= ARM_SPEED_NAV else d * ARM_SPEED_NAV / n)
      rv = geo.log_rot(R_DEFAULT @ s.R_ee_b.T)
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
    cos = scenes.APPROACH_DIRS[self.scene.approach] @ (-approach)
    ok_dir = bool((cos > math.cos(GRASP_ANGLE)).any())
    ok_free = not self.arm_collides(s)
    if ok_dist and ok_dir and ok_free and self.rng.random() < GRASP_P:
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
      r = 0.035
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
    _, (psi, alpha, gamma) = self.elbow_b(s.ee_b)
    out[L['q_arm']] = [psi, alpha, gamma, *geo.log_rot(s.R_ee_b @ R_DEFAULT.T)]
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
