"""v4 safety gate, potentials and reward parts for legmm_ag (brief §4, §5). Pure numpy.

Gate (all must hold for a trigger to start settle + skill):
  shoulder-object horizontal distance in [0.45, 0.70] m, object bearing in the body frame within
  +-45 deg, object support height in [0.50, 0.65] m (the scene's table range), base linear speed <= 0.3 m/s.

Potentials. For a state s with shoulder S (world), body yaw, object center c, d = |(c - S)_xy|:
  Phi_dist(s) = -w_dist * max(0.45 - d, 0, d - 0.70)            target position only, 0 inside the band
  Phi_q(s)    =  w_q * q*(s)
  q*(s)       =  w_ang(phi) * g_dist(d) * g_head(|bearing|)       in [0, 1], smooth, true object pose
    bearing  = angle of (c - S) relative to the body x axis
    g_dist   = 1 on [0.45, 0.70], smoothstep to 0 over a further 0.25 m on each side
    g_head   = 1 for |bearing| <= 45 deg, smoothstep to 0 over a further 30 deg
    phi      = angle in [0, 90] deg between the heading (S -> c) and the box's width axis (0 = facing a
               narrow face, 90 = facing a wide face); cylinders: phi = 0
    w_ang    = 1 for phi <= phi_ok, smoothstep to 0 over a further 30 deg
    phi_ok   = beta0 + twist_max, beta0 the largest heading offset from the width axis for which the
               closing width 2*h0*cos(beta) + 2*h1*sin(beta) still fits the effective opening (7.5 cm),
               h0, h1 = half thickness, half width; twist_max = 15 deg (the skill's wrist twist).
Strict shaping with gamma = 1 - 1/horizon: F = gamma * Phi(s') - Phi(s), and F = -Phi(s) at a real
termination (Phi = 0 there). Truncation keeps Phi(s').
"""
import math

import numpy as np

OPENING = 0.075              # effective gripper opening used by kin and scenes.graspable
TWIST_MAX_DEG = 15.0


def smoothstep(t):
  t = min(max(float(t), 0.0), 1.0)
  return t * t * (3.0 - 2.0 * t)


def soft_window(x, lo, hi, ramp):
  """1 on [lo, hi]; smoothstep down to 0 over `ramp` beyond either end."""
  if x < lo:
    return smoothstep(1.0 - (lo - x) / ramp)
  if x > hi:
    return smoothstep(1.0 - (x - hi) / ramp)
  return 1.0


def soft_ceiling(x, limit, ramp):
  return soft_window(x, -math.inf, limit, ramp)


def wrap(a):
  return (a + math.pi) % (2 * math.pi) - math.pi


def gate_cfg(cfg=None):
  g = dict(dist=(0.45, 0.70), bearing_deg=45.0, height=(0.50, 0.65), speed=0.3)
  g.update(dict(cfg or {}))
  return g


def geometry(shoulder_xy, yaw, obj_center):
  """(horizontal distance, signed bearing rad, world heading rad) of the object from the shoulder."""
  d = np.asarray(obj_center[:2], np.float64) - np.asarray(shoulder_xy, np.float64)
  heading = math.atan2(d[1], d[0])
  return float(np.hypot(*d)), wrap(heading - yaw), heading


def gate_report(shoulder_xy, yaw, obj_center, support_z, speed, cfg=None):
  """Per-condition booleans of the safety gate plus the measured values."""
  g = gate_cfg(cfg)
  dist, bearing, _ = geometry(shoulder_xy, yaw, obj_center)
  eps = 1e-9
  rep = dict(
      dist_ok=bool(g['dist'][0] - eps <= dist <= g['dist'][1] + eps),
      bearing_ok=bool(abs(bearing) <= math.radians(g['bearing_deg']) + eps),
      height_ok=bool(g['height'][0] - eps <= support_z <= g['height'][1] + eps),
      speed_ok=bool(speed <= g['speed'] + eps),
      dist=dist, bearing=bearing, support_z=float(support_z), speed=float(speed))
  rep['envelope_ok'] = rep['dist_ok'] and rep['bearing_ok'] and rep['height_ok']
  rep['ok'] = rep['envelope_ok'] and rep['speed_ok']
  return rep


def beta0(half, opening=OPENING):
  """Largest heading offset (rad) from the width axis that still fits the opening."""
  h0, h1 = float(half[0]), float(half[1])
  r = 2.0 * math.hypot(h0, h1)
  if opening >= r:
    return math.pi / 2
  return max(0.0, math.atan2(h1, h0) - math.acos(opening / r))


def width_angle(obj, heading):
  """Angle in [0, pi/2] between the heading and the width axis of a flat box (0 = narrow face)."""
  w = obj.R[:, 1]
  return math.acos(min(1.0, abs(math.cos(heading) * w[0] + math.sin(heading) * w[1])))


def q_star(shoulder_xy, yaw, obj, obj_center=None, twist_deg=TWIST_MAX_DEG):
  """Sector-based stance quality in [0, 1] (true pose, privileged)."""
  c = obj.center if obj_center is None else obj_center
  dist, bearing, heading = geometry(shoulder_xy, yaw, c)
  g_dist = soft_window(dist, 0.45, 0.70, 0.25)
  g_head = soft_ceiling(abs(bearing), math.radians(45.0), math.radians(30.0))
  if obj.shape == 'flat':
    phi_ok = beta0(obj.half) + math.radians(twist_deg)
    w_ang = soft_ceiling(width_angle(obj, heading), phi_ok, math.radians(30.0))
  else:
    w_ang = 1.0
  return float(w_ang * g_dist * g_head)


def dist_band(dist):
  return max(0.45 - dist, 0.0, dist - 0.70)


def potentials(shoulder_xy, yaw, obj, obj_center, shaping):
  """(Phi_dist, Phi_q, q*) for one state."""
  dist, _, _ = geometry(shoulder_xy, yaw, obj_center)
  q = q_star(shoulder_xy, yaw, obj, obj_center, shaping.get('twist_deg', TWIST_MAX_DEG))
  return (-float(shaping['w_dist']) * dist_band(dist), float(shaping['w_q']) * q, q)


def shaping_diff(phi_prev, phi_next, gamma, terminal):
  """Strict discounted potential difference; the potential is 0 after a real termination."""
  return float(gamma * (0.0 if terminal else phi_next) - phi_prev)


def compose_reward(rew_task, rew_qshape, lam):
  """Training-time reward: task part (incl. distance shaping) plus lambda times the q* shaping."""
  return rew_task + lam * rew_qshape


def gamma_from_horizon(horizon):
  return 1.0 - 1.0 / float(horizon)
