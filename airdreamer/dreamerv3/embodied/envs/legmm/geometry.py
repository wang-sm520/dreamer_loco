"""Vectorized geometry for the kinematic stand-in backend (pure numpy).

Conventions: world frame x/y horizontal, z up. Boxes are axis-aligned and
stored as (N, 6) arrays [xmin, ymin, zmin, xmax, ymax, zmax]. Vertical
cylinders are (M, 5) arrays [cx, cy, radius, zmin, zmax].
"""
import math

import numpy as np

EMPTY_BOXES = np.zeros((0, 6), np.float64)
EMPTY_CYLS = np.zeros((0, 5), np.float64)


def rot_z(yaw):
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(roll):
  c, s = math.cos(roll), math.sin(roll)
  return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_pitch(pitch):
  """Rotation about the body y axis. Negative pitch tilts the x axis down."""
  c, s = math.cos(pitch), math.sin(pitch)
  return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def axis_angle(vec):
  """Rotation matrix of an axis-angle vector (Rodrigues)."""
  vec = np.asarray(vec, np.float64)
  theta = float(np.linalg.norm(vec))
  if theta < 1e-9:
    return np.eye(3)
  k = vec / theta
  K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
  return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def log_rot(R):
  """Axis-angle vector of a rotation matrix."""
  cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
  theta = math.acos(cos)
  if theta < 1e-6:
    return np.zeros(3)
  if math.pi - theta < 1e-4:
    # Near pi: take the axis from the diagonal.
    axis = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
    i = int(np.argmax(axis))
    axis[(i + 1) % 3] *= np.sign(R[i, (i + 1) % 3] + 1e-12)
    axis[(i + 2) % 3] *= np.sign(R[i, (i + 2) % 3] + 1e-12)
    return theta * axis / np.linalg.norm(axis)
  w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  return theta * w / (2 * math.sin(theta))


def orthonormalize(R):
  u, _, vt = np.linalg.svd(R)
  return u @ vt


def camera_rays(width, height, hfov_deg, vfov_deg):
  """Ray directions in the camera frame (x forward, y left, z up), shape
  (H*W, 3), scaled so that the forward component is 1. A hit parameter t is
  then the z-depth that a depth camera reports."""
  tx = math.tan(math.radians(hfov_deg) / 2)
  tz = math.tan(math.radians(vfov_deg) / 2)
  u = (np.arange(width) + 0.5) / width * 2 - 1
  v = (np.arange(height) + 0.5) / height * 2 - 1
  uu, vv = np.meshgrid(u, v)
  dirs = np.stack([np.ones_like(uu), -uu * tx, -vv * tz], -1)
  return dirs.reshape(-1, 3)


def ray_boxes(origin, dirs, boxes):
  """Nearest positive hit parameter of rays against AABBs (slab method).

  origin: (3,), dirs: (R, 3), boxes: (N, 6). Returns (R,) with inf on miss.
  Works per axis on (R, N) arrays; zero direction components are replaced by a
  tiny value so that no nan appears.
  """
  if len(boxes) == 0:
    return np.full(len(dirs), np.inf, np.float32)
  dirs = np.asarray(dirs, np.float32)
  d = np.where(np.abs(dirs) < 1e-9, np.float32(1e-9), dirs)
  inv = (1.0 / d).astype(np.float32)
  lo = (boxes[:, :3] - origin).astype(np.float32)   # (N, 3)
  hi = (boxes[:, 3:] - origin).astype(np.float32)
  tnear = tfar = None
  for a in range(3):
    t0 = inv[:, a:a + 1] * lo[None, :, a]             # (R, N)
    t1 = inv[:, a:a + 1] * hi[None, :, a]
    mn, mx = np.minimum(t0, t1), np.maximum(t0, t1)
    tnear = mn if tnear is None else np.maximum(tnear, mn)
    tfar = mx if tfar is None else np.minimum(tfar, mx)
  t = np.where(tnear > 0, tnear, tfar)
  t[(tnear > tfar) | (tfar <= 0)] = np.inf
  return t.min(1)


def boxes_in_front(origin, forward, boxes, min_depth=0.0):
  """Mask of boxes that have at least one corner in front of the camera."""
  if len(boxes) == 0:
    return np.zeros(0, bool)
  # The max over corners of a linear function is attained per axis.
  f = np.asarray(forward)
  best = np.where(f >= 0, boxes[:, 3:], boxes[:, :3]) @ f - origin @ f
  return best > min_depth


def ray_obb(origin, dirs, center, half, R):
  """Nearest positive hit of rays against one oriented box (R: box -> world)."""
  o = (np.asarray(origin) - center) @ R
  d = np.asarray(dirs) @ R
  return ray_boxes(o, d, np.concatenate([-half, half])[None])


def obb_corners(center, half, R):
  signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
  return center[None] + (signs * half[None]) @ R.T


def ray_cylinders(origin, dirs, cyls):
  """Nearest positive hit of rays against finite vertical cylinders."""
  if len(cyls) == 0:
    return np.full(len(dirs), np.inf)
  ox = origin[0] - cyls[:, 0]                    # (M,)
  oy = origin[1] - cyls[:, 1]
  dx, dy, dz = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]   # (R, 1)
  a = dx * dx + dy * dy
  b = 2 * (dx * ox + dy * oy)
  c = ox * ox + oy * oy - cyls[:, 2] ** 2
  disc = b * b - 4 * a * c
  with np.errstate(divide='ignore', invalid='ignore'):
    sq = np.sqrt(np.maximum(disc, 0))
    ta = (-b - sq) / (2 * a)
    tb = (-b + sq) / (2 * a)
  best = np.full(disc.shape, np.inf)
  for t in (ta, tb):
    z = origin[2] + t * dz
    ok = (disc >= 0) & (t > 0) & (z >= cyls[:, 3]) & (z <= cyls[:, 4])
    best = np.minimum(best, np.where(ok, t, np.inf))
  # Caps.
  for zc in (cyls[:, 3], cyls[:, 4]):
    with np.errstate(divide='ignore', invalid='ignore'):
      t = (zc - origin[2]) / dz
    px = ox + t * dx
    py = oy + t * dy
    ok = (t > 0) & (px * px + py * py <= cyls[:, 2] ** 2)
    best = np.minimum(best, np.where(ok, t, np.inf))
  return best.min(-1)


def ray_floor(origin, dirs):
  with np.errstate(divide='ignore', invalid='ignore'):
    t = -origin[2] / dirs[:, 2]
  return np.where((dirs[:, 2] < 0) & (t > 0), t, np.inf)


def points_in_boxes(points, boxes, margin=0.0):
  """(P, N) bool: point p lies inside box n inflated by margin."""
  if len(boxes) == 0:
    return np.zeros((len(points), 0), bool)
  lo = boxes[None, :, :3] - margin
  hi = boxes[None, :, 3:] + margin
  p = points[:, None, :]
  return np.all((p >= lo) & (p <= hi), -1)


def points_in_cylinders(points, cyls, margin=0.0):
  if len(cyls) == 0:
    return np.zeros((len(points), 0), bool)
  dx = points[:, None, 0] - cyls[None, :, 0]
  dy = points[:, None, 1] - cyls[None, :, 1]
  z = points[:, None, 2]
  r = cyls[None, :, 2] + margin
  return ((dx * dx + dy * dy) <= r * r) & (z >= cyls[None, :, 3] - margin) & (
      z <= cyls[None, :, 4] + margin)


def obb_hits_boxes(center, yaw, half_extent, boxes):
  """2D separating-axis test between a yawed rectangle and AABBs (xy only).

  Returns (N,) bool."""
  if len(boxes) == 0:
    return np.zeros(0, bool)
  c, s = math.cos(yaw), math.sin(yaw)
  ax = np.array([[c, s], [-s, c]])                 # rectangle axes
  hx, hy = half_extent
  bc = (boxes[:, :2] + boxes[:, 3:5]) / 2          # (N, 2)
  bh = (boxes[:, 3:5] - boxes[:, :2]) / 2
  d = bc - np.asarray(center)[None]
  # World axes.
  r_rect_x = abs(c) * hx + abs(s) * hy
  r_rect_y = abs(s) * hx + abs(c) * hy
  sep = (np.abs(d[:, 0]) > bh[:, 0] + r_rect_x) | (np.abs(d[:, 1]) > bh[:, 1] + r_rect_y)
  # Rectangle axes.
  for i, half in enumerate((hx, hy)):
    a = ax[i]
    proj_d = np.abs(d @ a)
    r_box = bh[:, 0] * abs(a[0]) + bh[:, 1] * abs(a[1])
    sep |= proj_d > half + r_box
  return ~sep


def segment_boxes_free(p0, p1, boxes, margin=0.0, samples=8):
  """True if sampled points on the segment p0->p1 avoid all boxes."""
  ts = np.linspace(0.0, 1.0, samples)[:, None]
  pts = p0[None] * (1 - ts) + p1[None] * ts
  return not points_in_boxes(pts, boxes, margin).any()
