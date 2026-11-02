"""Procedural scenes for the kinematic backend, fully determined by (split, seed).

A scene is static geometry (axis-aligned boxes), one object to pick, one place
zone, and a robot start pose. Furniture layouts follow interfaces.SceneType.
"""
import dataclasses
import math
import zlib
from typing import Optional

import numpy as np
from scipy import ndimage

from . import geometry as geo
from .interfaces import SPLITS, SceneType

# Robot footprint half extents (length, width) used for clearance checks.
# Matches env.legmm.base.footprint [0.82, 0.44] in configs.yaml.
FOOT_HALF = (0.41, 0.22)
WALL_T = 0.1
WALL_H = 2.0
GRID_RES = 0.05
NAV_INFLATE = 0.3          # occupancy inflation used by the expert's A* (brief §7.1)
APPROACH_LEN = 0.25        # free corridor length for an approach direction (brief §6.2)
N_DIRS = 8                 # horizontal approach bins; index 8 = top-down


@dataclasses.dataclass
class SceneObject:
  shape: str                 # 'flat' (upright flat box) | 'cyl' (vertical cylinder)
  center: np.ndarray         # (3,) world
  half: np.ndarray           # (3,) half extents in the object frame; 'flat': x = thickness,
                             # y = width (wider than the gripper), z = height; 'cyl': (r, r, h/2)
  support_z: float           # z of the surface it rests on
  yaw: float = 0.0           # object frame heading (world z rotation)

  @property
  def R(self):
    return geo.rot_z(self.yaw)


@dataclasses.dataclass
class PlaceZone:
  center: np.ndarray         # (2,) xy
  half: np.ndarray           # (2,)
  support_z: float


@dataclasses.dataclass
class Scene:
  split: str
  seed: int
  scene_type: int
  room: np.ndarray           # (2,) size; the room spans [-W/2, W/2] x [-L/2, L/2]
  boxes: np.ndarray          # (N, 6) static solid geometry
  kinds: tuple               # per box: wall | table | obstacle | cubby | place_table
  obj: SceneObject
  place: PlaceZone
  start: np.ndarray          # (3,) x, y, yaw
  approach: np.ndarray       # (9,) bool; graspable and free, per entry of `dirs`
  occupancy: Optional[np.ndarray] = None   # (gx, gy) bool, inflated by NAV_INFLATE
  dirs: np.ndarray = None    # (9, 3) outward approach directions: 8 horizontal bins (in the
                             # object frame for flat boxes) + top-down

  def __post_init__(self):
    if self.dirs is None:
      self.dirs = object_dirs(self.obj)

  @property
  def grasp_point(self):
    return self.obj.center.copy()

  def place_point(self, obj_half_z):
    return np.array([*self.place.center, self.place.support_z + obj_half_z + 0.01])

  def fingerprint(self):
    parts = [self.boxes.round(6).tobytes(), self.obj.center.round(6).tobytes(),
             self.obj.half.round(6).tobytes(), self.place.center.round(6).tobytes(),
             self.start.round(6).tobytes(), self.approach.tobytes(),
             str((self.scene_type, self.obj.shape, round(self.obj.yaw, 6), self.kinds)).encode()]
    return zlib.crc32(b''.join(parts))


def approach_dirs():
  """(9, 3) unit vectors from the object outwards; the last one is +z."""
  ang = np.arange(N_DIRS) * 2 * math.pi / N_DIRS
  dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros(N_DIRS)], -1)
  return np.concatenate([dirs, [[0.0, 0.0, 1.0]]], 0)


APPROACH_DIRS = approach_dirs()


def object_dirs(obj):
  """Approach bins of an object: world bins for cylinders, bins rotated into the
  object frame for flat boxes (bins 2 and 6 are then the narrow faces +-y)."""
  if obj.shape != 'flat':
    return APPROACH_DIRS.copy()
  dirs = APPROACH_DIRS.copy()
  dirs[:N_DIRS] = dirs[:N_DIRS] @ obj.R.T
  return dirs


def graspable(obj, d, grip_max=0.075):
  """Can the gripper close across the object when travelling along -d?
  Fingers close horizontally, perpendicular to d."""
  if d[2] > 0.5:
    return False                       # top-down grasps are disabled (user decision)
  c = np.array([-d[1], d[0], 0.0])
  if obj.shape == 'flat':
    R = obj.R
    width = 2 * (obj.half[0] * abs(c @ R[:, 0]) + obj.half[1] * abs(c @ R[:, 1]))
  else:
    width = 2 * obj.half[0]
  return width < grip_max


def approach_free(obj, boxes, dirs=None):
  """Brief §6.2 plus graspability: a direction is approachable if the gripper
  can close across the object from it and a corridor of APPROACH_LEN from the
  object surface along it is free of obstacles."""
  dirs = object_dirs(obj) if dirs is None else dirs
  free = np.zeros(N_DIRS + 1, bool)
  R = obj.R
  for i, d in enumerate(dirs):
    if not graspable(obj, d):
      continue
    # Distance from the center to the surface along d.
    surf = obj.half[2] if i == N_DIRS else (
        abs(d @ R[:, 0]) * obj.half[0] + abs(d @ R[:, 1]) * obj.half[1]
        if obj.shape == 'flat' else obj.half[0])
    start = surf + 0.01
    ts = np.linspace(start, start + APPROACH_LEN, 6)
    pts = obj.center[None] + ts[:, None] * d[None]
    # Gripper width: also probe +-3 cm sideways.
    side = np.array([-d[1], d[0], 0.0]) if i < N_DIRS else np.array([1.0, 0.0, 0.0])
    pts = np.concatenate([pts, pts + 0.03 * side, pts - 0.03 * side], 0)
    free[i] = not geo.points_in_boxes(pts, boxes, 0.01).any()
  return free


def occupancy_grid(boxes, room, inflate=NAV_INFLATE, res=GRID_RES):
  """Boolean grid over the room, True = occupied after inflation."""
  nx = int(round(room[0] / res))
  ny = int(round(room[1] / res))
  xs = -room[0] / 2 + (np.arange(nx) + 0.5) * res
  ys = -room[1] / 2 + (np.arange(ny) + 0.5) * res
  occ = np.zeros((nx, ny), bool)
  for b in boxes:
    ix = (xs >= b[0] - inflate) & (xs <= b[3] + inflate)
    iy = (ys >= b[1] - inflate) & (ys <= b[4] + inflate)
    occ[np.ix_(ix, iy)] = True
  return occ


def world_to_cell(xy, room, res=GRID_RES):
  return (int(np.floor((xy[0] + room[0] / 2) / res)),
          int(np.floor((xy[1] + room[1] / 2) / res)))


def cell_to_world(ij, room, res=GRID_RES):
  return np.array([-room[0] / 2 + (ij[0] + 0.5) * res, -room[1] / 2 + (ij[1] + 0.5) * res])


def footprint_free(boxes, x, y, yaw, margin=0.0):
  hx, hy = FOOT_HALF[0] + margin, FOOT_HALF[1] + margin
  return not geo.obb_hits_boxes((x, y), yaw, (hx, hy), boxes).any()


def _box(cx, cy, sx, sy, z0, z1):
  return np.array([cx - sx / 2, cy - sy / 2, z0, cx + sx / 2, cy + sy / 2, z1])


def _walls(room):
  W, L = room
  t, h = WALL_T, WALL_H
  return [
      np.array([-W / 2 - t, -L / 2 - t, 0, W / 2 + t, -L / 2, h]),
      np.array([-W / 2 - t, L / 2, 0, W / 2 + t, L / 2 + t, h]),
      np.array([-W / 2 - t, -L / 2, 0, -W / 2, L / 2, h]),
      np.array([W / 2, -L / 2, 0, W / 2 + t, L / 2, h]),
  ]


class _Builder:

  def __init__(self, split, seed, cfg, attempt=0):
    self.split, self.seed, self.cfg = split, int(seed), cfg
    # The trailing non-zero tag matters: numpy's SeedSequence pads entropy with
    # zeros, so [crc, seed, 0] would be the same stream as the scene-type RNG
    # [crc, seed] and correlate the layout with the scene type.
    self.rng = np.random.default_rng([zlib.crc32(split.encode()), int(seed), attempt, 0x5CE7E])
    self.room = np.asarray(cfg.get('room_size', (8.0, 8.0)), np.float64)
    self.boxes, self.kinds = [], []
    for w in _walls(self.room):
      self.add(w, 'wall')

  def add(self, box, kind):
    self.boxes.append(np.asarray(box, np.float64))
    self.kinds.append(kind)

  def arr(self):
    return np.stack(self.boxes) if self.boxes else geo.EMPTY_BOXES

  def u(self, lo, hi):
    return float(self.rng.uniform(lo, hi))

  def make_object(self, x, y, support_z):
    """70% upright flat boxes (only the two narrow faces are graspable),
    30% cylinders as a control (user decision 2026-11-02)."""
    ocfg = self.cfg.get('object', {})
    self.rng.random()  # retain the layout stream's draw count
    if object_shape_of(self.split, self.seed, self.cfg) == 'flat':
      f = ocfg.get('flat', {})
      t = self.u(*f.get('thick', (0.03, 0.05)))
      w = self.u(*f.get('wide', (0.10, 0.14)))
      h = self.u(*f.get('height', (0.12, 0.15)))
      half = np.array([t / 2, w / 2, h / 2])
      return SceneObject('flat', np.array([x, y, support_z + h / 2]), half, support_z,
                         yaw=self.u(0.0, math.pi))
    c = ocfg.get('cyl', {})
    d = self.u(*c.get('diameter', (0.04, 0.07)))
    h = self.u(*c.get('height', (0.08, 0.20)))
    half = np.array([d / 2, d / 2, h / 2])
    return SceneObject('cyl', np.array([x, y, support_z + h / 2]), half, support_z)

  def free_xy(self, margin, avoid=(), tries=200):
    W, L = self.room
    for _ in range(tries):
      x = self.u(-W / 2 + margin, W / 2 - margin)
      y = self.u(-L / 2 + margin, L / 2 - margin)
      if all(np.hypot(x - ax, y - ay) >= r for ax, ay, r in avoid):
        pts = np.array([[x, y, 0.3]])
        if not geo.points_in_boxes(pts, self.arr(), margin).any():
          return x, y
    return None


def _wall_frame(b, side, along, depth, inset=0.0):
  """Center of a rectangle (along x depth) whose back touches wall `side`.

  Returns (cx, cy, sx, sy, normal) where normal points into the room."""
  W, L = b.room
  if side == 0:   # y = -L/2
    return along, -L / 2 + inset + depth / 2, None, depth, np.array([0.0, 1.0])
  if side == 1:   # y = +L/2
    return along, L / 2 - inset - depth / 2, None, depth, np.array([0.0, -1.0])
  if side == 2:   # x = -W/2
    return -W / 2 + inset + depth / 2, along, depth, None, np.array([1.0, 0.0])
  return W / 2 - inset - depth / 2, along, depth, None, np.array([-1.0, 0.0])


def _furniture(b, stype):
  """Adds the pick furniture and returns the object."""
  W, L = b.room
  if stype == SceneType.OPEN:
    sx, sy, h = b.u(0.6, 1.2), b.u(0.5, 0.8), b.u(0.40, 0.75)
    cx, cy = b.u(-W / 2 + 1.6, W / 2 - 1.6), b.u(-L / 2 + 1.6, L / 2 - 1.6)
    b.add(_box(cx, cy, sx, sy, 0, h), 'table')
    ox = cx + b.u(-sx / 2 + 0.1, sx / 2 - 0.1)
    oy = cy + b.u(-sy / 2 + 0.1, sy / 2 - 0.1)
    # Keep the object near an edge so that it is reachable from outside.
    if b.rng.random() < 0.5:
      ox = cx + np.sign(ox - cx + 1e-9) * (sx / 2 - b.u(0.08, 0.25))
    else:
      oy = cy + np.sign(oy - cy + 1e-9) * (sy / 2 - b.u(0.08, 0.25))
    return b.make_object(ox, oy, h), np.array([cx, cy])

  if stype in (SceneType.WALL_BACK, SceneType.CUBBY):
    side = int(b.rng.integers(4))
    span = (W if side in (0, 1) else L) / 2
    if stype == SceneType.WALL_BACK:
      width, depth, h = b.u(1.0, 1.4), b.u(0.5, 0.7), b.u(0.40, 0.75)
    else:
      # Cubby: 0.45-0.55 m tall, object 3-8 cm inside the opening (user decision 2026-11-02).
      # With 0.30-0.40 m and 8-18 cm no layout was reachable for A2 + AIRBOT; this gives
      # ~half reachable layouts (the other half tests recognising infeasibility).
      width, depth, h = b.u(0.5, 0.7), b.u(0.4, 0.5), b.u(0.45, 0.55)
    along = b.u(-span + width / 2 + 0.3, span - width / 2 - 0.3)
    cx, cy, sx, sy, n = _wall_frame(b, side, along, depth)
    sx = width if sx is None else sx
    sy = width if sy is None else sy
    t = np.array([-n[1], n[0]])                    # along the wall
    lateral = b.u(-width / 2 + 0.12, width / 2 - 0.12) if stype == SceneType.WALL_BACK \
        else b.u(-0.08, 0.08)
    front_in = b.u(0.08, 0.30) if stype == SceneType.WALL_BACK else b.u(0.03, 0.08)
    front = np.array([cx, cy]) + n * depth / 2
    oxy = front - n * front_in + t * lateral
    if stype == SceneType.WALL_BACK:
      b.add(_box(cx, cy, sx, sy, 0, h), 'table')
      return b.make_object(oxy[0], oxy[1], h), np.array([cx, cy])
    # Cubby: bottom, top, two sides and back, open towards n.
    pt = 0.03
    b.add(_box(cx, cy, sx, sy, 0, pt), 'cubby')
    b.add(_box(cx, cy, sx, sy, h - pt, h), 'cubby')
    if abs(n[0]) > 0:
      b.add(_box(cx, cy - sy / 2 + pt / 2, sx, pt, 0, h), 'cubby')
      b.add(_box(cx, cy + sy / 2 - pt / 2, sx, pt, 0, h), 'cubby')
      b.add(_box(cx - n[0] * (sx / 2 - pt / 2), cy, pt, sy, 0, h), 'cubby')
    else:
      b.add(_box(cx - sx / 2 + pt / 2, cy, pt, sy, 0, h), 'cubby')
      b.add(_box(cx + sx / 2 - pt / 2, cy, pt, sy, 0, h), 'cubby')
      b.add(_box(cx, cy - n[1] * (sy / 2 - pt / 2), sx, pt, 0, h), 'cubby')
    obj = b.make_object(oxy[0], oxy[1], pt)
    # The object must fit under the top panel.
    max_h = h - 2 * pt - 0.04
    if obj.half[2] * 2 > max_h:
      obj.half[2] = max_h / 2
      obj.center[2] = pt + obj.half[2]
    return obj, np.array([cx, cy])

  if stype == SceneType.SIDE_BLOCKED:
    sx, sy, h = b.u(0.6, 1.0), b.u(0.6, 0.9), b.u(0.40, 0.75)
    cx, cy = b.u(-W / 2 + 2.0, W / 2 - 2.0), b.u(-L / 2 + 2.0, L / 2 - 2.0)
    b.add(_box(cx, cy, sx, sy, 0, h), 'table')
    sides = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0])]
    open_i = int(b.rng.integers(4))
    n = sides[open_i]
    t = np.array([-n[1], n[0]])
    half_n = sx / 2 if n[0] else sy / 2
    half_t = sy / 2 if n[0] else sx / 2
    oxy = np.array([cx, cy]) + n * (half_n - b.u(0.08, 0.25)) + t * b.u(-half_t + 0.1, half_t - 0.1)
    others = [i for i in range(4) if i != open_i]
    n_block = 3 if b.rng.random() < 0.5 else 2
    blocked = list(b.rng.choice(others, n_block, replace=False))
    for i in blocked:
      m = sides[i]
      thick = b.u(0.3, 0.5)
      length = (sy if m[0] else sx) + 0.4
      hn = sx / 2 if m[0] else sy / 2
      c = np.array([cx, cy]) + m * (hn + b.u(0.0, 0.05) + thick / 2)
      bsx, bsy = (thick, length) if m[0] else (length, thick)
      b.add(_box(c[0], c[1], bsx, bsy, 0, b.u(0.9, 1.3)), 'obstacle')
    return b.make_object(oxy[0], oxy[1], h), np.array([cx, cy])

  if stype == SceneType.LOW_FLOOR:
    ox, oy = b.u(-W / 2 + 1.0, W / 2 - 1.0), b.u(-L / 2 + 1.0, L / 2 - 1.0)
    obj = b.make_object(ox, oy, 0.0)
    if b.rng.random() < 0.6:
      ang = b.u(0, 2 * math.pi)
      d = b.u(0.25, 0.45)
      s = b.u(0.2, 0.4)
      b.add(_box(ox + d * math.cos(ang), oy + d * math.sin(ang), s, s, 0, b.u(0.15, 0.4)), 'obstacle')
    return obj, np.array([ox, oy])

  if stype == SceneType.CORNER:
    sxw, syw, h = b.u(0.6, 1.0), b.u(0.6, 1.0), b.u(0.40, 0.75)
    qx, qy = (1 if b.rng.random() < 0.5 else -1), (1 if b.rng.random() < 0.5 else -1)
    cx, cy = qx * (W / 2 - sxw / 2), qy * (L / 2 - syw / 2)
    b.add(_box(cx, cy, sxw, syw, 0, h), 'table')
    ox = cx - qx * (sxw / 2 - b.u(0.1, 0.3))
    oy = cy - qy * (syw / 2 - b.u(0.1, 0.3))
    return b.make_object(ox, oy, h), np.array([cx, cy])

  raise ValueError(stype)


def _place_zone(b, obj):
  W, L = b.room
  half = np.array([0.175, 0.175])
  for _ in range(100):
    on_table = b.rng.random() < 0.5
    xy = b.free_xy(1.0, avoid=[(obj.center[0], obj.center[1], 2.5)])
    if xy is None:
      continue
    if on_table:
      h = b.u(0.40, 0.60)
      b.add(_box(xy[0], xy[1], 0.5, 0.5, 0, h), 'place_table')
      return PlaceZone(np.array(xy), half, h)
    return PlaceZone(np.array(xy), half, 0.0)
  raise RuntimeError('could not place the place zone')


def _connected(occ, room, start_xy, targets, radius):
  """True if start reaches, through free cells, a free cell within `radius` of
  each target xy."""
  lab, _ = ndimage.label(~occ)
  si, sj = world_to_cell(start_xy, room)
  if not (0 <= si < occ.shape[0] and 0 <= sj < occ.shape[1]) or occ[si, sj]:
    return False
  comp = lab == lab[si, sj]
  nx, ny = occ.shape
  xs = -room[0] / 2 + (np.arange(nx) + 0.5) * GRID_RES
  ys = -room[1] / 2 + (np.arange(ny) + 0.5) * GRID_RES
  X, Y = np.meshgrid(xs, ys, indexing='ij')
  for tx, ty in targets:
    near = (X - tx) ** 2 + (Y - ty) ** 2 <= radius ** 2
    if not (near & comp).any():
      return False
  return True


def scene_type_of(split, seed):
  types = SPLITS[split]['types']
  type_rng = np.random.default_rng([zlib.crc32(split.encode()), int(seed)])
  return SceneType(int(types[int(type_rng.integers(len(types)))]))


def object_shape_of(split, seed, cfg=None):
  """Choose the object stratum independently of layout retries and rejection."""
  frac = float((cfg or {}).get('object', {}).get('flat_frac', 0.7))
  if not 0.0 <= frac <= 1.0:
    raise ValueError(f'object.flat_frac must be in [0, 1], got {frac}')
  rng = np.random.default_rng([zlib.crc32(split.encode()), int(seed), 0x0B1EC7])
  return 'flat' if rng.random() < frac else 'cyl'


def make_scene(split, seed, cfg=None):
  cfg = dict(cfg or {})
  spec = SPLITS[split]
  types = spec['types']
  lo, hi = cfg.get('n_obstacles', (3, 8))
  if spec.get('layout') == 'large_clutter':
    lo, hi = hi, hi + 6
  stype = scene_type_of(split, seed)
  for attempt in range(30):
    b = _Builder(split, seed, cfg, attempt)
    obj, _ = _furniture(b, stype)
    place = _place_zone(b, obj)
    n_furn = len(b.boxes)
    start = None
    for _ in range(200):
      xy = b.free_xy(0.6, avoid=[(obj.center[0], obj.center[1], 2.0), (*place.center, 0.8)])
      if xy is None:
        continue
      yaw = b.u(-math.pi, math.pi)
      if footprint_free(b.arr(), xy[0], xy[1], yaw, 0.15):
        start = np.array([xy[0], xy[1], yaw])
        break
    if start is None:
      continue
    n_obs = int(b.rng.integers(lo, hi + 1))
    keepout = [(obj.center[0], obj.center[1], 1.2), (*place.center, 0.9), (start[0], start[1], 1.0)]
    for _ in range(n_obs):
      sx, sy = b.u(0.2, 0.6), b.u(0.2, 0.6)
      xy = b.free_xy(max(sx, sy) / 2 + 0.05, avoid=keepout)
      if xy is not None:
        b.add(_box(xy[0], xy[1], sx, sy, 0, b.u(0.3, 1.2)), 'obstacle')
    boxes = b.arr()
    occ = occupancy_grid(boxes, b.room)
    if not _connected(occ, b.room, start[:2], [obj.center[:2], place.center], 0.9):
      # Drop the clutter and retry the check once before resampling.
      b.boxes, b.kinds = b.boxes[:n_furn], b.kinds[:n_furn]
      boxes = b.arr()
      occ = occupancy_grid(boxes, b.room)
      if not _connected(occ, b.room, start[:2], [obj.center[:2], place.center], 0.9):
        continue
    approach = approach_free(obj, boxes)
    if not approach.any():
      continue    # e.g. both narrow faces of a flat box blocked: resample the layout
    return Scene(
        split=split, seed=int(seed), scene_type=int(stype), room=b.room.copy(),
        boxes=boxes, kinds=tuple(b.kinds), obj=obj, place=place, start=start,
        approach=approach, occupancy=occ, dirs=object_dirs(obj))
  raise RuntimeError(f'could not build a scene for {split}/{seed}')


def sample_seed(split, rng, scene_type=None, shape=None, cfg=None):
  lo, hi = SPLITS[split]['seeds']
  for _ in range(100000):
    seed = int(rng.integers(lo, hi))
    if scene_type is not None and scene_type_of(split, seed) != scene_type:
      continue
    if shape is not None and object_shape_of(split, seed, cfg) != shape:
      continue
    return seed
  raise RuntimeError(f'no seed found for {split}/{scene_type}/{shape}')
