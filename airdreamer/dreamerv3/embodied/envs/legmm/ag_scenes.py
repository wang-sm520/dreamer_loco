"""v4 tabletop scenes for legmm_ag (brief §2), fully determined by (split, seed).

One table (height 0.50-0.65, width 0.6-1.2 along x, depth 0.6-1.0 along y) with one target on it; the
target sits 0.05-0.25 m from its nearest edge (uniform along that edge), flat box (70%, only the narrow
faces graspable) or cylinder (30%), flat-box yaw uniform over 0-180 deg. 50% of the scenes block the
nearest edge with an actual solid box in scene.boxes (a room wall the table is pushed against, or a
tall furniture box flush to the edge), the other 50% leave every edge free. Floor clutter stays outside
a keep-out band around the table so the free edges stay free, but it fills the depth image.

The stratum (shape, blocked, blocker kind) depends on (split, seed) only. Solvability is a property of
the fixed-base skill and is computed in ag_skill (solvable_scene); this module never filters.
"""
import dataclasses
import math
import zlib

import numpy as np

from . import geometry as geo
from . import scenes as sc

AG_SPLITS = {'train': (0, 10_000), 'val': (10_000, 11_000), 'test': (20_000, 21_000)}
# Edge index -> outward normal in the world frame. Edge 0 is +y, as in the geometry audit.
EDGE_NORMALS = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
CLEARANCE = 1.3            # free floor kept around the table on the sides that are not blocked
CLUTTER_KEEPOUT = 1.1      # clutter boxes stay at least this far from the table top rectangle
START_KEEPOUT = 1.0
MAX_ATTEMPTS = 60


@dataclasses.dataclass
class AgScene(sc.Scene):
  meta: dict = dataclasses.field(default_factory=dict)


def default_scene_cfg(cfg=None):
  s = dict(table_height=(0.50, 0.65), table_width=(0.6, 1.2), table_depth=(0.6, 1.0),
           edge_dist=(0.05, 0.25), p_blocked=0.5, p_wall=0.5, blocker_thick=(0.3, 0.5),
           blocker_height=(0.8, 1.3), blocker_overhang=(0.1, 0.4), clutter=(3, 8),
           start_min_dist=1.5, flat_frac=0.7)
  s.update(dict((cfg or {}).get('scene', {})))
  return s


def seed_range(split):
  return AG_SPLITS[split]


def stratum_of(split, seed, cfg=None):
  """(shape, blocked, blocker kind) from (split, seed) alone; independent of layout attempts."""
  s = default_scene_cfg(cfg)
  frac = float((cfg or {}).get('object', {}).get('flat_frac', s['flat_frac']))
  rng = np.random.default_rng([zlib.crc32(split.encode()), int(seed), 0xA6, 0x57A7])
  shape = 'flat' if rng.random() < frac else 'cyl'
  blocked = bool(rng.random() < float(s['p_blocked']))
  kind = ('wall' if rng.random() < float(s['p_wall']) else 'furniture') if blocked else 'none'
  return shape, blocked, kind


def sample_seed(split, rng, shape=None, blocked=None, cfg=None):
  """Random seed of a split with the requested stratum (for resampling inside a stratum)."""
  lo, hi = AG_SPLITS[split]
  for _ in range(100000):
    seed = int(rng.integers(lo, hi))
    sh, bl, _ = stratum_of(split, seed, cfg)
    if (shape is None or sh == shape) and (blocked is None or bl == blocked):
      return seed
  raise RuntimeError(f'no seed for {split}/{shape}/{blocked}')


def _extent(obj_half, R, axis):
  """Half extent of the object's footprint along a horizontal unit vector."""
  a = np.array([axis[0], axis[1], 0.0])
  return float(sum(obj_half[i] * abs(a @ R[:, i]) for i in range(2)))


def _rect_gap(cx, cy, hx, hy, box):
  """Euclidean gap between an axis-aligned rectangle and a box's xy footprint (0 if they touch)."""
  dx = max(box[0] - (cx + hx), (cx - hx) - box[3], 0.0)
  dy = max(box[1] - (cy + hy), (cy - hy) - box[4], 0.0)
  return math.hypot(dx, dy)


def _make_object(rng, shape, ocfg, support_z):
  """Sizes and (for flat boxes) yaw; position is set by the caller."""
  u = lambda lo, hi: float(rng.uniform(lo, hi))
  if shape == 'flat':
    f = ocfg.get('flat', {})
    t, w, h = (u(*f.get('thick', (0.03, 0.05))), u(*f.get('wide', (0.10, 0.14))),
               u(*f.get('height', (0.12, 0.15))))
    return np.array([t / 2, w / 2, h / 2]), u(0.0, math.pi)
  c = ocfg.get('cyl', {})
  d, h = u(*c.get('diameter', (0.04, 0.07))), u(*c.get('height', (0.08, 0.20)))
  return np.array([d / 2, d / 2, h / 2]), 0.0


def _try_make(split, seed, cfg, attempt):
  s = default_scene_cfg(cfg)
  ocfg = dict((cfg or {}).get('object', {}))
  shape, blocked, kind = stratum_of(split, seed, cfg)
  rng = np.random.default_rng([zlib.crc32(split.encode()), int(seed), int(attempt), 0xA6, 0xC0DE])
  u = lambda lo, hi: float(rng.uniform(lo, hi))
  room = np.asarray((cfg or {}).get('room_size', (8.0, 8.0)), np.float64)
  W, L = room
  boxes, kinds = [], []
  for w in sc._walls(room):
    boxes.append(w)
    kinds.append('wall')

  sx, sy, th = u(*s['table_width']), u(*s['table_depth']), u(*s['table_height'])
  half_t = np.array([sx / 2, sy / 2])
  edge = int(rng.integers(4))
  n = EDGE_NORMALS[edge]
  t = np.array([-n[1], n[0]])
  half_n = sy / 2 if n[0] == 0 else sx / 2
  len_e = sx if n[0] == 0 else sy
  m = CLEARANCE
  tc = np.array([u(-W / 2 + m + sx / 2, W / 2 - m - sx / 2), u(-L / 2 + m + sy / 2, L / 2 - m - sy / 2)])
  gap = u(0.0, 0.03)
  if blocked and kind == 'wall':
    # Push the table against the room wall on the nearest edge's side (flush up to `gap`).
    wall = (L / 2 if n[1] != 0 else W / 2) * (n[1] + n[0])
    axis = 1 if n[1] != 0 else 0
    tc[axis] = wall - n[axis] * (gap + half_n)
  table = sc._box(tc[0], tc[1], sx, sy, 0.0, th)
  boxes.append(table)
  kinds.append('table')

  half, yaw = _make_object(rng, shape, ocfg, th)
  R = geo.rot_z(yaw)
  edge_d, along = None, None
  for _ in range(100):
    d = u(*s['edge_dist'])
    if shape == 'flat':
      yaw = u(0.0, math.pi)
      R = geo.rot_z(yaw)
    en = _extent(half, R, n)
    et = _extent(half, R, t)
    if d < en + 0.005:
      continue                               # the box would overhang the table edge
    if len_e - 2 * d < 1e-6:
      continue
    a = u(d, len_e - d)                      # uniform along the edge; the other edges are >= d away
    if a < et + 0.005 or len_e - a < et + 0.005:
      continue
    edge_d, along = d, a
    break
  if edge_d is None:
    return None
  oxy = tc + n * (half_n - edge_d) + t * (along - len_e / 2)
  obj = sc.SceneObject(shape, np.array([oxy[0], oxy[1], th + half[2]]), half, th, yaw=yaw)

  blocker = None
  if blocked and kind == 'furniture':
    thick, over, hb = u(*s['blocker_thick']), u(*s['blocker_overhang']), u(*s['blocker_height'])
    c = tc + n * (half_n + gap + thick / 2)
    size = (thick, len_e + 2 * over) if n[0] != 0 else (len_e + 2 * over, thick)
    blocker = sc._box(c[0], c[1], size[0], size[1], 0.0, hb)
    boxes.append(blocker)
    kinds.append('blocker')
  n_core = len(boxes)

  # Robot start: free floor, out of reach, with a connected path to the table.
  arr = np.stack(boxes)
  start = None
  for _ in range(200):
    xy = np.array([u(-W / 2 + 0.6, W / 2 - 0.6), u(-L / 2 + 0.6, L / 2 - 0.6)])
    if np.linalg.norm(xy - obj.center[:2]) < float(s['start_min_dist']):
      continue
    yaw0 = u(-math.pi, math.pi)
    if sc.footprint_free(arr, xy[0], xy[1], yaw0, 0.15):
      start = np.array([xy[0], xy[1], yaw0])
      break
  if start is None:
    return None

  # Floor clutter: outside the keep-out band around the table (so free edges stay free).
  lo, hi = s['clutter']
  for _ in range(int(rng.integers(lo, hi + 1))):
    cs = (u(0.2, 0.6), u(0.2, 0.6))
    for _try in range(60):
      xy = np.array([u(-W / 2 + cs[0] / 2 + 0.05, W / 2 - cs[0] / 2 - 0.05),
                     u(-L / 2 + cs[1] / 2 + 0.05, L / 2 - cs[1] / 2 - 0.05)])
      if _rect_gap(xy[0], xy[1], cs[0] / 2, cs[1] / 2, table) < CLUTTER_KEEPOUT:
        continue
      if blocker is not None and _rect_gap(xy[0], xy[1], cs[0] / 2, cs[1] / 2, blocker) < 0.5:
        continue
      if np.linalg.norm(xy - start[:2]) < START_KEEPOUT + 0.5 * max(cs):
        continue
      boxes.append(sc._box(xy[0], xy[1], cs[0], cs[1], 0.0, u(0.3, 1.2)))
      kinds.append('clutter')
      break
  arr = np.stack(boxes)
  occ = sc.occupancy_grid(arr, room)
  if not sc._connected(occ, room, start[:2], [obj.center[:2]], 0.9):
    arr, kinds = arr[:n_core], kinds[:n_core]
    occ = sc.occupancy_grid(arr, room)
    if not sc._connected(occ, room, start[:2], [obj.center[:2]], 0.9):
      return None

  dirs = sc.object_dirs(obj)
  approach = np.array([sc.graspable(obj, d) for d in dirs[:sc.N_DIRS]] + [False])
  alpha = 0.0
  if shape == 'flat':
    wv = obj.R[:2, 1]                         # width axis vs the nearest edge's tangent, folded to [0, 90]
    alpha = math.degrees(math.acos(min(1.0, abs(float(wv @ t)))))
  meta = dict(shape=shape, blocked=blocked, blocker=kind, edge=edge, edge_dist=float(edge_d),
              along=float(along), edge_len=float(len_e), table=[float(sx), float(sy), float(th)],
              table_center=[float(tc[0]), float(tc[1])], alpha_fold_deg=float(alpha),
              normal=n.tolist(), attempt=int(attempt))
  place = sc.PlaceZone(np.array([3.5, 3.5]), np.array([0.175, 0.175]), 0.0)   # unused in legmm_ag
  return AgScene(
      split=split, seed=int(seed), scene_type=0, room=room.copy(), boxes=arr, kinds=tuple(kinds),
      obj=obj, place=place, start=start, approach=approach, occupancy=occ, dirs=dirs, meta=meta)


def make_ag_scene(split, seed, cfg=None):
  for attempt in range(MAX_ATTEMPTS):
    scene = _try_make(split, seed, cfg, attempt)
    if scene is not None:
      return scene
  raise RuntimeError(f'could not build an ag scene for {split}/{seed}')


def head_on_stance(scene, rho=0.55, shoulder_dx=0.03):
  """Base pose (x, y, yaw) of a head-on stance at the nearest edge: shoulder at rho from the object
  along the edge normal, facing the object. Used by tests and the calibration policies."""
  n = np.asarray(scene.meta['normal'])
  sh = scene.obj.center[:2] + rho * n
  yaw = math.atan2(-n[1], -n[0])
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([sh[0] - c * shoulder_dx, sh[1] - s * shoulder_dx, yaw])
