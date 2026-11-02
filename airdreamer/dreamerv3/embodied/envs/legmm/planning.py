"""Grid path planning for the scripted expert (brief §7.1).

Shortest paths on the 5 cm occupancy grid with 8-connectivity. A single
Dijkstra run from the robot cell gives distances to every candidate staging
cell at once (equivalent to A* to each of them, but one pass).
"""
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import scenes

_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class GridPlanner:

  def __init__(self, scene, inflate=0.45):
    self.scene = scene
    self.room = scene.room
    self.res = scenes.GRID_RES
    self.inflate = inflate
    self.occ = scenes.occupancy_grid(scene.boxes, scene.room, inflate)
    self.shape = self.occ.shape
    self._graph = None
    self._start = None
    self._dist = None
    self._pred = None

  def cell(self, xy):
    i, j = scenes.world_to_cell(xy, self.room, self.res)
    return int(np.clip(i, 0, self.shape[0] - 1)), int(np.clip(j, 0, self.shape[1] - 1))

  def world(self, ij):
    return scenes.cell_to_world(ij, self.room, self.res)

  def free(self, xy):
    return not self.occ[self.cell(xy)]

  def _build(self, occ):
    nx, ny = occ.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, wts = [], [], []
    free = ~occ
    for di, dj in _NEIGH:
      a = free[max(0, -di): nx - max(0, di), max(0, -dj): ny - max(0, dj)]
      b = free[max(0, di): nx - max(0, -di) or None, max(0, dj): ny - max(0, -dj) or None]
      ok = a & b
      src = idx[max(0, -di): nx - max(0, di), max(0, -dj): ny - max(0, dj)][ok]
      dst = idx[max(0, di): nx - max(0, -di) or None, max(0, dj): ny - max(0, -dj) or None][ok]
      rows.append(src)
      cols.append(dst)
      wts.append(np.full(len(src), math.hypot(di, dj) * self.res))
    rows, cols, wts = map(np.concatenate, (rows, cols, wts))
    return csr_matrix((wts, (rows, cols)), shape=(nx * ny, nx * ny))

  def solve_from(self, xy):
    """Dijkstra from xy. Cells near the start are freed if the robot already
    sits inside the inflation margin."""
    occ = self.occ.copy()
    si, sj = self.cell(xy)
    if occ[si, sj]:
      r = int(math.ceil(self.inflate / self.res))
      ii, jj = np.ogrid[:occ.shape[0], :occ.shape[1]]
      hard = scenes.occupancy_grid(self.scene.boxes, self.room, 0.25)
      disk = (ii - si) ** 2 + (jj - sj) ** 2 <= r * r
      occ[disk & ~hard] = False
      occ[si, sj] = False
    self._occ_used = occ
    graph = self._build(occ)
    start = si * self.shape[1] + sj
    self._dist, self._pred = dijkstra(graph, indices=start, return_predecessors=True)
    self._start = start
    return self

  def dist_to(self, xy):
    i, j = self.cell(xy)
    return float(self._dist[i * self.shape[1] + j])

  def path_to(self, xy):
    i, j = self.cell(xy)
    node = i * self.shape[1] + j
    if not np.isfinite(self._dist[node]):
      return None
    cells = []
    while node != self._start and node >= 0:
      cells.append(divmod(int(node), self.shape[1]))
      node = self._pred[node]
    cells.append(divmod(int(self._start), self.shape[1]))
    pts = np.array([self.world(c) for c in reversed(cells)])
    return shortcut(pts, self._occ_used, self)


def shortcut(pts, occ, planner, step=0.05):
  """Greedy line-of-sight shortcutting on the occupancy grid."""
  if len(pts) <= 2:
    return pts
  out = [pts[0]]
  i = 0
  while i < len(pts) - 1:
    j = len(pts) - 1
    while j > i + 1 and not _visible(pts[i], pts[j], occ, planner, step):
      j -= 1
    out.append(pts[j])
    i = j
  return np.array(out)


def _visible(a, b, occ, planner, step):
  n = max(2, int(np.linalg.norm(b - a) / step) + 1)
  for t in np.linspace(0, 1, n):
    if occ[planner.cell(a * (1 - t) + b * t)]:
      return False
  return True


def pure_pursuit(pos, yaw, path, lookahead=0.35, vmax=0.6, wmax=1.0, k_yaw=2.0):
  """Returns (vx, wz, remaining) for following a polyline path. The lookahead
  point is measured along the path from the projection of pos onto it."""
  path = np.asarray(path, np.float64)
  if len(path) == 1:
    path = np.stack([pos, path[0]])
  best = (np.inf, 0, 0.0)
  for k in range(len(path) - 1):
    a, b = path[k], path[k + 1]
    ab = b - a
    t = float(np.clip((pos - a) @ ab / max(ab @ ab, 1e-12), 0, 1))
    d = float(np.linalg.norm(a + t * ab - pos))
    if d < best[0]:
      best = (d, k, t)
  _, k, t = best
  # Arc length along the path from the projection.
  pts = [path[k] + t * (path[k + 1] - path[k])] + list(path[k + 1:])
  target = pts[-1]
  acc = 0.0
  for p0, p1 in zip(pts[:-1], pts[1:]):
    seg = float(np.linalg.norm(p1 - p0))
    if acc + seg >= lookahead:
      target = p0 + (lookahead - acc) / max(seg, 1e-9) * (p1 - p0)
      break
    acc += seg
  remaining = float(sum(np.linalg.norm(p1 - p0) for p0, p1 in zip(pts[:-1], pts[1:])))
  remaining += best[0]
  dx, dy = target - pos
  err = wrap(math.atan2(dy, dx) - yaw)
  wz = float(np.clip(k_yaw * err, -wmax, wmax))
  vx = 0.0 if abs(err) > 0.6 else vmax * math.cos(err) * min(1.0, remaining / 0.5 + 0.2)
  return float(vx), wz, remaining


def wrap(a):
  return (a + math.pi) % (2 * math.pi) - math.pi
