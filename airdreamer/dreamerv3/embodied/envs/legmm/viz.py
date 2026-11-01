"""Top-down sketches and depth panels (sanity checks, QA packets, videos)."""
import math

import numpy as np

from . import interfaces as I
from . import scenes

KIND_COLORS = {
    'wall': '#555555', 'table': '#b5895a', 'obstacle': '#7a9cc6',
    'cubby': '#9c6ad1', 'place_table': '#d9b38c'}


def topdown(ax, scene, state=None, backend=None, title=None, path_xy=None):
  import matplotlib.patches as mp
  W, L = scene.room
  ax.set_xlim(-W / 2 - 0.2, W / 2 + 0.2)
  ax.set_ylim(-L / 2 - 0.2, L / 2 + 0.2)
  ax.set_aspect('equal')
  for b, kind in zip(scene.boxes, scene.kinds):
    ax.add_patch(mp.Rectangle((b[0], b[1]), b[3] - b[0], b[4] - b[1],
                              color=KIND_COLORS.get(kind, 'k'), alpha=0.8, lw=0))
  z = scene.place
  ax.add_patch(mp.Rectangle(z.center - z.half, *(2 * z.half), fill=False, ec='g', lw=1.5))
  obj = scene.obj if state is None else None
  c = scene.obj.center if state is None else state.obj_center
  ax.add_patch(mp.Circle(c[:2], max(scene.obj.half[0], 0.04), color='r'))
  for i, ok in enumerate(scene.approach[:scenes.N_DIRS]):
    d = scenes.APPROACH_DIRS[i]
    ax.plot([c[0], c[0] + 0.3 * d[0]], [c[1], c[1] + 0.3 * d[1]],
            color='g' if ok else 'r', lw=1)
  del obj
  if path_xy is not None and len(path_xy):
    p = np.asarray(path_xy)
    ax.plot(p[:, 0], p[:, 1], 'c-', lw=1)
  x, y, yaw = (scene.start if state is None else (state.x, state.y, state.yaw))
  hx, hy = scenes.FOOT_HALF
  corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
  Rz = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
  ax.add_patch(mp.Polygon(corners @ Rz.T + [x, y], fill=False, ec='k', lw=1.2))
  ax.arrow(x, y, 0.5 * math.cos(yaw), 0.5 * math.sin(yaw), head_width=0.1, color='k')
  if backend is not None and state is not None:
    pts = backend.arm_points(state)
    ax.plot(pts[:, 0], pts[:, 1], 'm.-', lw=1)
  names = [t.name for t in I.SceneType]
  ax.set_title(title or f'{scene.split}/{scene.seed} {names[scene.scene_type]}', fontsize=8)
  ax.tick_params(labelsize=6)


def depth_rgb(q):
  """uint8 depth (H, W, 1) -> RGB with invalid pixels in red."""
  q = np.asarray(q)[..., 0]
  img = np.stack([q, q, q], -1).astype(np.uint8)
  img[q == 0] = [200, 0, 0]
  return img
