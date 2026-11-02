"""Gallery of scenes per type: top-down sketch + head/wrist depth at a pose in
front of the object with the arm reaching towards it. Sanity check for M1.

python tools/legmm_scene_gallery.py --out docs/legmm/figs/m1_scenes.png
"""
import argparse
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import kin_backend, scenes, viz


def demo_pose(backend, scene):
  """Stand in front of the object along the first free approach direction."""
  obj = scene.obj
  free = np.where(scene.approach[:scenes.N_DIRS])[0]
  dirs = scenes.APPROACH_DIRS[free] if len(free) else scenes.APPROACH_DIRS[:1]
  s = backend.state
  for d in dirs:
    for dist in (0.75, 0.9, 1.1, 1.4):
      x, y = obj.center[0] + dist * d[0], obj.center[1] + dist * d[1]
      yaw = math.atan2(-d[1], -d[0])
      if scenes.footprint_free(scene.boxes, x, y, yaw):
        s.x, s.y, s.yaw = x, y, yaw
        s.h = float(np.clip(obj.center[2] - 0.05, *I.BASE_HEIGHT_RANGE))
        target_w = obj.center + 0.12 * np.array([d[0], d[1], 0.0])
        s.ee_b = backend.clip_ee(backend.R_body().T @ (target_w - backend.base_pos()), s)
        return True
  return False


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out', default='docs/legmm/figs/m1_scenes.png')
  p.add_argument('--per_type', type=int, default=2)
  a = p.parse_args()
  cfg = {}
  rows = []
  for stype in I.SceneType:
    split = 'test_constrained' if stype in (I.SceneType.CORNER, I.SceneType.CUBBY) else 'train'
    lo, hi = I.SPLITS[split]['seeds']
    found = 0
    for seed in range(lo, hi):
      sc = scenes.make_scene(split, seed, cfg)
      if sc.scene_type == int(stype):
        rows.append(sc)
        found += 1
        if found == a.per_type:
          break
  fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.6 * len(rows)))
  for r, sc in enumerate(rows):
    sys.path.insert(0, str(ROOT / 'tests' / 'legmm'))
    from legmm_utils import env_cfg
    b = kin_backend.KinBackend(env_cfg())
    b.reset(sc, np.random.default_rng(0))
    demo_pose(b, sc)
    viz.topdown(axes[r, 0], sc, b.state, b)
    axes[r, 1].imshow(viz.depth_rgb(b.render('head', noise=True)))
    axes[r, 1].set_title('head depth', fontsize=8)
    axes[r, 2].imshow(viz.depth_rgb(b.render('wrist', noise=True)))
    axes[r, 2].set_title('wrist depth', fontsize=8)
    for ax in axes[r, 1:]:
      ax.axis('off')
  fig.tight_layout()
  out = pathlib.Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=90)
  print('wrote', out)


if __name__ == '__main__':
  main()
