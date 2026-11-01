"""Measures what the placeholder arm/camera mounts imply at a pre-grasp pose
(gripper 10 cm back from the grasp point along a free horizontal approach,
brief §7.1): head-camera pixels occluded by the arm, and the wrist-camera
z-depth of the object compared with min_z.

python tools/legmm_mount_check.py --n 50 --out docs/legmm/m1_mount_check.json
"""
import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests' / 'legmm'))

import numpy as np

from embodied.envs.legmm import kin_backend, scenes
from embodied.envs.legmm import interfaces as I


def pregrasp(b, sc):
  s = b.state
  for i in np.where(sc.approach[:scenes.N_DIRS])[0]:
    d = scenes.APPROACH_DIRS[i]
    c = s.obj_center
    for stand in (0.45, 0.55, 0.65, 0.75):
      x, y = c[0] + stand * d[0], c[1] + stand * d[1]
      yaw = math.atan2(-d[1], -d[0])
      if not scenes.footprint_free(sc.boxes, x, y, yaw):
        continue
      s.x, s.y, s.yaw = x, y, yaw
      s.h = float(np.clip(c[2] + 0.05, 0.25, 0.55))
      s.pitch = -0.3 if c[2] < 0.3 else 0.0
      R = b.R_body()
      a_b = R.T @ (-d)
      y_b = np.cross([0, 0, 1.0], a_b); y_b /= np.linalg.norm(y_b)
      s.R_ee_b = np.stack([a_b, y_b, np.cross(a_b, y_b)], 1)
      target = c + 0.10 * d
      s.ee_b = b.clip_ee(R.T @ (target - b.base_pos()), s)
      if np.linalg.norm(b.ee_world()[0] - target) < 0.02:
        return True
  return False


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--n', type=int, default=50)
  p.add_argument('--out', default='')
  a = p.parse_args()
  from legmm_utils import env_cfg
  cfg = env_cfg()
  rows = []
  for seed in range(400):
    sc = scenes.make_scene('train', seed, cfg)
    b = kin_backend.KinBackend(cfg)
    b.reset(sc, np.random.default_rng(seed))
    if not pregrasp(b, sc):
      continue
    with_arm = b.render_true('head')
    pts = b.arm_points
    b.arm_points = lambda s=None: np.zeros((0, 3))
    no_arm = b.render_true('head')
    b.arm_points = pts
    occl = float(((with_arm < no_arm - 0.05)).mean())
    min_z = b.cams['head']['model'].min_z
    head_invalid = float((with_arm < min_z).mean())
    pos, R = b.camera_pose('wrist')
    obj_z = float((b.state.obj_center - pos) @ R[:, 0])
    wrist = b.render_true('wrist')
    rows.append(dict(seed=seed, type=int(sc.scene_type), head_arm_occluded=occl,
                     head_below_min_z=head_invalid, wrist_obj_zdepth=obj_z,
                     wrist_below_min_z=float((wrist < b.cams['wrist']['model'].min_z).mean())))
    if len(rows) >= a.n:
      break
  arr = lambda k: np.array([r[k] for r in rows])
  res = dict(
      n=len(rows),
      head_arm_occluded_mean=float(arr('head_arm_occluded').mean()),
      head_below_min_z_mean=float(arr('head_below_min_z').mean()),
      wrist_obj_zdepth_median=float(np.median(arr('wrist_obj_zdepth'))),
      wrist_obj_below_min_z_frac=float((arr('wrist_obj_zdepth') < cfg['camera_wrist']['min_z']).mean()),
      wrist_below_min_z_mean=float(arr('wrist_below_min_z').mean()),
      config=dict(arm_mount=cfg['arm']['mount_xyz'], head=cfg['camera_head'], wrist=cfg['camera_wrist']))
  print(json.dumps(res, indent=1, default=list))
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(dict(summary=res, rows=rows), indent=1, default=list))


if __name__ == '__main__':
  main()
