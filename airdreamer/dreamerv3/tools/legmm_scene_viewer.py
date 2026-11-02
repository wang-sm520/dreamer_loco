"""Scene viewer: per scene type a still (3D view, top-down, head/wrist depth at
the expert's grasp pose) and an mp4 of one expert episode.

python tools/legmm_scene_viewer.py --out docs/legmm/scenes
"""
import argparse
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests' / 'legmm'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from embodied.envs.legmm import expert as ex
from embodied.envs.legmm import geometry as geo
from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import kin_backend as kb
from embodied.envs.legmm import scenes, viz

COLORS = dict(viz.KIND_COLORS)


def _box_faces(b):
  x0, y0, z0, x1, y1, z1 = b
  v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
  idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
  return [v[i] for i in idx]


def draw_3d(ax, scene, backend, s, focus, radius=1.3, bad=None):
  for b, kind in zip(scene.boxes, scene.kinds):
    # Only geometry near the focus; walls are clipped to the view window.
    b = b.copy()
    b[0], b[1] = max(b[0], focus[0] - radius), max(b[1], focus[1] - radius)
    b[3], b[4] = min(b[3], focus[0] + radius), min(b[4], focus[1] + radius)
    if b[0] >= b[3] or b[1] >= b[4]:
      continue
    ax.add_collection3d(Poly3DCollection(_box_faces(b), facecolor=COLORS.get(kind, 'gray'),
                                         edgecolor='k', linewidths=0.3, alpha=0.35))
  obj = scene.obj
  ob = np.concatenate([s.obj_center - obj.half, s.obj_center + obj.half])
  ax.add_collection3d(Poly3DCollection(_box_faces(ob), facecolor='red', alpha=0.9))
  z = scene.place
  zx, zy = z.center
  hx, hy = z.half
  ax.plot([zx - hx, zx + hx, zx + hx, zx - hx, zx - hx], [zy - hy, zy - hy, zy + hy, zy + hy, zy - hy],
          [z.support_z + 0.002] * 5, 'g-', lw=2)
  # Robot body (A2 base_link mesh bounds) and legs.
  R, p = backend.R_body(s), backend.base_pos(s)
  bb = kb.BODY_BOX
  corners = np.array([[x, y, zz] for x in (bb[0], bb[3]) for y in (bb[1], bb[4]) for zz in (bb[2], bb[5])])
  cw = p + corners @ R.T
  faces = [cw[[0, 1, 3, 2]], cw[[4, 5, 7, 6]], cw[[0, 1, 5, 4]], cw[[2, 3, 7, 6]], cw[[0, 2, 6, 4]], cw[[1, 3, 7, 5]]]
  ax.add_collection3d(Poly3DCollection(faces, facecolor='#cfd6ee', edgecolor='k', linewidths=0.4, alpha=0.8))
  for hxl, hyl in ((0.26, 0.2), (0.26, -0.2), (-0.26, 0.2), (-0.26, -0.2)):
    top = p + R @ np.array([hxl, hyl, 0.0])
    ax.plot([top[0], top[0]], [top[1], top[1]], [top[2], 0.0], color='0.4', lw=2)
  # Arm: shoulder -> elbow -> wrist -> TCP, plus the wrist camera.
  elbow, _ = backend.elbow_b(s.ee_b, s.R_ee_b)
  chain_b = np.stack([backend.shoulder_b - [0, 0, kb.SHOULDER_Z], backend.shoulder_b, elbow,
                      backend.wrist_b(s.ee_b, s.R_ee_b), s.ee_b])
  chain = p + chain_b @ R.T
  ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], 'k-', lw=3)
  ax.scatter(*chain[-1], color='orange', s=20)
  cam = backend.to_world(backend.wrist_cam_b(s), s)
  ax.scatter(*cam, color='m', s=20, marker='s')
  hc, _ = backend.camera_pose('head', s)
  ax.scatter(*hc, color='b', s=20, marker='s')
  if bad is not None and len(bad):
    ax.scatter(bad[:, 0], bad[:, 1], bad[:, 2], color='red', s=40, marker='x')
  ax.set_xlim(focus[0] - radius, focus[0] + radius)
  ax.set_ylim(focus[1] - radius, focus[1] + radius)
  ax.set_zlim(0, 1.2)
  ax.set_box_aspect((2 * radius, 2 * radius, 1.2))
  ax.view_init(elev=22, azim=math.degrees(s.yaw) - 140)
  ax.tick_params(labelsize=5)


def pick_pose(env, e):
  """Puts the robot at the expert's pick plan with the TCP at the grasp."""
  b, s = env.backend, env.backend.state
  cands = ex.pick_candidates(b, env.scene, ex.radial_bases(s.obj_center[:2], env.scene.boxes))
  cands.sort(key=lambda c: c.score)
  for c in cands:
    if e._transit(c) is not None:
      s.x, s.y, s.yaw = c.base
      s.h = c.h
      Rb = b.R_body()
      s.R_ee_b = Rb.T @ c.R_w
      s.ee_b = Rb.T @ (c.goal_w - b.base_pos())
      return c
  return None


def cubby_attempt(env):
  """Best effort for CUBBY: stand in front, fully crouched, horizontal approach."""
  b, s, sc = env.backend, env.backend.state, env.scene
  i = int(np.argmax(scenes.APPROACH_DIRS[:scenes.N_DIRS, :2] @ (-s.obj_center[:2] / np.linalg.norm(s.obj_center[:2]))))
  d = scenes.APPROACH_DIRS[i][:2]
  for r in np.arange(0.45, 1.0, 0.02):
    x, y = s.obj_center[0] + r * d[0], s.obj_center[1] + r * d[1]
    yaw = math.atan2(-d[1], -d[0])
    if scenes.footprint_free(sc.boxes, x, y, yaw, 0.02):
      break
  s.x, s.y, s.yaw, s.h = x, y, yaw, I.BASE_HEIGHT_RANGE[0]
  a = np.array([-d[0], -d[1], 0.0])
  R_w = ex.grasp_frame(a, yaw)
  Rb = b.R_body()
  s.R_ee_b = Rb.T @ R_w
  target_b = Rb.T @ (s.obj_center - b.base_pos())
  s.ee_b = b.clip_ee(target_b, s)
  pts = b.arm_points(s)
  hit = geo.points_in_boxes(pts, sc.boxes, kb.LINK_RADIUS).any(1)
  reach_short = float(np.linalg.norm(s.ee_b - target_b))
  return pts[hit], reach_short


def run_video(env, e, path, fps=8, max_steps=450):
  import imageio
  frames = []
  traj = []
  obs = None
  skills = ['NAV', 'PICK', 'PLACE']
  for t in range(max_steps):
    act = e.act()
    obs = env.step({**act, 'reset': False})
    s = env.backend.state
    traj.append([s.x, s.y])
    fig = plt.figure(figsize=(10, 3.6), dpi=80)
    ax = fig.add_subplot(1, 3, 1)
    viz.topdown(ax, env.scene, s, env.backend, path_xy=np.array(traj),
                title=f't={t} {skills[int(act["skill"])]} phase={I.Phase(env.backend.phase()).name}')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(viz.depth_rgb(obs['depth_head']))
    ax2.set_title('head D435i depth', fontsize=8)
    ax2.axis('off')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(viz.depth_rgb(obs['depth_wrist']))
    ax3.set_title(f'wrist D405 depth  r={float(obs["reward"]):+.2f}', fontsize=8)
    ax3.axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    if obs['is_last']:
      break
  for _ in range(fps):
    frames.append(frames[-1])
  imageio.mimwrite(str(path), frames, fps=fps, quality=7, macro_block_size=8)
  return I.TERM_REASONS[int(obs['log/term_reason'])], t + 1


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out', default='docs/legmm/scenes')
  p.add_argument('--video', default='True')
  a = p.parse_args()
  from legmm_utils import make_env, zero_action
  out = pathlib.Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  summary = []
  for stype in I.SceneType:
    split = 'test_constrained' if stype in (I.SceneType.CORNER, I.SceneType.CUBBY) else 'train'
    lo, hi = I.SPLITS[split]['seeds']
    seed = next(sd for sd in range(lo, hi) if scenes.scene_type_of(split, sd) == stype)
    env = make_env(split, seed=0, domain_rand={'enabled': False})
    env.forced_seed = seed
    env.step(zero_action(reset=True))
    e = ex.ScriptedExpert(ex.ExpertConfig(diversity=False))
    e.reset(env.scene, np.random.default_rng(0), env.backend)
    snap = env.snapshot()
    start_state = env.backend.snapshot()
    bad, note = None, ''
    plan = pick_pose(env, e)
    if plan is None:
      bad, short = cubby_attempt(env)
      note = (f'no feasible grasp; best effort shown (crouched h={I.BASE_HEIGHT_RANGE[0]}): '
              f'TCP {short:.2f} m short, {len(bad)} arm points in collision (red x)')
    s = env.backend.state
    fig = plt.figure(figsize=(16, 8), dpi=90)
    ax = fig.add_subplot(2, 3, (1, 4), projection='3d')
    focus = 0.5 * (s.obj_center[:2] + np.array([s.x, s.y]))
    draw_3d(ax, env.scene, env.backend, s, focus, bad=bad)
    ax.set_title(f'{stype.name}  {split}/{seed}  3D at the grasp pose\n'
                 'blue box = A2 body, black = AIRBOT chain, magenta = wrist D405, blue square = head D435i',
                 fontsize=9)
    ax = fig.add_subplot(2, 3, (2, 5))
    viz.topdown(ax, env.scene, s, env.backend)
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(viz.depth_rgb(env.backend.render('head')))
    ax.set_title('head D435i depth (red = invalid / < min_z)', fontsize=8)
    ax.axis('off')
    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(viz.depth_rgb(env.backend.render('wrist')))
    ax.set_title('wrist D405 depth', fontsize=8)
    ax.axis('off')
    if note:
      fig.suptitle(note, fontsize=9, color='red')
    fig.tight_layout()
    fig.savefig(out / f'{stype.value}_{stype.name}.png')
    plt.close(fig)
    row = dict(type=stype.name, split=split, seed=seed, grasp_plan=plan is not None, note=note)
    if a.video == 'True':
      env.restore(snap)
      env.backend.restore(start_state)
      e = ex.ScriptedExpert(ex.ExpertConfig(diversity=False))
      e.reset(env.scene, np.random.default_rng(0), env.backend)
      reason, steps = run_video(env, e, out / f'{stype.value}_{stype.name}.mp4')
      row.update(video_result=reason, video_steps=steps)
    summary.append(row)
    print(row)


if __name__ == '__main__':
  main()
