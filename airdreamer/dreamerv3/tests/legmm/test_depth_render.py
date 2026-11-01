import math

import numpy as np
import pytest

from embodied.envs.legmm import geometry as geo
from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import kin_backend
from legmm_utils import env_cfg, wall_scene


def test_ray_box_known_distance():
  rays = geo.camera_rays(I.DEPTH_W, I.DEPTH_H, 87.0, 58.0)
  box = np.array([[2.0, -10, -10, 3.0, 10, 10]])
  t = geo.ray_boxes(np.zeros(3), rays, box)
  # z-depth of a fronto-parallel plane is constant over the whole image.
  np.testing.assert_allclose(t, 2.0, rtol=1e-5)
  # Behind the camera: no hit. Camera inside: exit distance.
  assert np.all(np.isinf(geo.ray_boxes(np.zeros(3), rays, np.array([[-3.0, -1, -1, -2.0, 1, 1]]))))
  inside = geo.ray_boxes(np.zeros(3), rays, np.array([[-1.0, -10, -10, 1.0, 10, 10]]))
  np.testing.assert_allclose(inside, 1.0, rtol=1e-5)


def test_ray_cylinder_and_floor():
  rays = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, -0.5]])
  cyl = np.array([[2.0, 0.0, 0.1, -1.0, 1.0]])
  t = geo.ray_cylinders(np.zeros(3), rays, cyl)
  assert t[0] == pytest.approx(1.9)
  tf = geo.ray_floor(np.array([0.0, 0.0, 0.5]), rays)
  assert np.isinf(tf[0]) and tf[1] == pytest.approx(1.0)


@pytest.mark.parametrize('dist', [0.8, 1.5, 3.0])
def test_head_depth_pixel_values(dist):
  cfg = env_cfg()
  cfg['camera_head'] = dict(cfg['camera_head'], mount_pitch_deg=0.0, mount_xyz=(0.30, 0.0, 0.10))
  b = kin_backend.KinBackend(cfg)
  b.reset(wall_scene(dist + 0.30), np.random.default_rng(0))
  b.state.h, b.state.pitch = 0.45, 0.0
  z = b.render_true('head')
  # The center row looks straight at the wall face; the arm is stowed above.
  center = z[I.DEPTH_H // 2 - 2: I.DEPTH_H // 2 + 2, I.DEPTH_W // 2 - 2: I.DEPTH_W // 2 + 2]
  np.testing.assert_allclose(center, dist, atol=1e-4)
  q = b.render('head', noise=False)
  max_z = cfg['camera_head']['max_z']
  back = q[I.DEPTH_H // 2, I.DEPTH_W // 2, 0] / 255 * max_z
  assert abs(back - dist) <= max_z / 255


def test_min_z_is_invalid():
  cfg = env_cfg()
  cfg['camera_head'] = dict(cfg['camera_head'], mount_pitch_deg=0.0, mount_xyz=(0.30, 0.0, 0.10))
  b = kin_backend.KinBackend(cfg)
  b.reset(wall_scene(0.30 + 0.20), np.random.default_rng(0))   # 0.20 m < min_z 0.28
  q = b.render('head', noise=False)
  assert np.all(q[I.DEPTH_H // 2, :, 0] == 0)


def test_arm_occludes_head_camera():
  cfg = env_cfg()
  cfg['camera_head'] = dict(cfg['camera_head'], mount_pitch_deg=0.0)
  b = kin_backend.KinBackend(cfg)
  b.reset(wall_scene(3.0), np.random.default_rng(0))
  free = b.render_true('head')
  b.state.ee_b = b.clip_ee(np.array([0.75, 0.0, 0.10]), b.state)
  occluded = b.render_true('head')
  assert (occluded < free - 0.5).mean() > 0.01
