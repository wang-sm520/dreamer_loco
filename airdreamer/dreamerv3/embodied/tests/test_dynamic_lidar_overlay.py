import pytest


torch = pytest.importorskip('torch')

import embodied.envs.omni.dynamic_geometry as dynamic_geometry
from embodied.envs.omni.dynamic_geometry import overlay_dynamic_obstacles_on_lidar
from embodied.envs.omni.sensors.ray_math import ray_sphere_batch


def test_ray_sphere_batch_supports_per_sphere_radii():
  origins = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
  directions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
  sphere_centers = torch.tensor([[10.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=torch.float32)
  radii = torch.tensor([0.5, 3.5], dtype=torch.float32)

  dists = ray_sphere_batch(origins, directions, sphere_centers, radii, 100.0)

  assert dists.shape == (1,)
  assert torch.allclose(dists, torch.tensor([4.5], dtype=torch.float32))


def test_overlay_dynamic_obstacles_on_lidar_culls_far_obstacles_but_keeps_reaching_box(monkeypatch):
  lidar_range = 10.0
  best_dists = torch.full((1,), lidar_range, dtype=torch.float32)
  ray_origins_w = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
  ray_dirs_unit = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
  obstacle_geometries = [
      ('sphere', torch.tensor([[30.0, 0.0, 0.0]], dtype=torch.float32), 1.0),
      ('box', torch.tensor([[10.2, 0.0, 0.0]], dtype=torch.float32), torch.tensor([[1.0, 0.1, 0.1]], dtype=torch.float32)),
      ('sphere', torch.tensor([[6.0, 0.0, 0.0]], dtype=torch.float32), 1.0),
  ]
  seen = {'boxes': 0, 'spheres': 0}

  def _fake_box_batch(origins, directions, box_centers, half_extents, max_dist):
    seen['boxes'] = int(box_centers.shape[0])
    return torch.full((origins.shape[0],), max_dist, dtype=origins.dtype, device=origins.device)

  def _fake_sphere_batch(origins, directions, sphere_centers, radius, max_dist):
    seen['spheres'] = int(sphere_centers.shape[0])
    return torch.full((origins.shape[0],), max_dist, dtype=origins.dtype, device=origins.device)

  monkeypatch.setattr(dynamic_geometry, 'ray_box_batch', _fake_box_batch)
  monkeypatch.setattr(dynamic_geometry, 'ray_sphere_batch', _fake_sphere_batch)

  overlaid = overlay_dynamic_obstacles_on_lidar(
      best_dists,
      ray_origins_w,
      ray_dirs_unit,
      obstacle_geometries,
      lidar_range=lidar_range,
      device='cpu',
  )

  assert overlaid.shape == (1,)
  assert torch.allclose(overlaid, torch.tensor([lidar_range], dtype=torch.float32))
  assert seen == {'boxes': 1, 'spheres': 1}
