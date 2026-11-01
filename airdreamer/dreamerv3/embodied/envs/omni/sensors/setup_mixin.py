"""Shared LiDAR and depth camera setup helpers for omni envs."""

from .depth import DepthCamera
from .lidar import LidarSensor


def setup_lidar(env, prim_utils):
  ground_path = str(env._lidar_mesh_prim)
  if not prim_utils.is_prim_path_valid(ground_path):
    print(f"Warning: LiDAR mesh_prim '{ground_path}' not found, LiDAR disabled.")
    return None

  lidar_sensor = LidarSensor(device=str(env._device))
  lidar_sensor.create(
      drone_prim_path=env._drone_prim_path,
      cfg={
          'range': float(env._lidar_range),
          'vfov': tuple(env._lidar_vfov),
          'vres': int(env._lidar_vres),
          'hres': int(env._lidar_hres),
          'hfov': float(env._lidar_hfov),
      },
      mesh_prim_paths=[ground_path],
  )
  try:
    lidar_sensor.rebuild_mesh(env._arena.scene)
  except Exception:
    pass
  return lidar_sensor


def setup_depth_camera(env):
  depth_camera = DepthCamera(device=str(env._device))
  depth_camera.create(
      env._drone_prim_path,
      cfg={
          'resolution': tuple(env._depth_resolution),
          'max_range': float(env._depth_max),
          'focal_length': float(env._depth_focal_length),
          'horizontal_aperture': float(env._depth_horizontal_aperture),
          'clipping_range': tuple(env._depth_clipping_range),
          'translation': tuple(env._depth_translation),
          'pitch_deg': float(env._depth_pitch_deg),
      },
  )
  return depth_camera


__all__ = ['setup_lidar', 'setup_depth_camera']
