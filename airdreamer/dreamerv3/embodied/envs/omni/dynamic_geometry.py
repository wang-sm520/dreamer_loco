"""Shared dynamic-object geometry helpers for omni drone environments."""

import numpy as np

from embodied.envs.omni.sensors.ray_math import ray_box_batch, ray_sphere_batch


def dynamic_position_np(
    center,
    motion_type,
    motion_axis,
    amplitude,
    omega,
    phase,
    motion_radius,
    t,
):
  """Compute dynamic-obstacle world position at time t."""
  c = np.asarray(center, np.float32).reshape(3)
  motion_kind = str(motion_type or 'line').lower()
  omega = float(omega)
  phase = float(phase)
  t = float(t)

  if motion_kind == 'circle':
    motion_radius = float(motion_radius)
    return np.asarray(
        [
            c[0] + motion_radius * np.cos(omega * t + phase),
            c[1] + motion_radius * np.sin(omega * t + phase),
            c[2],
        ],
        np.float32,
    )

  axis = np.asarray(motion_axis, np.float32).reshape(3)
  norm = float(np.linalg.norm(axis))
  if norm <= 1e-6:
    axis = np.asarray([1.0, 0.0, 0.0], np.float32)
  else:
    axis = axis / norm
  return c + axis * float(amplitude) * np.sin(omega * t + phase)


def point_box_distance_np(point, box_center, half_extents):
  delta = np.abs(point - box_center) - half_extents
  return float(np.linalg.norm(np.maximum(delta, 0.0)))


def dynamic_obstacle_box_half_extents(size, fallback_radius, device):
  import torch

  if isinstance(size, (int, float)):
    raw = [2.0 * float(size)] * 3
  else:
    raw = list(np.asarray(size, np.float32).reshape(-1))
  if not raw:
    raw = [2.0 * float(fallback_radius)] * 3
  while len(raw) < 3:
    raw.append(raw[-1])
  return 0.5 * torch.as_tensor(raw[:3], device=device, dtype=torch.float32).reshape(1, 3)


def dynamic_obstacles_world(arena, *, device, lidar_visible=None, collidable_only=False, fallback_radius=0.3):
  import torch

  obstacles = []
  peer_collidable = True if collidable_only else None
  for dyn in arena.iter_dynamic_objects(
      kind='dynamic_obstacle',
      lidar_visible=lidar_visible,
      peer_collidable=peer_collidable,
  ):
    if dyn.pose_w is None:
      continue
    try:
      center = torch.as_tensor(dyn.pose_w, device=device).reshape(-1, 3)[:1]
    except Exception:
      continue
    shape = str(getattr(dyn, 'shape', 'box') or 'box').lower()
    if shape == 'box':
      obstacles.append((
          'box',
          center,
          dynamic_obstacle_box_half_extents(dyn.size, fallback_radius=float(fallback_radius), device=device),
      ))
      continue
    if isinstance(dyn.size, (int, float)):
      radius = float(dyn.size)
    else:
      size_arr = np.asarray(dyn.size, np.float32).reshape(-1)
      radius = 0.5 * float(np.max(size_arr[:3])) if size_arr.size else float(fallback_radius)
    obstacles.append(('sphere', center, radius))
  return obstacles


def overlay_dynamic_obstacles_on_lidar(
    best_dists,
    ray_origins_w,
    ray_dirs_unit,
    obstacle_geometries,
    *,
    lidar_range,
    device,
):
  import torch

  if ray_origins_w is None or ray_dirs_unit is None or (not obstacle_geometries):
    return best_dists

  dtype = ray_origins_w.dtype
  lidar_range = float(lidar_range)
  ray_origin = torch.as_tensor(ray_origins_w, device=device, dtype=dtype).reshape(-1, 3)[:1]

  box_centers = []
  box_half_extents = []
  sphere_centers = []
  sphere_radii = []

  for shape, center_w, size in obstacle_geometries:
    center = torch.as_tensor(center_w, device=device, dtype=dtype).reshape(-1, 3)[:1]
    if shape == 'box':
      half_extents = torch.as_tensor(size, device=device, dtype=dtype).reshape(-1, 3)[:1]
      bound_radius = torch.norm(half_extents, dim=-1)
      if bool((torch.norm(center - ray_origin, dim=-1) <= (lidar_range + bound_radius)).item()):
        box_centers.append(center)
        box_half_extents.append(half_extents)
      continue
    radius = torch.as_tensor(size, device=device, dtype=dtype).reshape(-1)[:1]
    if bool((torch.norm(center - ray_origin, dim=-1) <= (lidar_range + radius)).item()):
      sphere_centers.append(center)
      sphere_radii.append(radius)

  if box_centers:
    best_dists = torch.minimum(
        best_dists,
        ray_box_batch(
            ray_origins_w,
            ray_dirs_unit,
            torch.cat(box_centers, dim=0),
            torch.cat(box_half_extents, dim=0),
            lidar_range,
        ),
    )
  if sphere_centers:
    best_dists = torch.minimum(
        best_dists,
        ray_sphere_batch(
            ray_origins_w,
            ray_dirs_unit,
            torch.cat(sphere_centers, dim=0),
            torch.cat(sphere_radii, dim=0),
            lidar_range,
        ),
    )
  return best_dists


def dynamic_spawn_forbidden_fn(
    arena,
    env_pos,
    *,
    clearance,
    fallback_radius,
    sample_horizon_s,
    sample_dt_s,
):
  samples = arena.sample_dynamic_obstacle_geometries(
      horizon_s=float(sample_horizon_s),
      dt_s=float(sample_dt_s),
      collidable_only=True,
  )
  if not samples:
    return None

  try:
    env_pos_np = np.asarray(env_pos.reshape(3).detach().cpu(), np.float32)
  except Exception:
    env_pos_np = np.asarray(env_pos, np.float32).reshape(3)

  local_samples = []
  for shape, center_w, size in samples:
    center_l = np.asarray(center_w, np.float32).reshape(3) - env_pos_np
    if shape == 'box':
      local_samples.append(('box', center_l, 0.5 * np.asarray(size, np.float32).reshape(3)))
    else:
      local_samples.append(('sphere', center_l, float(size)))

  margin = float(clearance) + float(fallback_radius)

  def _forbidden(x, y, z):
    p = np.asarray([x, y, z], np.float32)
    for shape, center_l, size in local_samples:
      if shape == 'box':
        if point_box_distance_np(p, center_l, size) < margin:
          return True
        continue
      if float(np.linalg.norm(p - center_l)) < (margin + float(size)):
        return True
    return False

  return _forbidden


def dynamic_obstacle_collision(center_world, obstacle_geometries, *, margin, device):
  import torch

  center = torch.as_tensor(center_world, device=device).reshape(1, 3)
  for shape, obs_center, obs_size in obstacle_geometries:
    if shape == 'box':
      delta = torch.clamp((center - obs_center).abs() - obs_size, min=0.0)
      dist = torch.norm(delta, dim=-1)
      hit = dist <= float(margin)
    else:
      dist = torch.norm(obs_center - center, dim=-1)
      hit = dist <= (float(margin) + float(obs_size))
    if bool(hit.any().item()):
      return True
  return False


__all__ = [
    'dynamic_position_np',
    'point_box_distance_np',
    'dynamic_obstacle_box_half_extents',
    'dynamic_obstacles_world',
    'overlay_dynamic_obstacles_on_lidar',
    'dynamic_spawn_forbidden_fn',
    'dynamic_obstacle_collision',
]
