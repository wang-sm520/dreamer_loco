"""Shared ray-geometry helpers for LiDAR post-processing."""


def ray_sphere_batch(origins, directions, sphere_centers, radius, max_dist):
  import torch

  ray_count = origins.shape[0]
  sphere_count = sphere_centers.shape[0]
  if sphere_count == 0:
    return torch.full((ray_count,), max_dist, device=origins.device)

  radius = torch.as_tensor(radius, device=origins.device, dtype=origins.dtype).reshape(-1)
  if radius.numel() == 1:
    radius = radius.expand(sphere_count)
  elif radius.numel() != sphere_count:
    raise ValueError(f'radius shape mismatch: got {tuple(radius.shape)}, spheres={sphere_count}')

  d = directions.unsqueeze(1).expand(ray_count, sphere_count, 3)
  oc = origins.unsqueeze(1).expand(ray_count, sphere_count, 3) - sphere_centers.unsqueeze(0).expand(
      ray_count, sphere_count, 3
  )
  a = (d * d).sum(dim=-1)
  b = 2.0 * (oc * d).sum(dim=-1)
  c = (oc * oc).sum(dim=-1) - radius.unsqueeze(0) * radius.unsqueeze(0)
  disc = b * b - 4.0 * a * c
  hit = disc >= 0
  disc_safe = disc.clamp(min=0.0)
  sqrt_disc = disc_safe.sqrt()
  t0 = (-b - sqrt_disc) / (2.0 * a + 1e-12)
  t1 = (-b + sqrt_disc) / (2.0 * a + 1e-12)
  t = torch.where(t0 > 0, t0, t1)
  t[~hit | (t < 0)] = max_dist
  return t.min(dim=1).values


def ray_box_batch(origins, directions, box_centers, half_extents, max_dist):
  import torch

  ray_count = origins.shape[0]
  box_count = box_centers.shape[0]
  if box_count == 0:
    return torch.full((ray_count,), max_dist, device=origins.device)

  half = half_extents.reshape(-1, 3)
  if half.shape[0] == 1 and box_count > 1:
    half = half.expand(box_count, 3)
  elif half.shape[0] != box_count:
    raise ValueError(f'half_extents shape mismatch: got {tuple(half.shape)}, boxes={box_count}')

  d = directions.unsqueeze(1).expand(ray_count, box_count, 3)
  o = origins.unsqueeze(1).expand(ray_count, box_count, 3) - box_centers.unsqueeze(0).expand(
      ray_count, box_count, 3
  )
  half = half.unsqueeze(0).expand(ray_count, box_count, 3)
  parallel = d.abs() <= 1e-9
  safe_d = torch.where(parallel, torch.ones_like(d), d)
  t1 = (-half - o) / safe_d
  t2 = (half - o) / safe_d
  tmin_axis = torch.minimum(t1, t2)
  tmax_axis = torch.maximum(t1, t2)
  inside = o.abs() <= half
  pos_inf = torch.full_like(tmin_axis, float('inf'))
  neg_inf = torch.full_like(tmin_axis, -float('inf'))
  tmin_axis = torch.where(parallel & inside, neg_inf, tmin_axis)
  tmax_axis = torch.where(parallel & inside, pos_inf, tmax_axis)
  tmin_axis = torch.where(parallel & (~inside), pos_inf, tmin_axis)
  tmax_axis = torch.where(parallel & (~inside), neg_inf, tmax_axis)
  t_enter = tmin_axis.max(dim=-1).values
  t_exit = tmax_axis.min(dim=-1).values
  hit = t_exit >= torch.clamp(t_enter, min=0.0)
  t = torch.where(t_enter > 0.0, t_enter, t_exit)
  t[~hit | (t < 0.0)] = max_dist
  return t.min(dim=1).values


def normalize_rays(ray_origins_w, ray_hits_w, target_size):
  """Return (ray_origins_w, ray_dirs_unit) with consistent [N, 3] shapes."""
  if ray_hits_w.ndim != 2 or ray_hits_w.shape[-1] != 3:
    ray_hits_w = ray_hits_w.reshape(-1, 3)
  if ray_origins_w.ndim != 2 or ray_origins_w.shape[-1] != 3:
    ray_origins_w = ray_origins_w.reshape(-1, 3)
  if ray_origins_w.shape[0] != int(target_size):
    ray_origins_w = ray_origins_w[:1].expand(int(target_size), 3)
  ray_dirs_w = ray_hits_w - ray_origins_w
  dir_norms = ray_dirs_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  ray_dirs_unit = ray_dirs_w / dir_norms
  return ray_origins_w, ray_dirs_unit


__all__ = ['ray_sphere_batch', 'ray_box_batch', 'normalize_rays']
