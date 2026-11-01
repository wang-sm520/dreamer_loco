"""Shared static sampler for Omni scenes."""

import math


class SceneSampler:
  def __init__(self, scene, bounds_cfg=None):
    self._scene = scene
    self._bounds_cfg = dict(bounds_cfg or {})

  def set_bounds_cfg(self, **kwargs):
    self._bounds_cfg.update(kwargs)

  def is_static_free(self, x, y, clearance):
    x = float(x)
    y = float(y)
    clearance = float(clearance)
    obstacle_collision_radius = float(
        self._bounds_cfg.get(
            'obstacle_collision_radius',
            self._bounds_cfg.get('collision_radius', 0.0),
        )
    )
    perimeter_collision_radius = float(
        self._bounds_cfg.get(
            'perimeter_collision_radius',
            self._bounds_cfg.get('collision_radius', 0.0),
        )
    )

    xy_limit = self._resolve_xy_limit(None)
    if xy_limit is not None and (abs(x) > float(xy_limit) or abs(y) > float(xy_limit)):
      return False

    for obs in getattr(self._scene, 'static_obstacles', []):
      cx, cy, _cz = obs.center
      sx, sy, _sz = obs.size
      buffer_xy = float(getattr(obs, 'buffer_xy', 0.0))
      yaw = float(getattr(obs, 'yaw', 0.0))
      margin_x = 0.5 * float(sx) + buffer_xy + clearance + obstacle_collision_radius
      margin_y = 0.5 * float(sy) + buffer_xy + clearance + obstacle_collision_radius
      dx = x - float(cx)
      dy = y - float(cy)
      if abs(yaw) > 1e-6:
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        lx = cos_y * dx + sin_y * dy
        ly = -sin_y * dx + cos_y * dy
      else:
        lx, ly = dx, dy
      if abs(lx) < margin_x and abs(ly) < margin_y:
        return False

    for item in getattr(self._scene, 'trap_bounds', []):
      tcx = float(item[0])
      tcy = float(item[1])
      tyaw = float(item[2])
      W = float(item[3])
      D = float(item[4])
      t = float(item[5])
      dx = x - tcx
      dy = y - tcy
      cos_y = math.cos(tyaw)
      sin_y = math.sin(tyaw)
      lx = cos_y * dx + sin_y * dy
      ly = -sin_y * dx + cos_y * dy
      if abs(lx) < W * 0.5 and t < ly < t + D:
        return False

    if not self._is_inside_perimeter(x, y, clearance, perimeter_collision_radius):
      return False
    return True

  def sample_start(
      self,
      *,
      height,
      clearance,
      max_tries=200,
      extra_forbidden_fn=None,
      xy_limit=None,
      device='cpu',
  ):
    import torch

    xy_limit = self._resolve_xy_limit(xy_limit)
    if xy_limit is None:
      return None

    for _ in range(int(max_tries)):
      tx = float((torch.rand(1, device=device) * 2.0 - 1.0) * float(xy_limit))
      ty = float((torch.rand(1, device=device) * 2.0 - 1.0) * float(xy_limit))
      tz = self._sample_height(height, device)
      if not self.is_static_free(tx, ty, clearance):
        continue
      if extra_forbidden_fn is not None and bool(extra_forbidden_fn(tx, ty, tz)):
        continue
      start = torch.zeros((1, 1, 3), device=device)
      start[..., 0] = tx
      start[..., 1] = ty
      start[..., 2] = tz
      return start
    return None

  def sample_target(
      self,
      *,
      start,
      height,
      min_dist,
      clearance,
      max_tries=200,
      extra_forbidden_fn=None,
      xy_limit=None,
      max_dist=None,
      distance_mode='xy',
      device='cpu',
  ):
    import torch

    xy_limit = self._resolve_xy_limit(xy_limit)
    if xy_limit is None:
      return None

    start = start.reshape(-1, 3)[0]
    sx = float(start[0].detach().cpu().item())
    sy = float(start[1].detach().cpu().item())
    sz = float(start[2].detach().cpu().item())
    min_dist = float(min_dist)
    max_dist = None if max_dist is None else float(max_dist)

    for _ in range(int(max_tries)):
      tx = float((torch.rand(1, device=device) * 2.0 - 1.0) * float(xy_limit))
      ty = float((torch.rand(1, device=device) * 2.0 - 1.0) * float(xy_limit))
      tz = self._sample_height(height, device)
      if not self.is_static_free(tx, ty, clearance):
        continue
      if str(distance_mode) == 'xyz':
        dist = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2 + (tz - sz) ** 2)
      elif str(distance_mode) == 'xy':
        dist = math.hypot(tx - sx, ty - sy)
      else:
        raise ValueError(f'Unknown distance_mode: {distance_mode}')
      if dist < min_dist:
        continue
      if max_dist is not None and dist > max_dist:
        continue
      if extra_forbidden_fn is not None and bool(extra_forbidden_fn(tx, ty, tz)):
        continue
      target = torch.zeros((1, 1, 3), device=device)
      target[..., 0] = tx
      target[..., 1] = ty
      target[..., 2] = tz
      return target
    return None

  def _sample_height(self, height, device):
    import torch

    if isinstance(height, (list, tuple)) and len(height) >= 2:
      hmin = float(height[0])
      hmax = float(height[1])
      return float(hmin + (hmax - hmin) * torch.rand(1, device=device))
    return float(height)

  def _resolve_xy_limit(self, xy_limit):
    if xy_limit is not None:
      return float(xy_limit)
    if 'xy_limit' in self._bounds_cfg and self._bounds_cfg['xy_limit'] is not None:
      return float(self._bounds_cfg['xy_limit'])
    terrain_cfg = getattr(self._scene, 'terrain_cfg', {}) or {}
    size = float(terrain_cfg.get('size', 30.0))
    return max(0.1, 0.5 * size - 1.0)

  def _is_inside_perimeter(self, x, y, clearance, collision_radius):
    perimeter_cfg = getattr(self._scene, 'perimeter_cfg', {}) or {}
    if not bool(perimeter_cfg.get('enabled', False)):
      return True
    terrain_cfg = getattr(self._scene, 'terrain_cfg', {}) or {}
    half = float(terrain_cfg.get('size', 30.0)) * 0.5
    thickness = float(perimeter_cfg.get('thickness', 0.3))
    wall_margin = thickness * 0.5 + float(clearance) + float(collision_radius)
    if abs(float(x)) > half - wall_margin:
      return False
    if abs(float(y)) > half - wall_margin:
      return False
    return True
