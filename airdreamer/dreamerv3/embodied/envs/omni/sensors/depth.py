"""Depth camera wrapper aligned with the standalone nav environment."""

import numpy as np


class DepthCamera:
  def __init__(self, device='cuda:0'):
    self._device = str(device)
    self._camera = None
    self._depth_max = 6.0
    self._depth_resolution = (80, 48)
    self._depth_clipping_range = (0.6, 6.0)
    self._last_depth_closest = self._depth_max

  def create(self, drone_prim_path, cfg):
    import omni.isaac.core.utils.prims as prim_utils
    from omni_drones.sensors.camera import Camera
    from omni_drones.sensors.config import PinholeCameraCfg

    self._depth_resolution = tuple(int(x) for x in cfg.get('resolution', (80, 48)))
    self._depth_max = float(cfg.get('max_range', 6.0))
    self._depth_clipping_range = tuple(float(x) for x in cfg.get('clipping_range', (0.6, 6.0)))
    depth_translation = tuple(float(x) for x in cfg.get('translation', (0.0, 0.0, 0.0)))
    focal_length = float(cfg.get('focal_length', 2.09))
    horizontal_aperture = float(cfg.get('horizontal_aperture', 3.896))

    usd_params = PinholeCameraCfg.UsdCameraCfg(
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        clipping_range=self._depth_clipping_range,
        f_stop=0.0,
    )
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0.0,
        resolution=self._depth_resolution,
        data_types=['distance_to_image_plane'],
        usd_params=usd_params,
    )
    camera = Camera(cfg=camera_cfg)
    cam_path = f'{drone_prim_path}/base_link/depth_camera'
    if not prim_utils.is_prim_path_valid(cam_path):
      import math as _math
      pitch_deg = float(cfg.get('pitch_deg', 0.0))
      target = (1.0, 0.0, _math.tan(_math.radians(pitch_deg)))
      camera.spawn(prim_paths=[cam_path], translations=[depth_translation], targets=[target])
    camera.initialize(prim_paths_expr=cam_path)
    self._camera = camera
    print(f'[DepthCamera] Initialized at {cam_path} res={self._depth_resolution} max={self._depth_max}')

  def raw(self):
    import torch

    if self._camera is None:
      return None
    try:
      images = self._camera.get_images()
    except RuntimeError:
      return None
    depth = images.get('distance_to_image_plane', None)
    if depth is None:
      return None
    depth = torch.nan_to_num(depth, nan=self._depth_max, posinf=self._depth_max, neginf=0.0)
    return depth

  def observe(self):
    import torch.nn.functional as F

    h, w = self._depth_resolution[1], self._depth_resolution[0]
    raw = self.raw()
    if raw is None:
      self._last_depth_closest = self._depth_max
      return np.zeros((h, w, 1), np.float32)

    near = self._depth_clipping_range[0]
    valid = raw[raw > near]
    self._last_depth_closest = float(valid.min()) if valid.numel() > 0 else self._depth_max

    depth = raw.clamp(0, self._depth_max) / self._depth_max
    if depth.ndim == 3:
      depth = depth.unsqueeze(0)
    if depth.shape[-2] != h or depth.shape[-1] != w:
      depth = F.interpolate(depth, size=(h, w), mode='area')
    depth = depth[0].permute(1, 2, 0)
    if depth.shape[-1] != 1:
      depth = depth[..., :1]
    return np.asarray(depth.detach().cpu(), np.float32)

  def close(self):
    self._camera = None
