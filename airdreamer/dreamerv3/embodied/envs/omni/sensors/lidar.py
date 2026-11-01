"""LiDAR wrapper with optional mesh rebuild helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LidarFrame:
  distances: object
  ray_origins_w: object
  ray_hits_w: object


class LidarSensor:
  def __init__(self, device='cuda:0'):
    self._device = str(device)
    self._sensor = None
    self._range = 4.0
    self._resolution = (36, 4)
    self._cfg = None

  @property
  def resolution(self):
    return tuple(int(x) for x in self._resolution)

  @property
  def lidar_range(self):
    return float(self._range)

  def create(self, drone_prim_path, cfg, mesh_prim_paths):
    import torch

    mesh_prim_paths = list(mesh_prim_paths or [])

    try:
      from isaaclab.sensors import RayCasterCfg, patterns
    except ImportError as e:
      raise RuntimeError('LiDAR requires isaaclab') from e

    self._range = float(cfg.get('range', 4.0))
    vfov = tuple(cfg.get('vfov', (-10.0, 20.0)))
    vres = int(cfg.get('vres', 4))
    hres = int(cfg.get('hres', 36))
    hfov = float(cfg.get('hfov', 360.0))
    self._resolution = (hres, vres)

    pattern_cfg = patterns.BpearlPatternCfg(
        horizontal_fov=hfov,
        horizontal_res=hfov / float(max(1, hres)),
        vertical_ray_angles=torch.linspace(float(vfov[0]), float(vfov[1]), int(vres)),
    )

    class _RayCasterCfgNoValidate(RayCasterCfg):
      # IsaacLab's generic configclass.validate() walks every field's
      # __dict__ recursively, including `class_type` (a reference to the
      # RayCaster class itself). Once RayCaster.meshes[prim] has been
      # populated by an earlier drone's rebuild_mesh(), that class-level
      # cache holds a real warp.Mesh, and warp's internal
      # Mesh -> Device -> Runtime -> Device -> ... bookkeeping is a genuine
      # (intentional, harmless) reference cycle that isn't a dataclass
      # field cycle. configclass's validator doesn't expect this and blows
      # the recursion limit. We always pass every field explicitly here, so
      # the MISSING-field check validate() performs has nothing to check;
      # skip it rather than walking into shared warp internals.
      def validate(self):
        return []

    ray_caster_cfg = _RayCasterCfgNoValidate(
        prim_path=f'{drone_prim_path}/base_link',
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=False,
        pattern_cfg=pattern_cfg,
        debug_vis=False,
        mesh_prim_paths=mesh_prim_paths,
    )
    self._cfg = ray_caster_cfg
    self._sensor = ray_caster_cfg.class_type(ray_caster_cfg)
    if hasattr(self._sensor, '_initialize_impl'):
      self._sensor._initialize_impl()

  def capture(self, dt):
    if self._sensor is None:
      return LidarFrame(None, None, None)
    self._sensor.update(float(dt))
    data = self._sensor.data
    if data is None or data.ray_hits_w is None or data.pos_w is None:
      return LidarFrame(None, None, None)

    diffs = data.ray_hits_w - data.pos_w.unsqueeze(1)
    dists = diffs.norm(dim=-1).clamp_max(float(self._range))
    if dists.ndim >= 2:
      dists = dists[0]

    try:
      ray_hits_w = data.ray_hits_w[0].reshape(-1, 3)
      base = data.pos_w[0].reshape(1, 3)
      ray_origins_w = base.expand(ray_hits_w.shape[0], 3)
    except Exception:
      ray_hits_w = None
      ray_origins_w = None
    return LidarFrame(dists, ray_origins_w, ray_hits_w)

  def observe(self, dt):
    frame = self.capture(dt)
    h, v = self.resolution
    if frame.distances is None:
      return np.zeros((h, v), np.float32)
    if int(frame.distances.numel()) != h * v:
      raise RuntimeError(
          f'LiDAR ray count mismatch: got {frame.distances.numel()}, expected {h * v} '
          f'(hres={h}, vres={v}).'
      )
    scan = float(self._range) - frame.distances.reshape(h, v)
    return np.asarray(scan.detach().cpu(), np.float32)

  def rebuild_mesh(self, scene):
    if self._sensor is None or scene is None:
      return
    import trimesh
    from pxr import UsdGeom, Usd
    import omni.usd
    from isaaclab.sensors.ray_caster import RayCaster

    try:
      from isaaclab.utils.warp import convert_to_warp_mesh
    except ImportError:
      from omni.isaac.orbit.utils.warp import convert_to_warp_mesh

    stage = omni.usd.get_context().get_stage()
    all_points = []
    all_indices = []
    vertex_offset = 0

    ground_path = str(self._cfg.mesh_prim_paths[0]) if self._cfg and self._cfg.mesh_prim_paths else '/World/ground'
    ground_prim = stage.GetPrimAtPath(ground_path)
    if ground_prim and ground_prim.IsValid():
      for prim in Usd.PrimRange(ground_prim):
        if not prim.IsA(UsdGeom.Mesh):
          continue
        usd_mesh = UsdGeom.Mesh(prim)
        pts_raw = usd_mesh.GetPointsAttr().Get()
        idx_raw = usd_mesh.GetFaceVertexIndicesAttr().Get()
        if pts_raw is None or idx_raw is None:
          continue
        xform = UsdGeom.Xformable(prim)
        mat = np.array(
            xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        )
        pts = np.array(pts_raw, dtype=np.float64)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_w = (mat @ pts_h.T).T[:, :3].astype(np.float32)
        idx = np.array(idx_raw, dtype=np.int32)
        all_points.append(pts_w)
        all_indices.append(idx + vertex_offset)
        vertex_offset += len(pts_w)

    for item in getattr(scene, 'obstacle_bounds', []):
      cx, cy, cz, sx, sy, sz = (float(item[i]) for i in range(6))
      yaw = float(item[6]) if len(item) > 6 else 0.0
      tm = trimesh.creation.box(extents=(sx, sy, sz))
      pts = tm.vertices.astype(np.float32)
      if abs(yaw) > 1e-6:
        cos_y = np.float32(np.cos(yaw))
        sin_y = np.float32(np.sin(yaw))
        rot = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        pts = (rot @ pts.T).T
      pts += np.array([cx, cy, cz], np.float32)
      idx = tm.faces.flatten().astype(np.int32)
      all_points.append(pts)
      all_indices.append(idx + vertex_offset)
      vertex_offset += len(pts)

    pc = getattr(scene, 'perimeter_cfg', {}) or {}
    tc = getattr(scene, 'terrain_cfg', {}) or {}
    if pc.get('enabled', False):
      ts = float(tc.get('size', 24.0))
      mx, my = ts / 2.0, ts / 2.0
      thick = float(pc.get('thickness', 0.3))
      wh = float(pc.get('height', 4.0))
      for px, py, ssx, ssy in [
          (0.0, my, 2 * mx + thick, thick),
          (0.0, -my, 2 * mx + thick, thick),
          (mx, 0.0, thick, 2 * my + thick),
          (-mx, 0.0, thick, 2 * my + thick),
      ]:
        tm = trimesh.creation.box(extents=(ssx, ssy, wh))
        pts = tm.vertices.astype(np.float32) + np.array([px, py, wh / 2.0], np.float32)
        idx = tm.faces.flatten().astype(np.int32)
        all_points.append(pts)
        all_indices.append(idx + vertex_offset)
        vertex_offset += len(pts)

    if not all_points:
      return

    combined_pts = np.concatenate(all_points, axis=0)
    combined_idx = np.concatenate(all_indices, axis=0)
    mesh_key = self._cfg.mesh_prim_paths[0]
    RayCaster.meshes[mesh_key] = convert_to_warp_mesh(combined_pts, combined_idx, device=str(self._device))

  def close(self):
    self._sensor = None
    self._cfg = None
