"""Debug-draw goal markers plus optional paper-quality trails/highlight."""

import threading


# Global registry so all PaperVisualizer instances share one debug draw
# and redraw all trails together (clear_lines is global).
_lock = threading.Lock()
_instances = {}  # env_id -> PaperVisualizer
_draw_interface = None


def _get_draw():
  global _draw_interface
  if _draw_interface is not None:
    return _draw_interface
  try:
    from omni.isaac.debug_draw import _debug_draw
    _draw_interface = _debug_draw.acquire_debug_draw_interface()
  except Exception:
    pass
  return _draw_interface


def _redraw_all():
  """Redraw trails for ALL registered instances in one pass."""
  draw = _get_draw()
  if draw is None:
    return
  draw.clear_lines()
  draw.clear_points()

  all_starts, all_ends, all_colors, all_widths = [], [], [], []
  marker_pts, marker_colors, marker_sizes = [], [], []

  for inst in _instances.values():
    pts = inst._points
    if len(pts) >= 2:
      n = len(pts) - 1
      for i in range(n):
        t = i / max(n - 1, 1)  # 0->1, green->red
        r = t
        g = 1.0 - t
        w = inst._width_min + (inst._width_max - inst._width_min) * t
        all_starts.append(pts[i])
        all_ends.append(pts[i + 1])
        all_colors.append((r, g, 0.0, 1.0))
        all_widths.append(w)
      # Start (green) and current (red) markers
      marker_pts.extend([pts[0], pts[-1]])
      marker_colors.extend([(0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)])
      marker_sizes.extend([10.0, 14.0])
    # Goal marker uses the same red as the USD sphere when drawn via debug draw.
    if inst._goal_pos is not None:
      marker_pts.append(inst._goal_pos)
      marker_colors.append((1.0, 0.1, 0.1, 1.0))
      marker_sizes.append(18.0)

  if all_starts:
    draw.draw_lines(all_starts, all_ends, all_colors, all_widths)
  if marker_pts:
    draw.draw_points(marker_pts, marker_colors, marker_sizes)


class PaperVisualizer:
  """Lightweight in-sim visualizer for paper figures.

  Goal markers are always available through debug draw. Trajectory trails
  and drone highlighting honor ``enabled``.
  """

  def __init__(self, cfg, drone_prim_path, env_id=0):
    cfg = cfg or {}
    self._enabled = bool(cfg.get('enabled', False))
    self._drone_prim_path = str(drone_prim_path)
    self._env_id = int(env_id)
    self._stride = max(1, int(cfg.get('trail_stride', 4)))
    self._points = []
    self._goal_pos = None
    self._step_counter = 0

    width_range = cfg.get('trail_width_range', [1.5, 5.0])
    self._width_min = float(width_range[0])
    self._width_max = float(width_range[1])

    self._highlight_enabled = bool(cfg.get('highlight_enabled', True))
    hl_color = cfg.get('highlight_color', [0.0, 0.6, 1.0])
    self._hl_color = (float(hl_color[0]), float(hl_color[1]), float(hl_color[2]))
    self._hl_intensity = float(cfg.get('highlight_intensity', 5.0))

    with _lock:
      _instances[self._env_id] = self

  # ------------------------------------------------------------------
  # Trajectory trail
  # ------------------------------------------------------------------

  def on_step(self, pos_world):
    """Record position and redraw all trails."""
    if not self._enabled:
      return
    self._step_counter += 1
    if self._step_counter % self._stride != 0:
      return

    try:
      p = pos_world.detach().cpu().reshape(-1).tolist()[:3]
    except Exception:
      import numpy as np
      p = list(float(x) for x in np.asarray(pos_world).reshape(-1)[:3])
    self._points.append(tuple(p))

    with _lock:
      _redraw_all()

  def set_goal(self, pos_world):
    """Set goal position for debug draw."""
    try:
      import numpy as np
      p = np.asarray(pos_world, np.float32).reshape(-1)[:3]
      self._goal_pos = tuple(float(x) for x in p)
    except Exception:
      self._goal_pos = None
    with _lock:
      _redraw_all()

  def on_reset(self):
    """Clear trail for new episode."""
    if not self._enabled:
      return
    self._points.clear()
    self._step_counter = 0
    with _lock:
      _redraw_all()

  # ------------------------------------------------------------------
  # Drone highlight (emissive material)
  # ------------------------------------------------------------------

  def apply_highlight(self):
    """Bind an emissive material to the drone for visual emphasis."""
    if not self._enabled or not self._highlight_enabled:
      return
    try:
      from pxr import UsdShade, Sdf, Gf
      import omni.isaac.core.utils.stage as stage_utils

      stage = stage_utils.get_current_stage()
      mat_path = f'/World/Materials/drone_highlight_{self._env_id}'
      mat = UsdShade.Material.Define(stage, mat_path)
      shader = UsdShade.Shader.Define(stage, f'{mat_path}/Shader')
      shader.CreateIdAttr('UsdPreviewSurface')
      # HDR emissive: multiply color by intensity for strong glow
      intensity = self._hl_intensity
      shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(
          Gf.Vec3f(
              self._hl_color[0] * intensity,
              self._hl_color[1] * intensity,
              self._hl_color[2] * intensity,
          )
      )
      shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
          Gf.Vec3f(0.15, 0.15, 0.15)
      )
      mat.CreateSurfaceOutput().ConnectToSource(
          UsdShade.ConnectableAPI(shader), 'surface'
      )

      from pxr import UsdGeom

      drone_prim = stage.GetPrimAtPath(self._drone_prim_path)
      if drone_prim and drone_prim.IsValid():
        # Bind to every Mesh descendant so child-level bindings don't override
        for desc in (p for p in drone_prim.GetDescendants() if p.IsA(UsdGeom.Mesh)):
          UsdShade.MaterialBindingAPI.Apply(desc).Bind(mat)
        # Also bind at top level as fallback
        UsdShade.MaterialBindingAPI.Apply(drone_prim).Bind(mat)
    except Exception as e:
      print(f'[PaperVisualizer] highlight failed: {e}')

  # ------------------------------------------------------------------
  # Cleanup
  # ------------------------------------------------------------------

  def close(self):
    with _lock:
      _instances.pop(self._env_id, None)
      _redraw_all()


__all__ = ['PaperVisualizer']
