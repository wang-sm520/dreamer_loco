"""Shared arena lifecycle and dynamic object registry for Omni environments."""

import atexit
import gc
import math
import os
import weakref
from dataclasses import dataclass

import numpy as np

from .scene_builder import SceneBuilder, SceneState
from .dynamic_geometry import dynamic_position_np
from .video import _VideoRecorder


@dataclass
class SlotState:
  env_id: int
  drone: object
  drone_prim_path: str
  active: bool
  env_pos: object
  publish_drone_dynamic: bool = False


@dataclass
class DynamicObject:
  object_id: str
  kind: str
  owner_env_id: int | None
  prim_path: str
  shape: str
  size: object
  pose_w: object
  lidar_visible: bool = True
  depth_visible: bool = True
  peer_collidable: bool = False


class OmniArena:
  @staticmethod
  def _resolve_gpu_ids(device_str):
    logical_idx = 0
    device_str = str(device_str)
    if device_str.startswith('cuda:'):
      try:
        logical_idx = int(device_str.split(':')[1])
      except (ValueError, IndexError):
        logical_idx = 0
    physical_idx = logical_idx
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible is not None:
      parts = [s.strip() for s in visible.split(',') if s.strip().isdigit()]
      if logical_idx < len(parts):
        physical_idx = int(parts[logical_idx])
    return int(logical_idx), int(physical_idx)

  def __init__(
      self,
      headless,
      physics_dt,
      device,
      num_drones,
      replicate_physics,
      drone_model,
      video_cfg=None,
      logdir=None,
      terrain_cfg=None,
      perimeter_cfg=None,
      obstacles_cfg=None,
      shutdown_callback=None,
      interaction_mode='multi',
      goal_visible_to_sensors=True,
      **_kw,
  ):
    self.headless = bool(headless)
    self.physics_dt = float(physics_dt)
    self.device = str(device)
    self._num_drones = int(num_drones)
    self._replicate_physics = bool(replicate_physics)
    interaction_mode = str(interaction_mode or 'multi')
    if interaction_mode not in ('multi', 'nav'):
      raise ValueError(f"interaction_mode must be 'multi' or 'nav', got '{interaction_mode}'")
    self._interaction_mode = interaction_mode
    self._goal_visible_to_sensors = bool(goal_visible_to_sensors)
    self._scene_drone_model = str(drone_model)

    self.simulation_app = None
    self.sim = None
    self._timeline = None

    self.scene = SceneState()
    self._terrain_cfg = dict(terrain_cfg or {})
    self._perimeter_cfg = dict(perimeter_cfg or {})
    self._obstacles_cfg = dict(obstacles_cfg or {})

    self._env_positions = None
    self._drone_pool = []
    self._drone_prim_paths = []
    self._free_env_ids = set(range(self._num_drones))
    self._active_envs = 0
    self._clients = weakref.WeakSet()
    self._sim_steps_total = 0
    self._train_steps_total = 0
    self._slot_states = {}
    self._dynamic_objects = {}
    self._dynamic_obstacle_specs = []
    self._dynamic_obstacle_view = None
    self._dynamic_obstacle_view_indices = {}
    self._dyn_centers = None  # vectorized motion cache; built after scene init
    self._sensor_render_env_ids = set()
    self._needs_sensor_render = False
    self._shutdown_callback = shutdown_callback

    self._init_sim()
    builder = SceneBuilder(self.sim)
    scene = builder.build(
        num_drones=self._num_drones,
        drone_model=self._scene_drone_model,
        device=self.device,
        terrain_cfg=self._terrain_cfg,
        perimeter_cfg=self._perimeter_cfg,
        obstacles_cfg=self._obstacles_cfg,
        create_goal_markers=bool(goal_visible_to_sensors),
    )
    self._apply_scene_state(scene)

    self._video_recorder = _VideoRecorder(video_cfg, logdir, self) if video_cfg else None
    self.sim.reset()
    self._ensure_timeline_playing()
    self.sim.step(False)
    self._initialize_dynamic_obstacle_view()
    self._sync_dynamic_drone_objects()
    self._register_dynamic_obstacles()
    self._build_dynamic_obstacle_motion_cache()
    self._sync_dynamic_obstacles(float(self._sim_steps_total) * float(self.physics_dt))

    atexit.register(self._cleanup_before_app_shutdown)

  def _init_sim(self):
    import builtins

    builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True
    from omni_drones import init_simulation_app

    logical_gpu, physical_gpu = self._resolve_gpu_ids(self.device)
    self.simulation_app = init_simulation_app({
        'headless': self.headless,
        'active_gpu': physical_gpu,
        'physics_gpu': logical_gpu,
    })
    try:
      import carb

      settings = carb.settings.get_settings()
      settings.set_int('/renderer/activeGpu', int(physical_gpu))
      settings.set_int('/physics/cudaDevice', int(logical_gpu))
    except Exception:
      pass

    import omni.usd
    from omni.isaac.core.utils import stage as stage_utils

    if stage_utils.get_current_stage() is None:
      omni.usd.get_context().new_stage()

    from omni.isaac.core.simulation_context import SimulationContext

    self.sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=float(self.physics_dt),
        rendering_dt=float(self.physics_dt),
        backend='torch',
        sim_params={},
        physics_prim_path='/physicsScene',
        device=str(self.device),
    )
    self._ensure_timeline_playing()

  def _apply_scene_state(self, scene):
    self.scene = scene
    self._drone_pool = list(scene.drone_pool)
    self._drone_prim_paths = list(scene.drone_prim_paths)
    self._env_positions = scene.env_positions
    self._obstacle_bounds = list(getattr(scene, 'obstacle_bounds', []))
    self._trap_bounds = list(getattr(scene, 'trap_bounds', []))
    self._dynamic_obstacle_specs = list(getattr(scene, 'dynamic_obstacles', []) or [])
    self._terrain_cfg = dict(getattr(scene, 'terrain_cfg', {}) or {})
    self._perimeter_cfg = dict(getattr(scene, 'perimeter_cfg', {}) or {})
    self._obstacles_cfg = dict(getattr(scene, 'obstacles_cfg', {}) or {})
    self._slot_states = {}
    for idx, drone in enumerate(self._drone_pool):
      self._slot_states[int(idx)] = SlotState(
          env_id=int(idx),
          drone=drone,
          drone_prim_path=str(self._drone_prim_paths[int(idx)]),
          active=False,
          env_pos=self._env_positions[int(idx)],
      )

  def step(self, substeps, headless, pre_substep_fns=None):
    self._ensure_timeline_playing()
    substeps = int(substeps)
    train_step_after = int(self._train_steps_total + int(self._active_envs))
    recorder = self._video_recorder
    rec_enabled = bool(recorder and recorder._enabled)
    needs_render = bool(self._needs_sensor_render)
    for substep in range(substeps):
      if substep > 0 and pre_substep_fns:
        for fn in pre_substep_fns:
          fn()
      t_next = float(self._sim_steps_total + 1) * float(self.physics_dt)
      self._sync_dynamic_obstacles(t_next)
      last = (substep == substeps - 1)
      render = last and ((not headless) or needs_render)
      capture = False
      if rec_enabled:
        capture = recorder.will_capture_frame(train_step_after, self._sim_steps_total)
        render = render or capture
      self.sim.step(render)
      self._sim_steps_total += 1
      self._sync_dynamic_drone_objects()
      if capture:
        recorder.on_physics_step(train_step_after, self._sim_steps_total, True)
    self._train_steps_total = train_step_after

  def step_reset_physics(self, render=False):
    """Advance physics during reset without counting an agent/train step."""
    self._ensure_timeline_playing()
    t_next = float(self._sim_steps_total + 1) * float(self.physics_dt)
    self._sync_dynamic_obstacles(t_next)
    self.sim.step(bool(render))
    self._sim_steps_total += 1
    self._sync_dynamic_drone_objects()

  def render_current(self):
    self._ensure_timeline_playing()
    self._sync_dynamic_obstacles(float(self._sim_steps_total) * float(self.physics_dt))
    self.sim.render()

  def _ensure_timeline_playing(self):
    if self._timeline is None:
      try:
        import omni.timeline

        self._timeline = omni.timeline.get_timeline_interface()
      except Exception:
        return
    if self._timeline and (self._timeline.is_stopped() or not self._timeline.is_playing()):
      self._timeline.play()

  def _cleanup_before_app_shutdown(self):
    for client in list(self._clients):
      try:
        client._drone = None
      except Exception:
        pass
    gc.collect()
    self.shutdown(shutdown_app=False)

  def acquire(self, drone_model, *, publish_drone_dynamic=True):
    if str(drone_model) != str(self._scene_drone_model):
      raise RuntimeError(
          f'Scene already initialized with drone_model={self._scene_drone_model}, '
          f'got drone_model={drone_model}.'
      )
    if not self._free_env_ids:
      raise RuntimeError('No free drone slots left. Increase num_drones in config.')
    env_id = min(self._free_env_ids)
    self._free_env_ids.remove(env_id)
    self._active_envs += 1

    slot = self._slot_states[int(env_id)]
    slot.active = True
    slot.publish_drone_dynamic = bool(publish_drone_dynamic)
    if slot.publish_drone_dynamic:
      self._register_or_update_drone_dynamic_object(int(env_id), slot)
    return (
        int(env_id),
        self._env_positions[int(env_id)],
        self._drone_prim_paths[int(env_id)],
        self._drone_pool[int(env_id)],
    )

  def release(self, env_id):
    if env_id is None:
      return
    env_id = int(env_id)
    if self._free_env_ids is not None:
      self._free_env_ids.add(env_id)
    if env_id in self._slot_states:
      self._slot_states[env_id].active = False
      self._slot_states[env_id].publish_drone_dynamic = False
    self.set_sensor_render_required(env_id, False)
    self.unregister_dynamic_object(self._drone_object_id(env_id))
    if self._active_envs:
      self._active_envs -= 1

  def iter_active_slot_states(self, *, exclude_env_id=None):
    for env_id, slot in self._slot_states.items():
      if not slot.active:
        continue
      if exclude_env_id is not None and int(env_id) == int(exclude_env_id):
        continue
      yield slot

  def set_goal_marker(self, env_id, pos_world):
    import numpy as np
    import omni.isaac.core.utils.prims as prim_utils
    from pxr import Gf, UsdGeom

    goal_visuals = getattr(self.scene, 'goal_visuals', [])
    idx = int(env_id)
    if idx < 0 or idx >= len(goal_visuals):
      return
    prim_path = goal_visuals[idx]
    prim = prim_utils.get_prim_at_path(prim_path)
    if not prim or not prim.IsValid():
      return
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    translate_op = None
    for op in ops:
      if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        translate_op = op
        break
    if translate_op is None:
      translate_op = xform.AddTranslateOp()
    p = np.asarray(pos_world, np.float32).reshape(3)
    translate_op.Set(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))

  def register_dynamic_object(self, obj):
    if isinstance(obj, DynamicObject):
      dyn = obj
    else:
      payload = dict(obj or {})
      object_id = str(payload.get('object_id', f'obj_{len(self._dynamic_objects)}'))
      dyn = DynamicObject(
          object_id=object_id,
          kind=str(payload.get('kind', 'unknown')),
          owner_env_id=payload.get('owner_env_id', None),
          prim_path=str(payload.get('prim_path', '')),
          shape=str(payload.get('shape', 'box')),
          size=payload.get('size', (1.0, 1.0, 1.0)),
          pose_w=payload.get('pose_w', None),
          lidar_visible=bool(payload.get('lidar_visible', True)),
          depth_visible=bool(payload.get('depth_visible', True)),
          peer_collidable=bool(payload.get('peer_collidable', False)),
      )
    self._dynamic_objects[str(dyn.object_id)] = dyn
    return str(dyn.object_id)

  def unregister_dynamic_object(self, object_id):
    self._dynamic_objects.pop(str(object_id), None)

  def update_dynamic_object_pose(self, object_id, pose_w):
    dyn = self._dynamic_objects.get(str(object_id), None)
    if dyn is None:
      return
    dyn.pose_w = pose_w

  def iter_dynamic_objects(
      self,
      *,
      kind=None,
      lidar_visible=None,
      depth_visible=None,
      peer_collidable=None,
      exclude_owner_env_id=None,
  ):
    for dyn in self._dynamic_objects.values():
      if kind is not None and str(dyn.kind) != str(kind):
        continue
      if lidar_visible is not None and bool(dyn.lidar_visible) != bool(lidar_visible):
        continue
      if depth_visible is not None and bool(dyn.depth_visible) != bool(depth_visible):
        continue
      if peer_collidable is not None and bool(dyn.peer_collidable) != bool(peer_collidable):
        continue
      if exclude_owner_env_id is not None and dyn.owner_env_id is not None:
        if int(dyn.owner_env_id) == int(exclude_owner_env_id):
          continue
      yield dyn

  @staticmethod
  def _spec_get(spec, key, default=None):
    if hasattr(spec, key):
      return getattr(spec, key)
    if isinstance(spec, dict):
      return spec.get(key, default)
    return default

  @staticmethod
  def _dynamic_obstacle_position(spec, t):
    return dynamic_position_np(
        center=OmniArena._spec_get(spec, 'center', (0.0, 0.0, 1.0)),
        motion_type=OmniArena._spec_get(spec, 'motion_type', 'line'),
        motion_axis=OmniArena._spec_get(spec, 'motion_axis', (1.0, 0.0, 0.0)),
        amplitude=OmniArena._spec_get(spec, 'amplitude', 0.0),
        omega=OmniArena._spec_get(spec, 'omega', 0.0),
        phase=OmniArena._spec_get(spec, 'phase', 0.0),
        motion_radius=OmniArena._spec_get(spec, 'motion_radius', 0.0),
        t=t,
    )

  @staticmethod
  def _set_prim_translation(prim_path, pos):
    import omni.isaac.core.utils.prims as prim_utils
    from pxr import Gf, UsdGeom

    prim = prim_utils.get_prim_at_path(str(prim_path))
    if prim is None or (not prim.IsValid()):
      return
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    translate_op = None
    for op in ops:
      if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        translate_op = op
        break
    if translate_op is None:
      translate_op = xform.AddTranslateOp()
    p = np.asarray(pos, np.float32).reshape(3)
    translate_op.Set(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))

  def _dynamic_obstacle_object_id(self, spec):
    object_id = str(self._spec_get(spec, 'object_id', 'dyn'))
    return f'obstacle:{object_id}'

  @staticmethod
  def _dynamic_obstacle_geometry(spec):
    shape = str(OmniArena._spec_get(spec, 'shape', 'box') or 'box').lower()
    if shape == 'box':
      size_arr = np.asarray(OmniArena._spec_get(spec, 'size', (0.7, 0.7, 0.7)), np.float32).reshape(-1)
      if size_arr.size == 0:
        size_arr = np.asarray([0.7, 0.7, 0.7], np.float32)
      while size_arr.size < 3:
        size_arr = np.concatenate([size_arr, size_arr[-1:]], axis=0)
      return 'box', tuple(float(max(1e-3, v)) for v in size_arr[:3])

    radius = OmniArena._spec_get(spec, 'collision_radius', None)
    if radius is None:
      radius = OmniArena._spec_get(spec, 'radius', None)
    if radius is None:
      size_arr = np.asarray(OmniArena._spec_get(spec, 'size', (0.7, 0.7, 0.7)), np.float32).reshape(-1)
      radius = 0.5 * float(np.max(size_arr[:3])) if size_arr.size else 0.35
    return 'sphere', float(max(1e-3, radius))

  def _initialize_dynamic_obstacle_view(self):
    if not self._dynamic_obstacle_specs:
      return
    try:
      from omni_drones.views import RigidPrimView
    except Exception:
      return
    try:
      view = RigidPrimView(
          "/World/DynamicObstacles/*",
          name="dynamic_obstacles_view",
          reset_xform_properties=False,
      )
      view.initialize()
    except Exception:
      return
    prim_paths = [str(path) for path in getattr(view, '_prim_paths', [])]
    if not prim_paths:
      return
    self._dynamic_obstacle_view = view
    self._dynamic_obstacle_view_indices = {
        str(path): int(idx) for idx, path in enumerate(prim_paths)
    }

  def _register_dynamic_obstacles(self):
    t_curr = float(self._sim_steps_total) * float(self.physics_dt)
    for spec in self._dynamic_obstacle_specs:
      object_id = self._dynamic_obstacle_object_id(spec)
      pose = self._dynamic_obstacle_position(spec, t_curr).reshape(1, 3)
      shape, size = self._dynamic_obstacle_geometry(spec)
      self.register_dynamic_object(DynamicObject(
          object_id=object_id,
          kind=str(self._spec_get(spec, 'kind', 'dynamic_obstacle')),
          owner_env_id=None,
          prim_path=str(self._spec_get(spec, 'prim_path', '')),
          shape=shape,
          size=size,
          pose_w=pose,
          lidar_visible=bool(self._spec_get(spec, 'lidar_visible', True)),
          depth_visible=bool(self._spec_get(spec, 'depth_visible', True)),
          peer_collidable=bool(self._spec_get(spec, 'drone_collidable', True)),
      ))

  def _build_dynamic_obstacle_motion_cache(self):
    """Pre-cache motion parameters as numpy arrays to enable vectorized position updates."""
    specs = self._dynamic_obstacle_specs
    if not specs:
      self._dyn_centers = None
      return
    n = len(specs)
    centers = np.zeros((n, 3), np.float32)
    omegas = np.zeros(n, np.float32)
    phases = np.zeros(n, np.float32)
    amplitudes = np.zeros(n, np.float32)
    axes = np.zeros((n, 3), np.float32)
    radii = np.zeros(n, np.float32)
    is_circle = np.zeros(n, bool)
    prim_paths = []
    view_indices = np.full(n, -1, np.int64)
    object_ids = []

    view = self._dynamic_obstacle_view
    for i, spec in enumerate(specs):
      centers[i] = np.asarray(self._spec_get(spec, 'center', (0.0, 0.0, 1.0)), np.float32).reshape(3)
      omegas[i] = float(self._spec_get(spec, 'omega', 0.0))
      phases[i] = float(self._spec_get(spec, 'phase', 0.0))
      amplitudes[i] = float(self._spec_get(spec, 'amplitude', 0.0))
      motion_type = str(self._spec_get(spec, 'motion_type', 'line') or 'line').lower()
      is_circle[i] = motion_type == 'circle'
      if is_circle[i]:
        radii[i] = float(self._spec_get(spec, 'motion_radius', 0.0))
      else:
        ax = np.asarray(self._spec_get(spec, 'motion_axis', (1.0, 0.0, 0.0)), np.float32).reshape(3)
        norm = float(np.linalg.norm(ax))
        axes[i] = ax / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0], np.float32)
      prim_path = str(self._spec_get(spec, 'prim_path', ''))
      prim_paths.append(prim_path)
      if view is not None:
        idx = self._dynamic_obstacle_view_indices.get(prim_path, None)
        if idx is not None:
          view_indices[i] = int(idx)
      object_ids.append(self._dynamic_obstacle_object_id(spec))

    self._dyn_centers = centers
    self._dyn_omegas = omegas
    self._dyn_phases = phases
    self._dyn_amplitudes = amplitudes
    self._dyn_axes = axes
    self._dyn_radii = radii
    self._dyn_is_circle = is_circle
    self._dyn_prim_paths = prim_paths
    self._dyn_view_indices = view_indices
    self._dyn_object_ids = object_ids
    self._dyn_positions = np.empty((n, 3), np.float32)  # reusable output buffer

  def _sync_dynamic_obstacles(self, t):
    if self._dyn_centers is None:
      return
    t = float(t)
    theta = self._dyn_omegas * t + self._dyn_phases  # [N]

    # Vectorized position computation into persistent buffer.
    np.copyto(self._dyn_positions, self._dyn_centers)
    line = ~self._dyn_is_circle
    if line.any():
      self._dyn_positions[line] += (
          (self._dyn_amplitudes[line] * np.sin(theta[line]))[:, None] * self._dyn_axes[line]
      )
    circ = self._dyn_is_circle
    if circ.any():
      self._dyn_positions[circ, 0] += self._dyn_radii[circ] * np.cos(theta[circ])
      self._dyn_positions[circ, 1] += self._dyn_radii[circ] * np.sin(theta[circ])

    # Update pose registry (direct dict lookup with pre-built id list, no spec iteration).
    for i, obj_id in enumerate(self._dyn_object_ids):
      dyn = self._dynamic_objects.get(obj_id)
      if dyn is not None:
        dyn.pose_w = self._dyn_positions[i:i+1]

    # Push positions to physics view or fall back to per-prim USD writes.
    view = self._dynamic_obstacle_view
    if view is not None:
      import torch
      has_view = self._dyn_view_indices >= 0
      if has_view.any():
        device = getattr(view, '_device', self.device)
        positions_t = torch.from_numpy(self._dyn_positions[has_view]).to(device)
        indices_t = torch.from_numpy(self._dyn_view_indices[has_view]).to(device)
        view.set_world_poses(positions=positions_t, env_indices=indices_t)
      for i in np.where(~has_view)[0]:
        self._set_prim_translation(self._dyn_prim_paths[i], self._dyn_positions[i])
    else:
      for i, prim_path in enumerate(self._dyn_prim_paths):
        self._set_prim_translation(prim_path, self._dyn_positions[i])

  def sample_dynamic_obstacle_geometries(self, *, horizon_s=0.0, dt_s=0.25, collidable_only=True):
    horizon = max(0.0, float(horizon_s))
    step = max(1e-3, float(dt_s))
    count = max(1, int(math.floor(horizon / step)) + 1)
    base_t = float(self._sim_steps_total) * float(self.physics_dt)
    samples = []
    for spec in self._dynamic_obstacle_specs:
      if collidable_only and (not bool(self._spec_get(spec, 'drone_collidable', True))):
        continue
      shape, size = self._dynamic_obstacle_geometry(spec)
      for idx in range(count):
        t = base_t + idx * step
        samples.append((shape, self._dynamic_obstacle_position(spec, t).copy(), size))
    return samples

  def update_slot_pose(self, env_id, pose_w):
    self.update_dynamic_object_pose(self._drone_object_id(int(env_id)), pose_w)

  def set_sensor_render_required(self, env_id, required):
    env_id = int(env_id)
    if bool(required):
      self._sensor_render_env_ids.add(env_id)
    else:
      self._sensor_render_env_ids.discard(env_id)
    self._needs_sensor_render = bool(self._sensor_render_env_ids)

  def _drone_object_id(self, env_id):
    return f'drone:{int(env_id)}'

  def _register_or_update_drone_dynamic_object(self, env_id, slot):
    pose_w = None
    try:
      pos_w, _ = slot.drone.get_world_poses()
      pose_w = pos_w.reshape(-1, 3)[:1]
    except Exception:
      try:
        pose_w = slot.env_pos.reshape(1, 3)
      except Exception:
        pose_w = None
    object_id = self._drone_object_id(int(env_id))
    if object_id in self._dynamic_objects:
      self._dynamic_objects[object_id].pose_w = pose_w
      self._dynamic_objects[object_id].owner_env_id = int(env_id)
      self._dynamic_objects[object_id].peer_collidable = True
      self._dynamic_objects[object_id].lidar_visible = True
      self._dynamic_objects[object_id].depth_visible = True
      return
    self.register_dynamic_object(DynamicObject(
        object_id=object_id,
        kind='drone',
        owner_env_id=int(env_id),
        prim_path=str(slot.drone_prim_path),
        shape='sphere',
        size=0.3,
        pose_w=pose_w,
        lidar_visible=True,
        depth_visible=True,
        peer_collidable=True,
    ))

  def _sync_dynamic_drone_objects(self):
    for env_id, slot in self._slot_states.items():
      if not slot.active or (not slot.publish_drone_dynamic):
        self.unregister_dynamic_object(self._drone_object_id(int(env_id)))
        continue
      self._register_or_update_drone_dynamic_object(int(env_id), slot)

  def shutdown(self, shutdown_app=True):
    try:
      import atexit
      atexit.unregister(self._cleanup_before_app_shutdown)
    except Exception:
      pass
    recorder = getattr(self, '_video_recorder', None)
    if recorder is not None:
      try:
        recorder.close()
      except Exception:
        pass
    if self.sim:
      for fn in ('stop', 'clear_all_callbacks', 'clear'):
        try:
          getattr(self.sim, fn)()
        except Exception:
          pass
      self.sim = None

    self._free_env_ids = None
    self._env_positions = None
    self._drone_pool = []
    self._drone_prim_paths = []
    self._slot_states = {}
    self._dynamic_objects = {}
    self._dynamic_obstacle_specs = []
    self._dynamic_obstacle_view = None
    self._dynamic_obstacle_view_indices = {}
    self._dyn_centers = None
    self._sensor_render_env_ids = set()
    self._needs_sensor_render = False

    try:
      from omni.isaac.core.simulation_context import SimulationContext

      if callable(getattr(SimulationContext, 'clear_instance', None)):
        SimulationContext.clear_instance()
      if hasattr(SimulationContext, '_instance'):
        SimulationContext._instance = None
    except Exception:
      pass

    try:
      from omni_drones.robots.robot import RobotBase

      RobotBase._robots.clear()
    except Exception:
      pass

    if shutdown_app and self.simulation_app:
      try:
        self.simulation_app.close()
      except Exception:
        pass
      self.simulation_app = None
    if shutdown_app:
      callback = getattr(self, '_shutdown_callback', None)
      if callback is not None:
        try:
          callback(self)
        except Exception:
          pass
