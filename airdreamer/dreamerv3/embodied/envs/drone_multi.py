import math
from collections import deque

import elements
import embodied
import numpy as np

from embodied.envs.omni.arena_manager import (
    arena_lock,
    get_shared_arena,
    request_forced_reset,
    consume_forced_reset,
    clear_forced_reset,
    get_start_positions,
)
from embodied.envs.omni.collision_checks import (
    base_link_contact_collision as _base_link_contact_collision_helper,
    read_base_link_contact_forces as _read_base_link_contact_forces,
)
from embodied.envs.omni.control import make_action_adapter
from embodied.envs.omni.domain_rand import DomainRandomizer
from embodied.envs.omni.dynamic_geometry import (
    dynamic_obstacle_collision,
    dynamic_obstacles_world,
    overlay_dynamic_obstacles_on_lidar,
    dynamic_spawn_forbidden_fn,
)
from embodied.envs.omni.sampling import SceneSampler
from embodied.envs.omni.sensors.ray_math import normalize_rays, ray_sphere_batch
from embodied.envs.omni.sensors.setup_mixin import (
    setup_depth_camera as _setup_depth_camera,
    setup_lidar as _setup_lidar,
)
from embodied.envs.omni.state_encoding import (
    encode_goal_relative_state_15d as _encode_goal_relative_state_15d,
)
from embodied.envs.omni.visualize import PaperVisualizer

_LOCK = arena_lock()
_START_POSITIONS = get_start_positions()



class OmniDronesMulti(embodied.Env):

  def __init__(
      self,
      task='multi',
      headless=True,
      physics_dt=1 / 60,
      substeps=1,
      device='cuda:0',
      max_episode_length=3600,
      drone_model='hummingbird',
      controller=None,
      action_mode='direct_rpm',
      action_limit_vel=1.5,
      action_limit_yaw_rate=math.pi,
      reward_effort_weight=0.1,
      seed=0,
      logdir=None,
      **kwargs,
  ):
    del task
    self._headless = bool(headless)
    self._physics_dt = float(physics_dt)
    self._substeps = int(substeps)
    self._device = str(device)
    self._max_episode_length = int(max_episode_length)
    self._from_ground = bool(kwargs.pop('from_ground', False))
    self._ground_spawn_z = float(kwargs.pop('ground_spawn_z', 0.1))
    ground_grace_secs = float(kwargs.pop('ground_grace_secs', 2.0))
    denom = float(self._physics_dt) * float(self._substeps)
    self._ground_grace_steps = int(max(0, round(ground_grace_secs / denom))) if denom > 0 else 0
    self._drone_model = str(drone_model)
    self._controller = str(controller) if controller is not None else None
    self._action_mode = str(action_mode or 'direct_rpm').lower()
    self._action_limit_vel = float(action_limit_vel)
    self._action_limit_yaw_rate = float(action_limit_yaw_rate)
    self._use_yaw_rate = bool(kwargs.pop('use_yaw_rate', True))
    obs_mode = str(kwargs.pop('obs_mode', 'auto') or 'auto').lower()
    if obs_mode == 'auto':
      obs_mode = 'goal_frame' if self._action_mode == 'velocity_cmd' else 'full_state'
    if obs_mode not in ('full_state', 'goal_frame'):
      raise ValueError(f'Unknown obs_mode: {obs_mode}')
    self._obs_mode = obs_mode
    self._reward_effort_weight = float(reward_effort_weight)
    self._seed = int(seed)
    self._logdir = str(logdir) if logdir is not None else None
    interaction_mode = str(kwargs.pop('interaction_mode', 'multi') or 'multi').lower()
    if interaction_mode != 'multi':
      raise ValueError(f'OmniDronesMulti requires interaction_mode="multi", got {interaction_mode!r}.')

    self._step_count = 0
    self._done = True

    terrain_cfg = kwargs.pop('terrain', {}) or {}
    perimeter_cfg = kwargs.pop('perimeter', {}) or {}
    obstacles_cfg = kwargs.pop('obstacles', {}) or {}
    dynamic_cfg = dict(obstacles_cfg.get('dynamic', {}) or {})
    dyn_items = dynamic_cfg.get('items', []) or []
    if isinstance(dyn_items, dict):
      dyn_count = len(dyn_items)
    elif isinstance(dyn_items, (list, tuple)):
      dyn_count = len(dyn_items)
    else:
      dyn_count = 0
    self._dynamic_obstacles_enabled = bool(dynamic_cfg.get('enabled', False)) and (dyn_count > 0)
    self._dynamic_sample_horizon_s = float(dynamic_cfg.get('sample_horizon_s', 2.0))
    self._dynamic_sample_dt_s = float(dynamic_cfg.get('sample_dt_s', 0.25))
    self._dynamic_collision_margin = float(dynamic_cfg.get('collision_margin', 0.0))

    terrain_size = float(terrain_cfg.get('size', 30.0))
    self._bound_xy = float(kwargs.pop('local_bound_xy', max(10.0, terrain_size * 0.45)))
    self._bound_z = float(kwargs.pop('local_bound_z', max(6.0, terrain_size * 0.45)))
    self._goal_min_dist = float(kwargs.pop('goal_min_dist', 3.0))
    self._goal_max_dist = float(kwargs.pop('goal_max_dist', min(8.0, self._bound_xy * 0.8)))
    self._goal_radius = float(kwargs.pop('goal_radius', 0.5))

    self._min_start_dist = float(kwargs.pop('min_start_dist', 2))
    self._collision_penalty = float(kwargs.pop('collision_penalty', 10.0))
    self._alive_bonus = float(kwargs.pop('alive_bonus', 1.0))
    self._goal_bonus = float(kwargs.pop('goal_bonus', 5.0))
    # Progress sigmoid reward parameters (per-step approach reward)
    self._progress_sigmoid_k = float(kwargs.pop('progress_sigmoid_k', 20.0))
    self._progress_sigmoid_offset = float(kwargs.pop('progress_sigmoid_offset', 0.005))
    # Sliding-window average speed reward parameters (strategic pressure)
    self._avg_speed_window = int(kwargs.pop('avg_speed_window', 40))
    self._avg_speed_threshold = float(kwargs.pop('avg_speed_threshold', 0.35))
    self._avg_speed_sigmoid_k = float(kwargs.pop('avg_speed_sigmoid_k', 8.0))
    self._avg_speed_warmup = int(kwargs.pop('avg_speed_warmup', 10))
    self._progress_history = deque(maxlen=max(1, self._avg_speed_window))
    self._base_link_collision_force_threshold = float(kwargs.pop('base_link_collision_force_threshold', 1e-4))
    self._drone_hit_radius = 0.3
    self._spawn_clearance = max(
        float(self._min_start_dist),
        2.0 * float(self._drone_hit_radius) + 0.05,
    )
    self._tilt_soft_deg = float(kwargs.pop('tilt_soft_deg', 35.0))
    self._tilt_hard_deg = float(kwargs.pop('tilt_hard_deg', 55.0))
    self._tilt_soft_cos = math.cos(math.radians(self._tilt_soft_deg))
    self._tilt_hard_cos = math.cos(math.radians(self._tilt_hard_deg))
    self._tilt_penalty_weight = float(kwargs.pop('tilt_penalty_weight', 0.5))
    self._angvel_penalty_weight = float(kwargs.pop('angvel_penalty_weight', 0.1))
    self._term_reason_names = (
        'none', 'timeout', 'nan', 'collision', 'ground', 'out_of_bounds', 'terminal'
    )
    self._prev_dist = None
    self._cached_action = None
    self._pre_substep_fn = None
    self._omni_cached_reset_obs = None
    self._omni_passive_reset_step = False
    self._omni_reset_hold_pending = False
    hold_after_reset = kwargs.pop('hold_after_reset', None)
    self._hold_after_reset = (not self._from_ground) if hold_after_reset is None else bool(hold_after_reset)
    self._goal_visible_to_sensors = bool(kwargs.pop('goal_visible_to_sensors', False))
    self._obs_space = None

    cfg_num_drones = kwargs.pop('num_drones', None)
    cfg_max_envs = int(kwargs.pop('max_envs', 16))
    if cfg_num_drones is None:
      cfg_num_drones = cfg_max_envs
    cfg_num_drones = int(cfg_num_drones)

    cfg_replicate_physics = bool(kwargs.pop('replicate_physics', False))

    lidar_cfg = kwargs.pop('lidar', {}) or {}
    self._lidar_as_obs = bool(lidar_cfg.pop('enabled', True))
    self._lidar_range = float(lidar_cfg.pop('range', 4.0))
    self._lidar_vfov = tuple(lidar_cfg.pop('vfov', (-10.0, 20.0)))
    self._lidar_vres = int(lidar_cfg.pop('vres', 4))
    self._lidar_hres = int(lidar_cfg.pop('hres', 36))
    self._lidar_hfov = float(lidar_cfg.pop('hfov', 360.0))
    self._lidar_resolution = (int(self._lidar_hres), int(self._lidar_vres))
    self._lidar_mesh_prim = '/World/ground'

    depth_cam_cfg = kwargs.pop('dcam', {}) or {}
    self._depth_camera_enabled = bool(depth_cam_cfg.pop('enabled', False))
    self._depth_resolution = tuple(int(x) for x in depth_cam_cfg.pop('resolution', (80, 48)))
    self._depth_max = float(depth_cam_cfg.pop('max_range', 6.0))
    self._depth_focal_length = float(depth_cam_cfg.pop('focal_length', 2.09))
    self._depth_horizontal_aperture = float(depth_cam_cfg.pop('horizontal_aperture', 3.896))
    self._depth_clipping_range = tuple(float(x) for x in depth_cam_cfg.pop('clipping_range', (0.6, 6.0)))
    self._depth_translation = tuple(float(x) for x in depth_cam_cfg.pop('translation', (0.0, 0.0, 0.0)))
    self._depth_pitch_deg = float(depth_cam_cfg.pop('pitch_deg', 0.0))
    self._dr_cfg = kwargs.pop('domain_rand', {}) or {}

    with _LOCK:
      video = kwargs.pop('video', None)
      self._arena = get_shared_arena(
          headless=self._headless,
          physics_dt=self._physics_dt,
          device=self._device,
          video_cfg=video,
          logdir=logdir,
          num_drones=cfg_num_drones,
          env_max_envs=cfg_max_envs,
          replicate_physics=cfg_replicate_physics,
          drone_model=self._drone_model,
          terrain_cfg=terrain_cfg,
          perimeter_cfg=perimeter_cfg,
          obstacles_cfg=obstacles_cfg,
          interaction_mode=interaction_mode,
          goal_visible_to_sensors=self._goal_visible_to_sensors,
      )
      self._arena._clients.add(self)
      self._env_id, self._env_pos, self._drone_prim_path, self._drone = self._arena.acquire(
          self._drone_model,
          publish_drone_dynamic=True,
      )

    viz_cfg = kwargs.pop('visualize', {}) or {}
    self._paper_viz = PaperVisualizer(viz_cfg, self._drone_prim_path, self._env_id)
    self._setup_scene()

  @property
  def obs_space(self):
    return self._obs_space

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.float32, (self._act_dim,), -1.0, 1.0),
    }

  def step(self, action):
    needs_step = self._omni_pre_step(action)
    if needs_step:
      pre_fns = [self._pre_substep_fn] if self._pre_substep_fn else None
      self._arena.step(self._substeps, self._headless, pre_substep_fns=pre_fns)
    return self._omni_post_step(global_step=needs_step, own_needs_step=needs_step)

  def _omni_pre_step(self, action):
    reset = bool(action.get('reset', False))
    if reset or self._done:
      self._omni_passive_reset_step = False
      self._omni_cached_reset_obs = self.reset()
      self._omni_reset_hold_pending = bool(self._hold_after_reset)
      self._pre_substep_fn = None
      return False
    if self._omni_reset_hold_pending:
      self._apply_reset_hold_action()
      self._omni_reset_hold_pending = False
      self._pre_substep_fn = self._apply_reset_hold_action
      return True
    act = np.array(action['action'], dtype=np.float32, copy=True).reshape(self._act_dim)
    self._cached_action = act
    self._action_adapter.apply(act)
    self._pre_substep_fn = self._reapply_cached_action
    return True

  def _reapply_cached_action(self):
    self._action_adapter.apply(self._cached_action)

  def _omni_passive_step(self):
    if self._omni_cached_reset_obs is None:
      return
    if self._omni_reset_hold_pending:
      self._apply_reset_hold_action()
    self._omni_reset_hold_pending = False
    self._omni_passive_reset_step = True

  def _omni_post_step(self, global_step=False, own_needs_step=False):
    if self._omni_cached_reset_obs is not None:
      if global_step and (not own_needs_step) and self._omni_passive_reset_step:
        obs = self._current_reset_obs()
      else:
        obs = self._omni_cached_reset_obs
      self._omni_cached_reset_obs = None
      self._omni_passive_reset_step = False
      return obs
    with _LOCK:
      forced = consume_forced_reset(self._env_id)
    if forced:
      self._done = True
      return self._make_obs(
          reward=-float(self._collision_penalty),
          is_first=False,
          is_last=True,
          is_terminal=True,
          term_reason='collision',
      )

    self._step_count += 1
    return self._compute_obs()

  def _apply_reset_hold_action(self):
    self._action_adapter.hold()

  def _current_reset_obs(self):
    import torch

    drone_state = self._drone.get_state(env_frame=False)
    self._prev_dist = torch.norm((self._target_local - drone_state[..., :3]).reshape(-1, 3), dim=-1).reshape(1, 1).detach()
    return self._make_obs(
        reward=0.0,
        is_first=True,
        is_last=False,
        is_terminal=False,
        term_reason='none',
        drone_state=drone_state,
    )

  def _positions_far_enough(self, start_local, others, min_start_threshold, horizontal_only=False):
    import torch

    for other_start in others:
      delta = (start_local - other_start).reshape(-1, 3)
      if horizontal_only:
        start_dist = torch.norm(delta[..., :2], dim=-1)
      else:
        start_dist = torch.norm(delta, dim=-1)
      if bool((start_dist < min_start_threshold).any().item()):
        return False
    return True

  def _spawn_far_enough(self, start_local, other_starts, other_currents, min_start_threshold):
    if not self._positions_far_enough(start_local, other_starts, min_start_threshold):
      return False
    if not self._positions_far_enough(start_local, other_currents, min_start_threshold):
      return False
    if self._from_ground:
      if not self._positions_far_enough(
          start_local,
          other_currents,
          min_start_threshold,
          horizontal_only=True,
      ):
        return False
    return True

  def _spawn_clearance_score(self, start_local, other_starts, other_currents):
    import torch

    groups = [(other_starts, False), (other_currents, False)]
    if self._from_ground:
      groups.append((other_currents, True))
    best = float('inf')
    for others, horizontal_only in groups:
      for other_pos in others:
        delta = (start_local - other_pos).reshape(-1, 3)
        dist = torch.norm(delta[..., :2], dim=-1) if horizontal_only else torch.norm(delta, dim=-1)
        best = min(best, float(dist.min().detach().cpu().item()))
    return best

  def _get_other_start_positions(self):
    # Called from reset() which already holds _LOCK – do NOT re-acquire.
    snapshot = list(_START_POSITIONS.items())
    others = []
    for idx, start_local in snapshot:
      if int(idx) == int(self._env_id):
        continue
      others.append(start_local.to(self._torch_device))
    return others


  def _get_other_drone_positions_local(self):
    positions = []
    for slot in self._arena.iter_active_slot_states(exclude_env_id=self._env_id):
      try:
        pos_w, _ = slot.drone.get_world_poses()
        pos = pos_w.reshape(-1, 3)[:1].to(self._torch_device)
        pos_local = pos - self._env_pos.reshape(1, 3)
        positions.append(pos_local.reshape(1, 1, 3))
      except Exception:
        continue
    return positions

  def _reset_start_local(self, safe_xy, safe_z_min, safe_z_max):
    import torch

    min_start_threshold = float(self._spawn_clearance)
    other_starts = self._get_other_start_positions()
    other_currents = self._get_other_drone_positions_local()
    dynamic_forbidden = self._dynamic_spawn_forbidden_fn(min_start_threshold)

    def _peer_filter(x, y, z):
      start_local = torch.zeros((1, 1, 3), device=self._torch_device)
      start_local[..., 0] = float(x)
      start_local[..., 1] = float(y)
      start_local[..., 2] = float(z)
      peer_blocked = not self._spawn_far_enough(start_local, other_starts, other_currents, min_start_threshold)
      dyn_blocked = bool(dynamic_forbidden and dynamic_forbidden(x, y, z))
      return bool(peer_blocked or dyn_blocked)

    height = float(self._ground_spawn_z) if self._from_ground else (float(safe_z_min), float(safe_z_max))
    start_local = self._sampler.sample_start(
        height=height,
        clearance=float(self._spawn_clearance),
        max_tries=120,
        extra_forbidden_fn=_peer_filter,
        xy_limit=float(safe_xy),
        device=self._torch_device,
    )
    if start_local is not None:
      if not self._from_ground:
        start_local[..., 2].clamp_(safe_z_min, safe_z_max)
      return start_local

    z = float(self._ground_spawn_z) if self._from_ground else float(min(safe_z_max, safe_z_min + 0.5))
    spacing = max(min_start_threshold, 0.4)
    steps = max(3, int(math.ceil((2.0 * safe_xy) / spacing)) + 1) if safe_xy > 0.0 else 1
    if steps % 2 == 0:
      steps += 1
    coords = [0.0] if safe_xy <= 0.0 else np.linspace(-safe_xy, safe_xy, steps).tolist()
    candidates = [(0.0, 0.0)]
    candidates.extend(
        (float(x), float(y))
        for x in coords for y in coords
        if abs(float(x)) > 1e-6 or abs(float(y)) > 1e-6
    )
    candidates.sort(key=lambda xy: xy[0] * xy[0] + xy[1] * xy[1])
    best_start = None
    best_score = -float('inf')
    for x, y in candidates:
      if not self._sampler.is_static_free(x, y, float(self._spawn_clearance)):
        continue
      start_local = torch.zeros((1, 1, 3), device=self._torch_device)
      start_local[..., 0] = float(x)
      start_local[..., 1] = float(y)
      start_local[..., 2] = float(z)
      if dynamic_forbidden and dynamic_forbidden(x, y, z):
        continue
      if self._spawn_far_enough(start_local, other_starts, other_currents, min_start_threshold):
        return start_local
      score = self._spawn_clearance_score(start_local, other_starts, other_currents)
      if score > best_score:
        best_score = score
        best_start = start_local.clone()
    if best_start is not None:
      return best_start

    start_local = torch.zeros((1, 1, 3), device=self._torch_device)
    start_local[..., 2] = float(z)
    return start_local

  def reset(self):
    import torch
    from omni_drones.utils.torch import euler_to_quaternion

    self._paper_viz.on_reset()
    self._step_count = 0
    self._done = False
    self._omni_passive_reset_step = False
    self._omni_reset_hold_pending = False
    self._progress_history.clear()
    self._arena._ensure_timeline_playing()

    view_ids = torch.as_tensor([0], device=self._torch_device)
    self._drone._reset_idx(view_ids, True)
    self._domain_rand.on_reset()
    if self._from_ground and getattr(self._drone, 'throttle', None) is not None:
      self._drone.throttle[view_ids] = 0.0

    safe_xy, safe_z_min, safe_z_max = self._safe_bounds()
    self._sampler.set_bounds_cfg(xy_limit=float(safe_xy))
    max_dist = min(float(self._goal_max_dist), safe_xy * 2.0)
    min_dist = max(5.0, float(self._goal_min_dist))
    max_dist = max(float(max_dist), float(min_dist))

    with _LOCK:
      _START_POSITIONS.pop(int(self._env_id), None)
      start_local = self._reset_start_local(safe_xy, safe_z_min, safe_z_max)
      target = self._sample_target(start_local, min_dist=min_dist, max_dist=max_dist, fallback=True)
      if target is None:
        target = self._fallback_target(start_local)
      _START_POSITIONS[int(self._env_id)] = start_local.detach().clone()

    self._target_local = target
    self._target_dir = (target - start_local).reshape(1, 1, 3)
    self._target_dir[..., 2] = 0.0
    self._move_target_marker(target.reshape(3))

    pos_local = start_local
    rpy = torch.zeros((1, 1, 3), device=self._torch_device)
    rot = euler_to_quaternion(rpy)
    pos_world = pos_local + self._env_pos.reshape(1, 1, 3)
    self._drone.set_world_poses(pos_world, rot, view_ids)
    self._drone.set_velocities(torch.zeros_like(self._drone.get_velocities()), view_ids)
    if self._drone.is_articulation and self._drone.rotor_joint_indices is not None:
      self._drone._view.set_joint_velocities(
          torch.zeros(1, len(self._drone.rotor_joint_indices), device=self._torch_device),
          joint_indices=self._drone.rotor_joint_indices,
      )
    if self._depth_camera_enabled:
      # Refresh camera annotators without advancing physics; an extra reset-time
      # physics step can desynchronize the recorded frame from obstacle poses.
      self._arena.render_current()

    self._arena.update_slot_pose(self._env_id, pos_world.reshape(-1, 3)[:1])
    self._contact_baseline = _read_base_link_contact_forces(self._drone, self._torch_device)
    self._prev_dist = torch.norm((self._target_local - pos_local).reshape(-1, 3), dim=-1).reshape(1, 1)
    return self._make_obs(
        reward=0.0,
        is_first=True,
        is_last=False,
        is_terminal=False,
        term_reason='none',
    )

  def _sample_target(self, current_pos, min_dist=None, max_dist=None, fallback=True):
    safe_xy, safe_z_min, safe_z_max = self._safe_bounds()
    if max_dist is None:
      max_dist = min(float(self._goal_max_dist), safe_xy * 2.0)
    if min_dist is None:
      min_dist = max(5.0, float(self._goal_min_dist))

    dynamic_forbidden = self._dynamic_spawn_forbidden_fn(float(self._drone_hit_radius))
    target = self._sampler.sample_target(
        start=current_pos,
        height=(float(safe_z_min), float(safe_z_max)),
        min_dist=float(min_dist),
        max_dist=float(max_dist),
        clearance=float(self._drone_hit_radius),
        max_tries=80,
        extra_forbidden_fn=dynamic_forbidden,
        xy_limit=float(safe_xy),
        distance_mode='xyz',
        device=self._torch_device,
    )
    if target is not None:
      return target
    if not fallback:
      return None
    return self._fallback_target(current_pos)

  def close(self):
    self._paper_viz.close()
    with _LOCK:
      env_id = int(getattr(self, '_env_id', -1))
      _START_POSITIONS.pop(env_id, None)
      clear_forced_reset(env_id)
      lidar_sensor = getattr(self, '_lidar_sensor', None)
      if lidar_sensor is not None:
        try:
          lidar_sensor.close()
        except Exception:
          pass
      depth_camera = getattr(self, '_depth_camera', None)
      if depth_camera is not None:
        try:
          depth_camera.close()
        except Exception:
          pass
      arena = getattr(self, '_arena', None)
      if arena is not None:
        arena.release(getattr(self, '_env_id', None))
      self._drone = None

  def _setup_scene(self):
    import torch
    import omni.isaac.core.utils.prims as prim_utils
    from omni_drones.robots.robot import RobotBase

    self._torch_device = torch.device(self._device)
    torch.manual_seed(self._seed)

    self._env_pos = self._env_pos.to(self._torch_device)

    self._drone.n = 1
    self._drone.initialize(prim_paths_expr=self._drone_prim_path, track_contact_forces=True)
    self._drone._envs_positions = self._env_pos.reshape(1, 1, 3)
    RobotBase._envs_positions = self._arena._env_positions.unsqueeze(1)

    self._lidar_sensor = _setup_lidar(self, prim_utils)  # Always create LiDAR; _lidar_as_obs only controls obs
    self._depth_camera = None
    if self._depth_camera_enabled:
      self._depth_camera = _setup_depth_camera(self)
      self._arena.set_sensor_render_required(self._env_id, True)

    self._action_adapter = make_action_adapter(
        self._action_mode,
        drone=self._drone,
        device=str(self._device),
        controller=self._controller,
        action_limit_vel=self._action_limit_vel,
        action_limit_yaw_rate=self._action_limit_yaw_rate,
        use_yaw_rate=self._use_yaw_rate,
    )
    self._act_dim = int(self._action_adapter.action_dim())
    self._domain_rand = DomainRandomizer(
        self._dr_cfg,
        drone=self._drone,
        action_adapter=self._action_adapter,
        device=self._torch_device,
    )
    self._domain_rand.setup()
    # Align state encoding with nav: always use the 15-D goal-frame state.
    self._obs_dim = 15
    self._obs_space = self._build_obs_space()

    self._sampler = SceneSampler(
        self._arena.scene,
        bounds_cfg={
            'xy_limit': float(self._bound_xy * 0.45),
            'collision_radius': float(self._drone_hit_radius),
        },
    )
    self._target_local = torch.zeros((1, 1, 3), device=self._torch_device)
    self._target_dir = torch.zeros((1, 1, 3), device=self._torch_device)
    self._target_dir[..., 0] = 1.0
    self._paper_viz.apply_highlight()

  def _move_target_marker(self, target_local):
    try:
      target_world = (target_local.reshape(3) + self._env_pos.reshape(3)).detach().cpu().numpy()
      if self._goal_visible_to_sensors:
        self._arena.set_goal_marker(self._env_id, target_world)
      else:
        self._paper_viz.set_goal(target_world)
    except Exception:
      pass

  def _fallback_target(self, start_local):
    import torch

    safe_xy, safe_z_min, safe_z_max = self._safe_bounds()
    target = torch.zeros_like(start_local)
    target[..., 0] = (torch.rand((1, 1, 1), device=self._torch_device) * 2.0 - 1.0) * safe_xy
    target[..., 1] = (torch.rand((1, 1, 1), device=self._torch_device) * 2.0 - 1.0) * safe_xy
    target[..., 2] = safe_z_min + (safe_z_max - safe_z_min) * torch.rand((1, 1, 1), device=self._torch_device)
    return target

  def _dynamic_peer_positions_world(self):
    import torch

    positions = []
    for dyn in self._arena.iter_dynamic_objects(
        kind='drone',
        lidar_visible=True,
        exclude_owner_env_id=self._env_id,
    ):
      if dyn.pose_w is None:
        continue
      try:
        pos = torch.as_tensor(dyn.pose_w, device=self._torch_device).reshape(-1, 3)[:1]
        positions.append(pos)
      except Exception:
        continue
    return positions

  def _dynamic_obstacles_world(self, *, lidar_visible=None, collidable_only=False):
    return dynamic_obstacles_world(
        self._arena,
        device=self._torch_device,
        lidar_visible=lidar_visible,
        collidable_only=collidable_only,
        fallback_radius=float(self._drone_hit_radius),
    )

  def _dynamic_spawn_forbidden_fn(self, clearance):
    if not self._dynamic_obstacles_enabled:
      return None
    return dynamic_spawn_forbidden_fn(
        self._arena,
        self._env_pos,
        clearance=float(clearance),
        fallback_radius=float(self._drone_hit_radius),
        sample_horizon_s=float(self._dynamic_sample_horizon_s),
        sample_dt_s=float(self._dynamic_sample_dt_s),
    )

  def _dynamic_obstacle_collision(self, center_world):
    if not self._dynamic_obstacles_enabled:
      return False
    obstacle_geometries = self._dynamic_obstacles_world(lidar_visible=None, collidable_only=True)
    margin = float(self._drone_hit_radius) + float(self._dynamic_collision_margin)
    return dynamic_obstacle_collision(
        center_world,
        obstacle_geometries,
        margin=margin,
        device=self._torch_device,
    )

  def _lidar_obs(self, dt):
    import torch

    target_h = int(self._lidar_resolution[0])
    target_v = int(self._lidar_resolution[1])
    lidar_range = float(self._lidar_range)

    lidar_sensor = getattr(self, '_lidar_sensor', None)
    if lidar_sensor is None:
      return np.zeros((target_h, target_v), np.float32)

    frame = lidar_sensor.capture(float(dt))
    if frame.distances is None:
      return np.zeros((target_h, target_v), np.float32)

    target_size = target_h * target_v
    terrain_dists = frame.distances
    if int(terrain_dists.numel()) != target_size:
      raise RuntimeError(
          f'LiDAR ray count mismatch: got {terrain_dists.numel()}, expected {target_size} '
          f'(hres={target_h}, vres={target_v}). '
          'Adjust lidar config to match RayCaster pattern.'
      )

    best_dists = terrain_dists
    ray_origins_w = None
    ray_dirs_unit = None
    if frame.ray_origins_w is not None and frame.ray_hits_w is not None:
      ray_origins_w, ray_dirs_unit = normalize_rays(frame.ray_origins_w, frame.ray_hits_w, target_size)

    other_positions = self._dynamic_peer_positions_world()
    if other_positions and ray_origins_w is not None:
      other_pos = torch.cat(other_positions, dim=0).reshape(-1, 3)
      drone_dists = ray_sphere_batch(
          ray_origins_w,
          ray_dirs_unit,
          other_pos,
          self._drone_hit_radius,
          lidar_range,
      )
      best_dists = torch.min(best_dists, drone_dists)

    obstacle_geometries = self._dynamic_obstacles_world(lidar_visible=True, collidable_only=False)
    best_dists = overlay_dynamic_obstacles_on_lidar(
        best_dists,
        ray_origins_w,
        ray_dirs_unit,
        obstacle_geometries,
        lidar_range=lidar_range,
        device=self._torch_device,
    )

    # Keep LiDAR semantics aligned with nav: return unnormalized proximity as
    # (range - distance), where 0 means no nearby hit and larger is closer.
    scan = torch.clamp(lidar_range - best_dists, min=0.0, max=lidar_range)
    scan = scan.reshape(target_h, target_v)
    return np.asarray(scan.detach().cpu(), np.float32)

  def _depth_obs(self):
    h, w = int(self._depth_resolution[1]), int(self._depth_resolution[0])
    depth_camera = getattr(self, '_depth_camera', None)
    if depth_camera is None:
      return np.zeros((h, w, 1), np.float32)
    return depth_camera.observe()

  def _build_obs_space(self):
    spaces = {
        'state': elements.Space(np.float32, (int(self._obs_dim),)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/reach_goal': elements.Space(np.float32),
        'log/term_reason': elements.Space(np.int32, (), 0, len(self._term_reason_names)),
    }
    if self._obs_mode == 'goal_frame':
      spaces['direction'] = elements.Space(np.float32, (3,))
    if self._lidar_as_obs:
      spaces['lidar'] = elements.Space(np.float32, tuple(int(x) for x in self._lidar_resolution))
    if self._depth_camera_enabled:
      spaces['depth'] = elements.Space(np.float32, (int(self._depth_resolution[1]), int(self._depth_resolution[0]), 1))
    return spaces

  def _encode_obs_state(self, drone_state):
    target_dir = self._target_dir if self._obs_mode == 'goal_frame' else None
    return _encode_goal_relative_state_15d(drone_state, self._target_local, target_dir)

  def _make_obs(
      self,
      *,
      reward,
      is_first,
      is_last,
      is_terminal,
      reach_goal=False,
      term_reason='none',
      drone_state=None,
      lidar_override=None,
      depth_override=None,
  ):
    if drone_state is None:
      drone_state = self._drone.get_state(env_frame=False)
    state_np, direction_np = self._encode_obs_state(drone_state)
    reason_id = self._term_reason_names.index(term_reason) if term_reason in self._term_reason_names else 0
    obs = {
        'state': state_np.astype(np.float32),
        'reward': np.asarray(reward, np.float32),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'log/reach_goal': np.asarray(float(bool(reach_goal)), np.float32),
        'log/term_reason': np.asarray(reason_id, np.int32),
    }
    if direction_np is not None:
      obs['direction'] = direction_np.astype(np.float32)
    if self._lidar_as_obs:
      if lidar_override is None:
        lidar_override = self._lidar_obs(0.0)
      obs['lidar'] = np.asarray(lidar_override, np.float32)
    if self._depth_camera_enabled:
      if depth_override is None:
        depth_override = self._depth_obs()
      obs['depth'] = np.asarray(depth_override, np.float32)
    return obs

  def _safe_bounds(self):
    safe_xy = float(self._bound_xy * 0.45)
    safe_z_min = 0.8
    safe_z_max = float(min(self._bound_z * 0.5, 10.0))
    return safe_xy, safe_z_min, safe_z_max

  def _lidar_collision(self, lidar_scan, hit_threshold=None):
    # lidar_scan stores proximity = (lidar_range - distance); max proximity = min distance.
    if hit_threshold is None:
      hit_threshold = float(self._drone_hit_radius)
    if lidar_scan.size == 0:
      return False
    closest_distance = float(self._lidar_range) - float(np.max(lidar_scan))
    return bool(closest_distance <= float(hit_threshold))

  def _base_link_contact_collision(self, ground_hit, force_threshold=None):
    if force_threshold is None:
      force_threshold = float(self._base_link_collision_force_threshold)
    return _base_link_contact_collision_helper(
        self._drone,
        self._torch_device,
        ground_hit,
        force_threshold=float(force_threshold),
        baseline=getattr(self, '_contact_baseline', None),
    )

  def _update_target_on_success(self, drone_state):
    import torch

    success = bool(
        (torch.norm(self._target_local - drone_state[..., :3], dim=-1) < float(self._goal_radius)).item()
    )
    if not success:
      return False
    current_pos = drone_state[..., :3]
    self._target_local = self._sample_target(current_pos, fallback=True)
    self._target_dir = (self._target_local - current_pos).reshape(1, 1, 3)
    self._target_dir[..., 2] = 0.0
    self._prev_dist = torch.norm((self._target_local - current_pos).reshape(-1, 3), dim=-1).reshape(1, 1)
    self._move_target_marker(self._target_local.reshape(3))
    return True

  def _resolve_term_reason(self, timeout, hasnan, ground_hit, collision, out_of_bounds):
    if not (timeout or bool(hasnan.item()) or bool(ground_hit.item()) or collision or bool(out_of_bounds.item())):
      return 'none'
    if timeout:
      return 'timeout'
    if hasnan.item():
      return 'nan'
    if bool(ground_hit.item()):
      return 'ground'
    if collision:
      return 'collision'
    if bool(out_of_bounds.item()):
      return 'out_of_bounds'
    return 'terminal'

  def _get_close_peer_env_ids(self, center_world, radius):
    import torch

    center = torch.as_tensor(center_world, device=self._torch_device).reshape(1, 3)
    threshold = float(radius)
    close_ids = set()
    for dyn in self._arena.iter_dynamic_objects(
        kind='drone',
        peer_collidable=True,
        exclude_owner_env_id=self._env_id,
    ):
      if dyn.owner_env_id is None or dyn.pose_w is None:
        continue
      try:
        pos = torch.as_tensor(dyn.pose_w, device=self._torch_device).reshape(-1, 3)[:1]
      except Exception:
        continue
      dist = torch.norm(pos - center, dim=-1)
      if bool((dist <= threshold).any().item()):
        close_ids.add(int(dyn.owner_env_id))
    return sorted(close_ids)

  def _compute_obs(self):
    import torch

    lidar = self._lidar_obs(self._physics_dt * self._substeps)  # Always computed; _lidar_as_obs controls obs inclusion
    depth = self._depth_obs() if self._depth_camera_enabled else None
    drone_state = self._drone.get_state(env_frame=False)
    self._paper_viz.on_step(drone_state[..., :3] + self._env_pos.reshape(1, 1, 3))

    pos_dist = torch.norm(self._target_local - drone_state[..., :3], dim=-1)
    step_dt = float(self._physics_dt) * float(self._substeps)

    reward_effort = -self._reward_effort_weight * self._drone.throttle_difference.to(pos_dist.dtype)

    up_z = float(self._drone.up[..., 2].item())
    angvel = self._drone.vel[..., 3:6]
    angvel_norm = torch.norm(angvel, dim=-1)

    tilt_penalty = 0.0
    if up_z < self._tilt_soft_cos:
      tilt_ratio = (self._tilt_soft_cos - up_z) / max(1e-6, self._tilt_soft_cos - self._tilt_hard_cos)
      tilt_penalty = self._tilt_penalty_weight * min(1.0, max(0.0, tilt_ratio))
    angvel_penalty = self._angvel_penalty_weight * angvel_norm

    # -- Alive bonus --
    alive_bonus = float(self._alive_bonus)

    # -- Per-step progress reward (sigmoid-shaped, [-0.7, +0.3]) --
    progress = 0.0
    if self._prev_dist is not None:
      progress = float((self._prev_dist - pos_dist).reshape(()).detach().cpu().item())
    self._prev_dist = pos_dist.detach().reshape(1, 1)
    centered_progress = progress - self._progress_sigmoid_offset
    raw_progress = 1.0 / (1.0 + math.exp(-self._progress_sigmoid_k * centered_progress))
    reward_progress = raw_progress * 1.0 - 0.7

    # -- Sliding-window average speed reward (sigmoid-shaped, [-0.7, +0.3]) --
    self._progress_history.append(progress)
    if len(self._progress_history) < self._avg_speed_warmup:
      reward_avg_speed = 0.0
    else:
      avg_speed = sum(self._progress_history) / (len(self._progress_history) * max(step_dt, 1e-6))
      centered_speed = avg_speed - self._avg_speed_threshold
      raw_speed = 1.0 / (1.0 + math.exp(-self._avg_speed_sigmoid_k * centered_speed))
      reward_avg_speed = raw_speed * 1.0 - 0.7

    reward = (
        alive_bonus
        + reward_progress
        + reward_avg_speed
        + reward_effort
        - tilt_penalty
        - angvel_penalty
    )

    local_pos = drone_state[..., :3]
    grace = bool(self._from_ground and (int(self._step_count) < int(self._ground_grace_steps)))
    if grace:
      ground_hit = (local_pos[..., 2] < 0.2) & (self._drone.up[..., 2] < 0)
    else:
      ground_hit = (local_pos[..., 2] < 0.2) | ((local_pos[..., 2] < 0.4) & (drone_state[..., 9] < -0.5))
    out_of_bounds = (
        (local_pos[..., 0].abs() > self._bound_xy)
        | (local_pos[..., 1].abs() > self._bound_xy)
        | (local_pos[..., 2] > self._bound_z)
    )
    hasnan = torch.isnan(drone_state).any(-1)

    current_world = (drone_state[..., :3] + self._env_pos.reshape(1, 1, 3)).reshape(3)
    close_env_ids = self._get_close_peer_env_ids(current_world, 2.0 * self._drone_hit_radius)
    proximity_collision = len(close_env_ids) > 0
    if proximity_collision:
      with _LOCK:
        request_forced_reset(close_env_ids)

    dynamic_obstacle_collision = bool((not grace) and self._dynamic_obstacle_collision(current_world))
    base_link_collision = bool((not grace) and self._base_link_contact_collision(ground_hit))
    collision = bool(proximity_collision or dynamic_obstacle_collision or base_link_collision)
    if (not collision) and (not grace) and self._lidar_sensor is not None:
      collision = self._lidar_collision(lidar)
    if collision:
      reward -= float(self._collision_penalty)

    reached_goal = self._update_target_on_success(drone_state)
    if reached_goal:
      reward += float(self._goal_bonus)
      self._progress_history.clear()

    terminal = bool((out_of_bounds | ground_hit | hasnan).item() or collision)
    timeout = self._step_count >= self._max_episode_length
    is_last = terminal or timeout
    self._done = is_last

    reason = self._resolve_term_reason(timeout, hasnan, ground_hit, collision, out_of_bounds) if is_last else 'none'
    return self._make_obs(
        reward=np.asarray(reward.reshape(()).detach().cpu(), np.float32),
        is_first=False,
        is_last=bool(is_last),
        is_terminal=bool(terminal),
        reach_goal=reached_goal,
        term_reason=reason,
        drone_state=drone_state,
        lidar_override=lidar,
        depth_override=depth,
    )
