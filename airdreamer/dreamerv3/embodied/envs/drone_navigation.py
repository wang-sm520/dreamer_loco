import math

import elements
import embodied
import numpy as np

from embodied.envs.omni.arena_manager import arena_lock, get_shared_arena
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
from embodied.envs.omni.sensors.ray_math import normalize_rays
from embodied.envs.omni.sensors.setup_mixin import (
    setup_depth_camera as _setup_depth_camera,
    setup_lidar as _setup_lidar,
)
from embodied.envs.omni.state_encoding import (
    encode_goal_relative_state_15d as _encode_goal_relative_state_15d,
)
from embodied.envs.omni.visualize import PaperVisualizer

_LOCK = arena_lock()


class OmniDronesNavigation(embodied.Env):
  _TERM_REASON_NAMES = (
      'none', 'timeout', 'nan', 'collision', 'ground', 'out_of_bounds', 'terminal'
  )

  def __init__(
      self,
      task='nav',
      headless=True,
      physics_dt=1 / 60,
      substeps=1,
      device='cuda:0',
      max_episode_length=2000,
      drone_model='hummingbird',
      controller='lee_position',
      action_mode='velocity_cmd',
      action_limit_vel=1.5,
      action_limit_yaw_rate=math.pi,
      use_yaw_rate=True,
      obs_mode='goal_frame',
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
    self._drone_model = str(drone_model)
    self._controller = str(controller) if controller is not None else None
    self._action_mode = str(action_mode or 'velocity_cmd').lower()
    self._action_limit_vel = float(action_limit_vel)
    self._action_limit_yaw_rate = float(action_limit_yaw_rate)
    self._use_yaw_rate = bool(use_yaw_rate)
    self._seed = int(seed)
    self._logdir = str(logdir) if logdir is not None else None

    obs_mode = str(obs_mode or 'goal_frame').lower()
    if obs_mode == 'auto':
      obs_mode = 'goal_frame'
    if obs_mode != 'goal_frame':
      raise ValueError(f'Navigation only supports obs_mode="goal_frame", got {obs_mode!r}.')
    self._obs_mode = obs_mode

    interaction_mode = str(kwargs.pop('interaction_mode', 'nav') or 'nav').lower()
    if interaction_mode != 'nav':
      raise ValueError(
          f'OmniDronesNavigation requires interaction_mode="nav", got {interaction_mode!r}.'
      )

    self._action_body_vel_scale = kwargs.pop('action_body_vel_scale', None)
    self._action_body_vel_clip = kwargs.pop('action_body_vel_clip', None)
    self._include_direction_obs = bool(kwargs.pop('include_direction_obs', True))

    self._step_count = 0
    self._done = True
    self._omni_cached_reset_obs = None
    self._omni_passive_reset_step = False
    self._omni_reset_hold_pending = False

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
    self._map_range = [
        0.5 * terrain_size,
        0.5 * terrain_size,
        float(kwargs.pop('local_bound_z', 3.0)),
    ]
    max_flight_height = kwargs.pop('max_flight_height', None)
    if max_flight_height is not None:
      self._map_range[2] = float(max_flight_height)
    self._spawn_xy_range = float(kwargs.pop('spawn_xy_range', max(0.1, self._map_range[0] - 1.0)))
    self._spawn_height = float(kwargs.pop('spawn_height', 1.0))
    _sgh = kwargs.pop('start_goal_height_range', None)
    if _sgh is not None:
      if isinstance(_sgh, (list, tuple)) and len(_sgh) >= 2:
        self._start_goal_height_range = [float(_sgh[0]), float(_sgh[1])]
      else:
        _h = float(_sgh)
        self._start_goal_height_range = [_h, _h]
    else:
      self._start_goal_height_range = None
    self._spawn_min_dist = float(kwargs.pop('spawn_min_dist', 6.0))
    self._goal_reach_dist = float(kwargs.pop('goal_reach_dist', 3.0))
    self._obs_clearance = float(kwargs.pop('obs_clearance', 1.5))
    self._collision_radius = float(kwargs.pop('collision_radius', 0.3))
    self._base_link_collision_force_threshold = float(
        kwargs.pop('base_link_collision_force_threshold', 1e-4)
    )

    self._goal_visible_to_sensors = bool(kwargs.pop('goal_visible_to_sensors', True))
    self._from_ground = bool(kwargs.pop('from_ground', False))
    self._ground_spawn_z = float(kwargs.pop('ground_spawn_z', 0.1))
    ground_grace_secs = float(kwargs.pop('ground_grace_secs', 2.0))
    denom = float(self._physics_dt) * float(self._substeps)
    self._ground_grace_steps = int(max(0, round(ground_grace_secs / denom))) if denom > 0 else 0
    hold_after_reset = kwargs.pop('hold_after_reset', None)
    self._hold_after_reset = (not self._from_ground) if hold_after_reset is None else bool(hold_after_reset)
    self._reset_step_after_spawn = bool(kwargs.pop('reset_step_after_spawn', True))

    self._reward_progress_scale = float(kwargs.pop('reward_progress_scale', 10.0))
    self._reward_safety_scale = float(kwargs.pop('reward_safety_scale', 5.0))
    self._reward_smoothness_scale = float(kwargs.pop('reward_smoothness_scale', 0.0))
    self._reward_height_scale = float(kwargs.pop('reward_height_scale', 2.0))
    self._reward_collision_penalty = float(kwargs.pop('reward_collision_penalty', 5.0))
    self._reward_reach_goal_bonus = float(kwargs.pop('reward_reach_goal_bonus', 100.0))
    self._safety_margin = float(kwargs.pop('safety_margin', 1.0))
    self._dr_cfg = kwargs.pop('domain_rand', {}) or {}

    cfg_num_drones = kwargs.pop('num_drones', None)
    cfg_max_envs = int(kwargs.pop('max_envs', 16))
    if cfg_num_drones is None:
      cfg_num_drones = cfg_max_envs
    cfg_num_drones = int(cfg_num_drones)
    cfg_replicate_physics = bool(kwargs.pop('replicate_physics', False))

    lidar_cfg = kwargs.pop('lidar', {}) or {}
    self._lidar_as_obs = bool(lidar_cfg.pop('enabled', False))
    self._lidar_range = float(lidar_cfg.pop('range', 4.0))
    self._lidar_vfov = tuple(lidar_cfg.pop('vfov', (-10.0, 20.0)))
    self._lidar_vres = int(lidar_cfg.pop('vres', 4))
    self._lidar_hres = int(lidar_cfg.pop('hres', 72))
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

    self._obs_space = self._build_obs_space()

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
          interaction_mode='nav',
          goal_visible_to_sensors=self._goal_visible_to_sensors,
      )
      self._arena._clients.add(self)
      self._env_id, self._env_pos, self._drone_prim_path, self._drone = self._arena.acquire(
          self._drone_model,
          publish_drone_dynamic=False,
      )

    self._cached_action = None
    self._pre_substep_fn = None

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
      self._omni_reset_hold_pending = False
      self._pre_substep_fn = None
      if self._reset_step_after_spawn:
        self._apply_reset_hold_action()
        self._pre_substep_fn = self._apply_reset_hold_action
        return True
      self._omni_reset_hold_pending = bool(self._hold_after_reset)
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
      if global_step and (own_needs_step or self._omni_passive_reset_step):
        obs = self._current_reset_obs()
      else:
        obs = self._omni_cached_reset_obs
      self._omni_cached_reset_obs = None
      self._omni_passive_reset_step = False
      return obs
    self._step_count += 1
    return self._compute_obs()

  def _apply_reset_hold_action(self):
    self._action_adapter.hold()

  def _zero_reset_velocities(self, view_ids):
    import torch

    self._drone.set_velocities(torch.zeros_like(self._drone.get_velocities()), view_ids)
    if self._drone.is_articulation and self._drone.rotor_joint_indices is not None:
      self._drone._view.set_joint_velocities(
          torch.zeros(1, len(self._drone.rotor_joint_indices), device=self._torch_device),
          joint_indices=self._drone.rotor_joint_indices,
      )

  def _current_reset_obs(self):
    root_state = self._drone.get_state(env_frame=False)
    state_np, direction_np = self._encode_goal_frame_state(root_state)

    pos_local = root_state[..., :3]
    vel_w = root_state[..., 7:10]
    rpos = self._target_local - pos_local
    distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)

    self._prev_distance_2d = distance_2d.detach().clone()
    self._prev_vel_w = vel_w.detach().clone()

    init_dist_2d = float(self._init_distance_2d.reshape(()).detach().cpu().item())
    current_dist_2d = float(distance_2d.reshape(()).detach().cpu().item())
    goal_progress = float((init_dist_2d - current_dist_2d) / (init_dist_2d + 1e-5))

    lidar = self._lidar_obs(self._physics_dt * self._substeps)
    depth = self._depth_obs() if self._depth_camera_enabled else None
    return self._make_obs(
        reward=0.0,
        is_first=True,
        is_last=False,
        is_terminal=False,
        term_reason='none',
        state_override=state_np,
        direction_override=direction_np,
        lidar_override=lidar,
        depth_override=depth,
        reach_goal=False,
        goal_progress=goal_progress,
    )

  def reset(self):
    import torch
    from omni_drones.utils.torch import euler_to_quaternion

    self._paper_viz.on_reset()
    self._step_count = 0
    self._done = False
    self._omni_passive_reset_step = False
    self._omni_reset_hold_pending = False
    self._arena._ensure_timeline_playing()

    view_ids = torch.as_tensor([0], device=self._torch_device)
    self._drone._reset_idx(view_ids, True)
    self._domain_rand.on_reset()
    if self._from_ground and getattr(self._drone, 'throttle', None) is not None:
      self._drone.throttle[view_ids] = 0.0

    start_local, target_local = self._sample_nav_start_target()
    self._target_local = target_local
    self._target_dir = (target_local - start_local).reshape(1, 1, 3)
    self._target_dir[..., 2] = 0.0
    self._move_target_marker(target_local.reshape(3))

    pos_world = start_local + self._env_pos.reshape(1, 1, 3)
    rpy = torch.zeros((1, 1, 3), device=self._torch_device)
    rpy[..., 2] = torch.rand((1, 1), device=self._torch_device) * (2.0 * math.pi) - math.pi
    rot = euler_to_quaternion(rpy)
    self._drone.set_world_poses(pos_world, rot, view_ids)
    self._zero_reset_velocities(view_ids)
    if self._reset_step_after_spawn:
      self._arena.step_reset_physics(render=bool(self._depth_camera_enabled))
      self._zero_reset_velocities(view_ids)
    elif self._depth_camera_enabled:
      self._arena.render_current()

    rpos = self._target_local - start_local
    self._prev_distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True).detach().clone()
    self._init_distance_2d = self._prev_distance_2d.detach().clone()
    self._contact_baseline = _read_base_link_contact_forces(self._drone, self._torch_device)
    self._prev_vel_w = torch.zeros(1, 1, 3, device=self._torch_device)
    return self._make_obs(
        reward=0.0,
        is_first=True,
        is_last=False,
        is_terminal=False,
        term_reason='none',
        reach_goal=False,
        goal_progress=0.0,
    )

  def close(self):
    self._paper_viz.close()
    with _LOCK:
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

  def _build_obs_space(self):
    spaces = {
        'state': elements.Space(np.float32, (15,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/term_reason': elements.Space(np.int32, (), 0, len(self._TERM_REASON_NAMES)),
        'log/reach_goal': elements.Space(np.float32),
        'log/goal_progress': elements.Space(np.float32),
    }
    if self._include_direction_obs:
      spaces['direction'] = elements.Space(np.float32, (3,))
    if self._lidar_as_obs:
      spaces['lidar'] = elements.Space(np.float32, tuple(int(x) for x in self._lidar_resolution))
    if self._depth_camera_enabled:
      h, w = int(self._depth_resolution[1]), int(self._depth_resolution[0])
      spaces['depth'] = elements.Space(np.float32, (h, w, 1))
    return spaces

  def _setup_scene(self):
    import torch
    import omni.isaac.core.utils.prims as prim_utils
    from omni_drones.robots.robot import RobotBase

    self._torch_device = torch.device(self._device)
    torch.manual_seed(self._seed)

    self._env_pos = self._env_pos.to(self._torch_device)
    self._drone.n = 1
    self._drone.initialize(prim_paths_expr=self._drone_prim_path, track_contact_forces=False)
    self._drone._envs_positions = self._env_pos.reshape(1, 1, 3)
    RobotBase._envs_positions = self._arena._env_positions.unsqueeze(1)

    self._lidar_sensor = _setup_lidar(self, prim_utils)
    self._depth_camera = None
    if self._depth_camera_enabled:
      self._depth_camera = _setup_depth_camera(self)
      self._arena.set_sensor_render_required(self._env_id, True)

    self._action_adapter = make_action_adapter(
        self._action_mode,
        drone=self._drone,
        device=str(self._device),
        controller=self._controller or 'lee_position',
        action_limit_vel=self._action_limit_vel,
        action_limit_yaw_rate=self._action_limit_yaw_rate,
        use_yaw_rate=self._use_yaw_rate,
        action_body_vel_scale=self._action_body_vel_scale,
        action_body_vel_clip=self._action_body_vel_clip,
    )
    self._act_dim = int(self._action_adapter.action_dim())
    self._domain_rand = DomainRandomizer(
        self._dr_cfg,
        drone=self._drone,
        action_adapter=self._action_adapter,
        device=self._torch_device,
    )
    self._domain_rand.setup()
    self._sampler = SceneSampler(
        self._arena.scene,
        bounds_cfg={
            'xy_limit': float(self._spawn_xy_range),
            'obstacle_collision_radius': 0.0,
            'perimeter_collision_radius': float(self._collision_radius),
        },
    )
    self._target_local = torch.zeros((1, 1, 3), device=self._torch_device)
    self._target_dir = torch.zeros((1, 1, 3), device=self._torch_device)
    self._target_dir[..., 0] = 1.0
    self._current_nav_height = float(self._spawn_height)
    self._prev_distance_2d = torch.zeros((1, 1, 1), device=self._torch_device)
    self._init_distance_2d = torch.zeros((1, 1, 1), device=self._torch_device)
    self._prev_vel_w = torch.zeros((1, 1, 3), device=self._torch_device)
    self._paper_viz.apply_highlight()

  def _sample_nav_start_target(self):
    if self._start_goal_height_range is not None:
      import torch
      hmin, hmax = self._start_goal_height_range
      episode_h = float(
          torch.rand(1, device=self._torch_device) * (hmax - hmin) + hmin
      )
      self._current_nav_height = episode_h
      start_height = float(self._ground_spawn_z) if self._from_ground else episode_h
      target_height = episode_h
    else:
      start_height = float(self._ground_spawn_z) if self._from_ground else float(self._spawn_height)
      target_height = float(self._spawn_height)
      self._current_nav_height = target_height
    xy_limit = float(max(0.1, self._spawn_xy_range))
    clearance = float(self._obs_clearance)
    extra_forbidden_fn = self._dynamic_spawn_forbidden_fn(clearance)

    start = self._sampler.sample_start(
        height=start_height,
        clearance=clearance,
        max_tries=200,
        extra_forbidden_fn=extra_forbidden_fn,
        xy_limit=xy_limit,
        device=self._torch_device,
    )
    if start is None:
      import torch

      start = torch.zeros((1, 1, 3), device=self._torch_device)
      start[..., 2] = start_height

    target = self._sampler.sample_target(
        start=start,
        height=target_height,
        min_dist=float(self._spawn_min_dist),
        clearance=clearance,
        max_tries=200,
        extra_forbidden_fn=extra_forbidden_fn,
        xy_limit=xy_limit,
        distance_mode='xy',
        device=self._torch_device,
    )
    if target is None:
      target = self._fallback_target_mirror(start)
    return start, target

  def _fallback_target_mirror(self, start_local):
    import torch

    target = torch.zeros_like(start_local)
    target[..., 0] = -start_local[..., 0]
    target[..., 1] = -start_local[..., 1]
    target[..., 2] = float(getattr(self, '_current_nav_height', self._spawn_height))
    return target

  def _move_target_marker(self, target_local):
    try:
      target_world = (target_local.reshape(3) + self._env_pos.reshape(3)).detach().cpu().numpy()
      if self._goal_visible_to_sensors:
        self._arena.set_goal_marker(self._env_id, target_world)
      else:
        self._paper_viz.set_goal(target_world)
    except Exception:
      pass

  def _lidar_obs(self, dt):
    import torch

    h, v = int(self._lidar_resolution[0]), int(self._lidar_resolution[1])
    lidar_sensor = getattr(self, '_lidar_sensor', None)
    if lidar_sensor is None:
      return np.zeros((h, v), np.float32)

    frame = lidar_sensor.capture(float(dt))
    if frame.distances is None:
      return np.zeros((h, v), np.float32)

    target_size = h * v
    terrain_dists = frame.distances
    if int(terrain_dists.numel()) != target_size:
      raise RuntimeError(
          f'LiDAR ray count mismatch: got {terrain_dists.numel()}, expected {target_size} '
          f'(hres={h}, vres={v}).'
      )

    best_dists = terrain_dists
    lidar_range = float(self._lidar_range)
    ray_origins_w = None
    ray_dirs_unit = None
    if frame.ray_origins_w is not None and frame.ray_hits_w is not None:
      ray_origins_w, ray_dirs_unit = normalize_rays(frame.ray_origins_w, frame.ray_hits_w, target_size)

    obstacle_geometries = self._dynamic_obstacles_world(lidar_visible=True, collidable_only=False)
    best_dists = overlay_dynamic_obstacles_on_lidar(
        best_dists,
        ray_origins_w,
        ray_dirs_unit,
        obstacle_geometries,
        lidar_range=lidar_range,
        device=self._torch_device,
    )

    # Encode as proximity = (range - distance): 0 means nothing nearby, range means contact.
    scan = torch.clamp(lidar_range - best_dists, min=0.0, max=lidar_range)
    return np.asarray(scan.reshape(h, v).detach().cpu(), np.float32)

  def _depth_obs(self):
    h, w = int(self._depth_resolution[1]), int(self._depth_resolution[0])
    depth_camera = getattr(self, '_depth_camera', None)
    if depth_camera is None:
      return np.zeros((h, w, 1), np.float32)
    return depth_camera.observe()

  def _closest_lidar_distance(self, lidar_scan):
    # lidar_scan stores proximity = (lidar_range - distance); max proximity = min distance.
    if self._lidar_sensor is None or lidar_scan.size == 0:
      return float(self._lidar_range)
    return float(self._lidar_range) - float(np.max(lidar_scan))

  def _lidar_collision(self, lidar_scan, hit_threshold=None):
    threshold = float(self._collision_radius if hit_threshold is None else hit_threshold)
    return bool(self._closest_lidar_distance(lidar_scan) <= threshold)

  def _dynamic_obstacles_world(self, *, lidar_visible=None, collidable_only=False):
    return dynamic_obstacles_world(
        self._arena,
        device=self._torch_device,
        lidar_visible=lidar_visible,
        collidable_only=collidable_only,
        fallback_radius=float(self._collision_radius),
    )

  def _dynamic_obstacle_collision(self, center_world):
    if not self._dynamic_obstacles_enabled:
      return False
    margin = float(self._collision_radius) + float(self._dynamic_collision_margin)
    obstacle_geometries = self._dynamic_obstacles_world(lidar_visible=None, collidable_only=True)
    return dynamic_obstacle_collision(
        center_world,
        obstacle_geometries,
        margin=margin,
        device=self._torch_device,
    )

  def _dynamic_spawn_forbidden_fn(self, clearance):
    if not self._dynamic_obstacles_enabled:
      return None
    return dynamic_spawn_forbidden_fn(
        self._arena,
        self._env_pos,
        clearance=float(clearance),
        fallback_radius=float(self._collision_radius),
        sample_horizon_s=float(self._dynamic_sample_horizon_s),
        sample_dt_s=float(self._dynamic_sample_dt_s),
    )

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

  def _encode_goal_frame_state(self, root_state):
    return _encode_goal_relative_state_15d(root_state, self._target_local, self._target_dir)

  def _make_obs(
      self,
      *,
      reward,
      is_first,
      is_last,
      is_terminal,
      term_reason='none',
      state_override=None,
      direction_override=None,
      lidar_override=None,
      depth_override=None,
      reach_goal=False,
      goal_progress=0.0,
  ):
    needs_recompute = state_override is None or (
        self._include_direction_obs and direction_override is None
    )
    if needs_recompute:
      root_state = self._drone.get_state(env_frame=False)
      state_override, direction_override = self._encode_goal_frame_state(root_state)
    reason_id = (
        self._TERM_REASON_NAMES.index(term_reason) if term_reason in self._TERM_REASON_NAMES else 0
    )
    obs = {
        'state': np.asarray(state_override, np.float32),
        'reward': np.asarray(reward, np.float32),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'log/term_reason': np.asarray(reason_id, np.int32),
        'log/reach_goal': np.asarray(float(bool(reach_goal)), np.float32),
        'log/goal_progress': np.asarray(float(goal_progress), np.float32),
    }
    if self._include_direction_obs:
      obs['direction'] = np.asarray(direction_override, np.float32)
    if self._lidar_as_obs:
      if lidar_override is None:
        lidar_override = self._lidar_obs(0.0)
      obs['lidar'] = np.asarray(lidar_override, np.float32)
    if self._depth_camera_enabled:
      if depth_override is None:
        depth_override = self._depth_obs()
      obs['depth'] = np.asarray(depth_override, np.float32)
    return obs

  def _resolve_term_reason(self, timeout, hasnan, ground_hit, collision, out_of_bounds, reach_goal):
    if timeout:
      return 'timeout'
    if hasnan:
      return 'nan'
    if ground_hit:
      return 'ground'
    if collision:
      return 'collision'
    if out_of_bounds:
      return 'out_of_bounds'
    if reach_goal:
      return 'terminal'
    return 'none'

  def _compute_nav_reward_done(self, root_state, lidar):
    import torch

    pos_local = root_state[..., :3]
    vel_w = root_state[..., 7:10]
    rpos = self._target_local - pos_local
    distance = rpos.norm(dim=-1, keepdim=True)
    distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)

    reward_progress = float((self._prev_distance_2d - distance_2d).reshape(()).detach().cpu().item())
    self._prev_distance_2d = distance_2d.detach().clone()

    closest_distance = self._closest_lidar_distance(lidar)
    safety_margin = max(1e-6, float(self._safety_margin))
    penetration = max(0.0, safety_margin - closest_distance) / safety_margin
    reward_safety = -(penetration ** 2)

    vel_change = torch.norm(vel_w - self._prev_vel_w, dim=-1)
    penalty_smooth = float(vel_change.reshape(()).detach().cpu().item())
    self._prev_vel_w = vel_w.detach().clone()

    penalty_height = float(
        ((self._target_local[..., 2] - pos_local[..., 2]) ** 2).reshape(()).detach().cpu().item()
    )

    reward = (
        float(self._reward_progress_scale) * reward_progress
        + float(self._reward_safety_scale) * reward_safety
        - float(self._reward_smoothness_scale) * penalty_smooth
        - float(self._reward_height_scale) * penalty_height
    )

    reach_goal = bool(float(distance.reshape(()).detach().cpu().item()) < float(self._goal_reach_dist))
    init_dist_2d = float(self._init_distance_2d.reshape(()).detach().cpu().item())
    current_dist_2d = float(distance_2d.reshape(()).detach().cpu().item())
    goal_progress = float((init_dist_2d - current_dist_2d) / (init_dist_2d + 1e-5))
    if reach_goal:
      reward += float(self._reward_reach_goal_bonus)

    grace = bool(self._from_ground and (int(self._step_count) < int(self._ground_grace_steps)))
    if grace:
      ground_hit = (pos_local[..., 2] < 0.2) & (self._drone.up[..., 2] < 0)
    else:
      ground_hit = pos_local[..., 2] < 0.2
    out_xy = (pos_local[..., 0].abs() > float(self._map_range[0])) | (pos_local[..., 1].abs() > float(self._map_range[1]))
    above = pos_local[..., 2] > float(self._map_range[2])
    hasnan = bool(torch.isnan(root_state).any().item())

    collision = False
    center_world = (root_state[..., :3] + self._env_pos.reshape(1, 1, 3)).reshape(1, 3)
    if (not collision) and (not grace):
      collision = self._dynamic_obstacle_collision(center_world)
    if (not collision) and (not grace) and self._lidar_sensor is not None:
      collision = self._lidar_collision(lidar)
    if collision:
      reward -= float(self._reward_collision_penalty)

    out_of_bounds = bool((out_xy | above).any().item())
    ground_hit_bool = bool(ground_hit.any().item())
    terminal = bool(ground_hit_bool or out_of_bounds or hasnan or collision or reach_goal)
    timeout = int(self._step_count) >= int(self._max_episode_length)
    is_last = bool(terminal or timeout)
    reason = (
        self._resolve_term_reason(timeout, hasnan, ground_hit_bool, collision, out_of_bounds, reach_goal)
        if is_last else 'none'
    )
    return reward, is_last, terminal, reason, reach_goal, goal_progress

  def _compute_obs(self):
    root_state = self._drone.get_state(env_frame=False)
    self._paper_viz.on_step(root_state[..., :3] + self._env_pos.reshape(1, 1, 3))
    state_np, direction_np = self._encode_goal_frame_state(root_state)

    lidar = self._lidar_obs(self._physics_dt * self._substeps)
    depth = self._depth_obs() if self._depth_camera_enabled else None
    reward, is_last, is_terminal, reason, reach_goal, goal_progress = self._compute_nav_reward_done(
        root_state, lidar
    )
    self._done = bool(is_last)
    return self._make_obs(
        reward=reward,
        is_first=False,
        is_last=is_last,
        is_terminal=is_terminal,
        term_reason=reason,
        state_override=state_np,
        direction_override=direction_np,
        lidar_override=lidar,
        depth_override=depth,
        reach_goal=reach_goal,
        goal_progress=goal_progress,
    )
