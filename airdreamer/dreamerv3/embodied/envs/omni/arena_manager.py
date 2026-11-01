"""Shared arena singleton manager for Omni environments."""

import json
import threading

_LOCK = threading.Lock()
_ARENA = None
_START_POSITIONS = {}
_FORCED_RESET_ENV_IDS = set()


def arena_lock():
  return _LOCK


def get_shared_arena(
    *,
    headless=None,
    physics_dt=None,
    device=None,
    video_cfg=None,
    logdir=None,
    num_drones=None,
    env_max_envs=None,
    replicate_physics=None,
    drone_model=None,
    terrain_cfg=None,
    perimeter_cfg=None,
    obstacles_cfg=None,
    interaction_mode='multi',
    goal_visible_to_sensors=True,
    **_kw,
):
  global _ARENA
  from .arena import OmniArena

  interaction_mode = str(interaction_mode or 'multi')
  if interaction_mode not in ('multi', 'nav'):
    raise ValueError(f"interaction_mode must be 'multi' or 'nav', got '{interaction_mode}'")

  if _ARENA is None:
    if headless is None or physics_dt is None or device is None or drone_model is None:
      raise RuntimeError('Arena not initialized and required arguments missing.')
    if num_drones is None:
      num_drones = env_max_envs if env_max_envs is not None else 1
    _ARENA = OmniArena(
        headless=headless,
        physics_dt=physics_dt,
        device=device,
        num_drones=num_drones,
        replicate_physics=replicate_physics,
        drone_model=drone_model,
        video_cfg=video_cfg,
        logdir=logdir,
        terrain_cfg=terrain_cfg,
        perimeter_cfg=perimeter_cfg,
        obstacles_cfg=obstacles_cfg,
        interaction_mode=interaction_mode,
        goal_visible_to_sensors=goal_visible_to_sensors,
        shutdown_callback=_clear_global_arena,
    )
    return _ARENA

  requested = _build_scene_signature(
      headless=headless,
      physics_dt=physics_dt,
      device=device,
      num_drones=num_drones if num_drones is not None else env_max_envs,
      replicate_physics=replicate_physics,
      drone_model=drone_model,
      terrain_cfg=terrain_cfg,
      perimeter_cfg=perimeter_cfg,
      obstacles_cfg=obstacles_cfg,
      interaction_mode=interaction_mode,
      goal_visible_to_sensors=goal_visible_to_sensors,
  )
  _validate_scene_signature(_ARENA, requested)
  return _ARENA


def clear_shared_arena(arena=None):
  global _ARENA
  if arena is not None and _ARENA is not arena:
    return
  _ARENA = None
  _START_POSITIONS.clear()
  _FORCED_RESET_ENV_IDS.clear()


def _clear_global_arena(arena):
  clear_shared_arena(arena)


def _build_scene_signature(
    headless=None,
    physics_dt=None,
    device=None,
    num_drones=None,
    replicate_physics=None,
    drone_model=None,
    terrain_cfg=None,
    perimeter_cfg=None,
    obstacles_cfg=None,
    interaction_mode=None,
    goal_visible_to_sensors=None,
    **_kw,
):
  sig = {}
  if headless is not None:
    sig['headless'] = bool(headless)
  if physics_dt is not None:
    sig['physics_dt'] = float(physics_dt)
  if device is not None:
    sig['device'] = str(device)
  if num_drones is not None:
    sig['num_drones'] = int(num_drones)
  if replicate_physics is not None:
    sig['replicate_physics'] = bool(replicate_physics)
  if drone_model is not None:
    sig['drone_model'] = str(drone_model)
  if interaction_mode is not None:
    sig['interaction_mode'] = str(interaction_mode)
  if goal_visible_to_sensors is not None:
    sig['goal_visible_to_sensors'] = bool(goal_visible_to_sensors)
  if terrain_cfg is not None:
    sig['terrain_cfg'] = _normalize_cfg(terrain_cfg)
  if perimeter_cfg is not None:
    sig['perimeter_cfg'] = _normalize_cfg(perimeter_cfg)
  if obstacles_cfg is not None:
    sig['obstacles_cfg'] = _normalize_cfg(obstacles_cfg)
  return sig


def _normalize_cfg(cfg):
  if cfg is None:
    return {}
  def _to_json(obj):
    # Explicitly convert numpy/torch scalars and arrays to native Python types
    # so two semantically different configs cannot produce the same serialization.
    try:
      import numpy as np
      if isinstance(obj, np.integer):
        return int(obj)
      if isinstance(obj, np.floating):
        return float(obj)
      if isinstance(obj, np.ndarray):
        return obj.tolist()
    except ImportError:
      pass
    try:
      import torch
      if isinstance(obj, torch.Tensor):
        return obj.tolist()
    except ImportError:
      pass
    return str(obj)
  return json.loads(json.dumps(dict(cfg), sort_keys=True, default=_to_json))


def _validate_scene_signature(arena, requested):
  mismatches = []

  def _check(key, arena_val, req_val):
    if req_val is None:
      return
    if isinstance(arena_val, float) and isinstance(req_val, float):
      if abs(arena_val - req_val) > 1e-12:
        mismatches.append(f'  {key}: arena={arena_val}, requested={req_val}')
      return
    if isinstance(arena_val, dict) and isinstance(req_val, dict):
      arena_json = json.dumps(arena_val, sort_keys=True, default=str)
      requested_json = json.dumps(req_val, sort_keys=True, default=str)
      if arena_json != requested_json:
        mismatches.append(f'  {key}: arena={arena_json}, requested={requested_json}')
      return
    if str(arena_val) != str(req_val):
      mismatches.append(f'  {key}: arena={arena_val}, requested={req_val}')

  _check('headless', arena.headless, requested.get('headless'))
  _check('physics_dt', arena.physics_dt, requested.get('physics_dt'))
  _check('device', arena.device, requested.get('device'))
  _check('num_drones', arena._num_drones, requested.get('num_drones'))
  _check('replicate_physics', arena._replicate_physics, requested.get('replicate_physics'))
  _check('drone_model', arena._scene_drone_model, requested.get('drone_model'))
  _check('interaction_mode', arena._interaction_mode, requested.get('interaction_mode'))
  _check('goal_visible_to_sensors', arena._goal_visible_to_sensors, requested.get('goal_visible_to_sensors'))
  _check('terrain_cfg', _normalize_cfg(arena._terrain_cfg), requested.get('terrain_cfg'))
  _check('perimeter_cfg', _normalize_cfg(arena._perimeter_cfg), requested.get('perimeter_cfg'))
  _check('obstacles_cfg', _normalize_cfg(arena._obstacles_cfg), requested.get('obstacles_cfg'))

  if mismatches:
    raise RuntimeError(
        'Scene signature mismatch: cannot reuse existing arena.\n'
        'Mismatched fields:\n'
        + '\n'.join(mismatches)
        + '\nPhase 3 does not support mixed multi/nav in the same arena.'
    )


def request_forced_reset(env_ids):
  # Must be called under arena_lock().
  for env_id in env_ids or []:
    _FORCED_RESET_ENV_IDS.add(int(env_id))


def consume_forced_reset(env_id):
  # Must be called under arena_lock().
  env_id = int(env_id)
  if env_id in _FORCED_RESET_ENV_IDS:
    _FORCED_RESET_ENV_IDS.discard(env_id)
    return True
  return False


def clear_forced_reset(env_id):
  # Must be called under arena_lock().
  _FORCED_RESET_ENV_IDS.discard(int(env_id))


def get_start_positions():
  return _START_POSITIONS
