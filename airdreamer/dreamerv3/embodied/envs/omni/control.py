"""Shared action adapters for Omni drone tasks."""

import math


def _parse_vel_triplet(value, fallback_scalar=None):
  """Return a (x, y, z) float tuple from a scalar, list, or None."""
  if value is None:
    if fallback_scalar is not None:
      v = float(fallback_scalar)
      return (v, v, v)
    return None
  if isinstance(value, (list, tuple)):
    v = [float(x) for x in value]
    while len(v) < 3:
      v.append(v[-1] if v else 0.0)
    return (v[0], v[1], v[2])
  v = float(value)
  return (v, v, v)


class ActionAdapter:
  def action_dim(self):
    raise NotImplementedError

  def apply(self, policy_action, drone_state=None):
    raise NotImplementedError

  def hold(self, drone_state=None):
    raise NotImplementedError


class DirectRPMAdapter(ActionAdapter):
  def __init__(self, drone, device='cuda:0', hold_mode='hover'):
    self._drone = drone
    self._device = str(device)
    self._act_dim = int(self._drone.action_spec.shape[-1])
    self._hold_mode = str(hold_mode or 'hover').lower()

  def action_dim(self):
    return int(self._act_dim)

  def apply(self, policy_action, drone_state=None):
    import torch

    del drone_state
    act = torch.as_tensor(policy_action, device=self._device, dtype=torch.float32).reshape(-1)
    if int(act.numel()) != int(self._act_dim):
      raise ValueError(f'direct_rpm expects {self._act_dim} dims, got {int(act.numel())}.')
    act = act.clamp(-1.0, 1.0).reshape(1, 1, self._act_dim)
    return self._drone.apply_action(act)

  def hold(self, drone_state=None):
    import torch

    del drone_state
    hold_act = torch.zeros((1, 1, self._act_dim), device=self._device)
    if self._hold_mode == 'hover' and getattr(self._drone, 'throttle', None) is not None:
      throttle = torch.as_tensor(self._drone.throttle, device=self._device, dtype=torch.float32)
      throttle = throttle.reshape(-1, self._act_dim)[:1]
      hold_act = torch.clamp(throttle.square() * 2.0 - 1.0, -1.0, 1.0).reshape(1, 1, self._act_dim)
    return self._drone.apply_action(hold_act)


class VelocityCommandAdapter(ActionAdapter):
  def __init__(
      self,
      drone,
      device='cuda:0',
      controller='lee_position',
      action_limit_vel=1.5,
      action_limit_yaw_rate=math.pi,
      use_yaw_rate=True,
      action_body_vel_scale=None,
      action_body_vel_clip=None,
  ):
    self._drone = drone
    self._device = str(device)
    self._controller_name = str(controller) if controller else 'lee_position'
    self._action_limit_vel = float(action_limit_vel)
    self._action_limit_yaw_rate = float(action_limit_yaw_rate)
    self._use_yaw_rate = bool(use_yaw_rate)
    # Per-axis velocity scale; falls back to uniform action_limit_vel when None.
    self._vel_scale = _parse_vel_triplet(action_body_vel_scale, fallback_scalar=action_limit_vel)
    self._vel_clip = _parse_vel_triplet(action_body_vel_clip) or self._vel_scale
    self._controller = self._build_controller(self._controller_name, drone, self._device)
    self._validate_controller_signature(self._controller, require_yaw_rate=self._use_yaw_rate)

  def action_dim(self):
    return 4 if self._use_yaw_rate else 5

  @staticmethod
  def _controller_root_state(drone_state):
    return drone_state[..., :13]

  @staticmethod
  def _sanitize_root_state(drone_state):
    import torch

    state = torch.nan_to_num(drone_state, nan=0.0, posinf=0.0, neginf=0.0).clone()
    quat = state[..., 3:7]
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    bad = norm < 1e-6
    quat = torch.where(bad, torch.zeros_like(quat), quat / norm.clamp(min=1e-6))
    quat[..., 0] = torch.where(bad[..., 0], torch.ones_like(quat[..., 0]), quat[..., 0])
    state[..., 3:7] = quat
    return state

  @staticmethod
  def _sanitize_motor_cmds(motor_cmds):
    import torch

    return torch.nan_to_num(motor_cmds, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

  def apply(self, policy_action, drone_state=None):
    import torch
    from omni_drones.utils.torch import quat_rotate

    expected_dim = self.action_dim()
    act = torch.as_tensor(policy_action, device=self._device, dtype=torch.float32).reshape(-1)
    if int(act.numel()) != expected_dim:
      raise ValueError(f'velocity_cmd expects {expected_dim} dims, got {int(act.numel())}.')
    if drone_state is None:
      drone_state = self._drone.get_state(env_frame=False)
    drone_state = self._sanitize_root_state(self._controller_root_state(drone_state))
    quat_w = drone_state[..., 3:7]
    scale = torch.tensor(list(self._vel_scale), device=self._device, dtype=torch.float32)
    clip = torch.tensor(list(self._vel_clip), device=self._device, dtype=torch.float32)
    vel_body = torch.clamp(
        act[:3].reshape(1, 1, 3) * scale.reshape(1, 1, 3),
        -clip.reshape(1, 1, 3),
        clip.reshape(1, 1, 3),
    )
    target_vel = quat_rotate(quat_w, vel_body)
    kwargs = {
        'root_state': drone_state,
        'target_vel': target_vel,
    }
    if self._use_yaw_rate:
      kwargs['target_yaw_rate'] = torch.tensor(
          [[[float(act[3]) * float(self._action_limit_yaw_rate)]]],
          device=self._device,
      )
    else:
      target_yaw = math.atan2(float(act[4]), float(act[3]))
      kwargs['target_yaw'] = torch.tensor(
          [[[target_yaw]]],
          device=self._device,
      )
    motor_cmds = self._sanitize_motor_cmds(self._controller.compute(**kwargs))
    return self._drone.apply_action(motor_cmds)

  def hold(self, drone_state=None):
    import torch

    if drone_state is None:
      drone_state = self._drone.get_state(env_frame=False)
    drone_state = self._sanitize_root_state(self._controller_root_state(drone_state))
    hold_vel = torch.zeros((1, 1, 3), device=self._device)
    kwargs = {
        'root_state': drone_state,
        'target_vel': hold_vel,
    }
    if self._use_yaw_rate:
      kwargs['target_yaw_rate'] = torch.zeros((1, 1, 1), device=self._device)
    else:
      from omni_drones.utils.torch import quaternion_to_euler
      quat_w = drone_state[..., 3:7]
      euler = quaternion_to_euler(quat_w)
      kwargs['target_yaw'] = euler[..., 2:3]
    motor_cmds = self._sanitize_motor_cmds(self._controller.compute(**kwargs))
    return self._drone.apply_action(motor_cmds)

  @staticmethod
  def _build_controller(name, drone, device):
    try:
      from omni_drones.controllers import ControllerBase
      registry = getattr(ControllerBase, 'REGISTRY', {})
      key = str(name)
      if key in registry:
        ctrl_cls = registry[key]
        return ctrl_cls(drone.gravity[1], drone.params).to(device)
    except Exception:
      pass

    try:
      from omni_drones.controllers import LeePositionController

      return LeePositionController(g=-9.81, uav_params=drone.params).to(device)
    except Exception as exc:
      raise RuntimeError(
          f'Unable to construct velocity controller "{name}". '
          'Please install a compatible omni_drones controller runtime.'
      ) from exc

  @staticmethod
  def _validate_controller_signature(controller, require_yaw_rate=True):
    import inspect

    compute_sig = inspect.signature(controller.compute)
    if require_yaw_rate:
      if 'target_yaw_rate' not in compute_sig.parameters:
        raise TypeError(
            "Controller.compute must support 'target_yaw_rate'. "
            'Detected incompatible omni_drones runtime.'
        )
    else:
      if 'target_yaw' not in compute_sig.parameters:
        raise TypeError(
            "Controller.compute must support 'target_yaw'. "
            'Detected incompatible omni_drones runtime.'
        )


def make_action_adapter(
    action_mode,
    *,
    drone,
    device='cuda:0',
    controller=None,
    action_limit_vel=None,
    action_limit_yaw_rate=None,
    use_yaw_rate=True,
    hold_mode='hover',
    action_body_vel_scale=None,
    action_body_vel_clip=None,
):
  mode = str(action_mode or 'direct_rpm').lower()
  if mode == 'direct_rpm':
    return DirectRPMAdapter(drone=drone, device=device, hold_mode=hold_mode)
  if mode == 'velocity_cmd':
    return VelocityCommandAdapter(
        drone=drone,
        device=device,
        controller=controller or 'lee_position',
        action_limit_vel=1.5 if action_limit_vel is None else float(action_limit_vel),
        action_limit_yaw_rate=math.pi if action_limit_yaw_rate is None else float(action_limit_yaw_rate),
        use_yaw_rate=bool(use_yaw_rate),
        action_body_vel_scale=action_body_vel_scale,
        action_body_vel_clip=action_body_vel_clip,
    )
  raise ValueError(f'Unknown action_mode: {action_mode}')
