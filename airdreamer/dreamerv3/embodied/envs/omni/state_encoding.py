"""Observation state encoders for Omni drone task environments."""

import numpy as np


def encode_goal_relative_state_15d(drone_state, target_local, target_dir=None):
  """Encode 15-D goal-relative state; return (state_np, direction_np)."""
  import torch
  from omni_drones.utils.torch import quat_rotate_inverse

  pos_local = drone_state[..., :3]
  quat_w = drone_state[..., 3:7]
  vel_w = drone_state[..., 7:10]
  vel_ang_w = drone_state[..., 10:13]

  rpos = target_local - pos_local
  distance = rpos.norm(dim=-1, keepdim=True)
  distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
  distance_z = rpos[..., 2:3]

  rpos_body = quat_rotate_inverse(quat_w, rpos / distance.clamp(min=1e-6))
  vel_body = quat_rotate_inverse(quat_w, vel_w)
  vel_ang_body = quat_rotate_inverse(quat_w, vel_ang_w)

  quat_obs = quat_w.reshape(-1, 4)[0].detach().clone()
  if quat_obs[0] < 0:
    quat_obs = -quat_obs

  state_t = torch.cat([
      rpos_body.reshape(-1, 3),
      distance_2d.reshape(-1, 1),
      distance_z.reshape(-1, 1),
      vel_body.reshape(-1, 3),
      vel_ang_body.reshape(-1, 3),
      quat_obs.reshape(-1, 4),
  ], dim=-1).reshape(-1)
  state = np.asarray(state_t.detach().cpu(), np.float32)

  direction = None
  if target_dir is not None:
    direction_t = quat_rotate_inverse(quat_w, target_dir).reshape(-1)
    direction = np.asarray(direction_t.detach().cpu(), np.float32)
  return state, direction


__all__ = ['encode_goal_relative_state_15d']
