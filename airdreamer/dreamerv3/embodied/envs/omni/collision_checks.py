"""Collision detection helpers for omni drone environments."""


def read_base_link_contact_forces(drone, device):
  """Return the base_link net contact forces as an (N, 3) tensor, or None."""
  import torch

  try:
    forces = drone.base_link.get_net_contact_forces()
  except Exception:
    return None
  if forces is None:
    return None
  if not isinstance(forces, torch.Tensor):
    forces = torch.as_tensor(forces, device=device)
  forces = forces.reshape(-1, 3)
  if forces.numel() == 0:
    return None
  return forces.detach().clone()


def base_link_contact_collision(drone, device, ground_hit, force_threshold=1e-4, baseline=None):
  """Detect contact-force collision on drone base_link.

  PhysX keeps reporting a body's last contact impulse after the body is
  teleported out of contact (e.g. on reset), so when `baseline` (a snapshot
  taken right after reset) is given, only a contact reading that has changed
  since that snapshot counts as a new collision.
  """
  import torch

  if bool(ground_hit):
    return False

  forces = read_base_link_contact_forces(drone, device)
  if forces is None:
    return False

  hit = torch.norm(forces, dim=-1) > float(force_threshold)
  if baseline is not None and baseline.shape == forces.shape:
    changed = torch.norm(forces - baseline.to(forces.device), dim=-1) > float(force_threshold)
    hit = hit & changed
  return bool(hit.any().item())


__all__ = ['base_link_contact_collision', 'read_base_link_contact_forces']
