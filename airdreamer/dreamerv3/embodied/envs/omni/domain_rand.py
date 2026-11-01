"""Shared domain-randomization helpers for omni drone environments."""


class DomainRandomizer:
  """Manages drone dynamics and controller gain randomization."""

  def __init__(self, cfg, *, drone, action_adapter, device):
    cfg = cfg or {}
    self._enabled = bool(cfg.get('enabled', False))
    self._drone_cfg = dict(cfg.get('drone', {}) or {})
    self._ctrl_cfg = dict(cfg.get('controller', {}) or {})
    self._drone = drone
    self._adapter = action_adapter
    self._device = device
    self._base_gains = None

  def setup(self):
    if not self._enabled:
      return
    if self._drone_cfg:
      self._drone.setup_randomization({'train': self._drone_cfg})
    ctrl = self._get_lee_controller()
    if ctrl is not None and self._ctrl_cfg:
      self._base_gains = {
          'pos': ctrl.pos_gain.data.clone(),
          'vel': ctrl.vel_gain.data.clone(),
          'att': ctrl.attitute_gain.data.clone(),
          'ang_rate': ctrl.ang_rate_gain.data.clone(),
      }

  def on_reset(self):
    if not self._enabled or self._base_gains is None or not self._ctrl_cfg:
      return
    import torch

    ctrl = self._get_lee_controller()
    if ctrl is None:
      return

    def _scale(base, key):
      rng = self._ctrl_cfg.get(key)
      if rng is None:
        return base
      lo, hi = float(rng[0]), float(rng[1])
      return base * torch.empty(base.shape, device=self._device).uniform_(lo, hi)

    ctrl.pos_gain.data = _scale(self._base_gains['pos'], 'pos_gain_scale')
    ctrl.vel_gain.data = _scale(self._base_gains['vel'], 'vel_gain_scale')
    ctrl.attitute_gain.data = _scale(self._base_gains['att'], 'att_gain_scale')
    ctrl.ang_rate_gain.data = _scale(self._base_gains['ang_rate'], 'ang_rate_gain_scale')

  def _get_lee_controller(self):
    from embodied.envs.omni.control import VelocityCommandAdapter

    if isinstance(self._adapter, VelocityCommandAdapter):
      return getattr(self._adapter, '_controller', None)
    return None
