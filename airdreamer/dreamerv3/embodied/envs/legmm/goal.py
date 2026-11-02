"""Goal teacher (brief §4.3): simulates a noisy, delayed, lossy RGB detector."""
import math

import numpy as np

from . import interfaces as I


def encode_target(p_world, yaw_world, base_pos, R_body, base_yaw, valid, double=False):
  """8-dim block of interfaces.GOAL_LAYOUT for one target. double=True encodes
  twice the relative yaw (objects that look the same after a 180 deg turn)."""
  d = np.asarray(p_world, np.float64) - base_pos
  n = float(np.linalg.norm(d))
  dir_b = R_body.T @ d / max(n, 1e-6)
  rel = (yaw_world - base_yaw) * (2.0 if double else 1.0)
  return np.array([*dir_b, math.hypot(d[0], d[1]), d[2], math.sin(rel), math.cos(rel),
                   float(valid)], np.float32)


class GtNoisyGoal:
  """World-frame target estimates with per-step noise, a per-episode latency of
  latency_steps[0]..[1] steps, and per-step dropout (hold last value, valid=0)."""

  def __init__(self, pos_sigma=0.03, yaw_sigma=0.1, latency_steps=(0, 2), dropout=0.05, **kw):
    self.pos_sigma = float(pos_sigma)
    self.yaw_sigma = float(yaw_sigma)
    self.latency_steps = tuple(int(x) for x in latency_steps)
    self.dropout = float(dropout)
    self.rng = np.random.default_rng(0)

  def reset(self, rng):
    self.rng = rng
    self.latency = int(rng.integers(self.latency_steps[0], self.latency_steps[1] + 1))
    self.history = []
    self.last = None

  def __call__(self, targets, base_pos, R_body, base_yaw):
    """targets: [(p_world (3,), yaw_world, double), ...] for pick and place."""
    doubles = [t[2] if len(t) > 2 else False for t in targets]
    noisy = [(np.asarray(t[0]) + self.rng.normal(0, self.pos_sigma, 3),
              t[1] + self.rng.normal(0, self.yaw_sigma)) for t in targets]
    self.history.append(noisy)
    if len(self.history) > self.latency + 1:
      self.history.pop(0)
    delayed = self.history[0]
    out = []
    for i, (p, yaw) in enumerate(delayed):
      drop = self.rng.random() < self.dropout
      if drop and self.last is not None:
        p, yaw = self.last[i]
      out.append((p, yaw, not drop))
    self.last = [(p, yaw) for p, yaw, _ in out]
    return np.concatenate([
        encode_target(p, yaw, base_pos, R_body, base_yaw, valid, dbl)
        for (p, yaw, valid), dbl in zip(out, doubles)])

  def state(self):
    return (self.latency, list(self.history), self.last)

  def set_state(self, st):
    self.latency, history, self.last = st
    self.history = list(history)


def make_goal_provider(cfg):
  cfg = dict(cfg or {})
  kind = cfg.pop('type', 'gt_noisy')
  if kind == 'gt_noisy':
    return GtNoisyGoal(**cfg)
  if kind == 'gt':
    return GtNoisyGoal(pos_sigma=0.0, yaw_sigma=0.0, latency_steps=(0, 0), dropout=0.0)
  raise NotImplementedError(kind)
