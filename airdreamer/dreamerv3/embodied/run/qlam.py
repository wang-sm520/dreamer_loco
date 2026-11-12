"""Weight lambda of the q* shaping term for legmm_ag (brief §5).

reward = rew_task + lambda * rew_qshape is recombined at training time. lambda comes from the real
(not imagined) training episodes: the success rate of the last `window` finished episodes, a truncated
episode counting as a failure. Modes:
  anneal  lambda = clip((hi - success) / (hi - lo), 0, 1), kept non-increasing (a running minimum), and
          1.0 until `window` episodes have finished; defaults window 200, lo 0.2 -> hi 0.5
  fixed   lambda = 1 always (q* shaping as a reference)
  none    lambda = 0 always (pure no-q* control)
It lives in the training process only: the environment processes share no state with it. The train loop
injects the current lambda into every batch (key qstar_lam).
"""
import collections
import threading

import numpy as np


def lambda_of(success, lo, hi):
  return float(min(max((hi - success) / (hi - lo), 0.0), 1.0))


class QLambda:

  def __init__(self, mode='anneal', window=200, lo=0.2, hi=0.5):
    if mode not in ('anneal', 'fixed', 'none'):
      raise ValueError(f'unknown qstar mode: {mode}')
    if not 0.0 <= lo < hi <= 1.0:
      raise ValueError(f'need 0 <= lo < hi <= 1, got lo={lo} hi={hi}')
    self.mode, self.window, self.lo, self.hi = mode, int(window), float(lo), float(hi)
    self.recent = collections.deque(maxlen=self.window)
    self.episodes = 0
    self._lam = 0.0 if mode == 'none' else 1.0
    self._lock = threading.Lock()

  def update(self, success):
    """Call once per finished real episode with 1.0 for a success and 0.0 otherwise."""
    with self._lock:
      self.recent.append(float(success))
      self.episodes += 1
      if self.mode == 'anneal' and len(self.recent) >= self.window:
        self._lam = min(self._lam, lambda_of(float(np.mean(self.recent)), self.lo, self.hi))

  def on_step(self, tran, *args, **kwargs):
    """Driver callback: reads log/success on the last transition of an episode."""
    if bool(tran['is_last']):
      self.update(float(tran['log/success']))

  @property
  def lam(self):
    with self._lock:
      return self._lam

  @property
  def success_rate(self):
    with self._lock:
      return float(np.mean(self.recent)) if self.recent else 0.0

  def stats(self):
    with self._lock:
      return dict(
          lam=float(self._lam),
          success_rate=float(np.mean(self.recent)) if self.recent else 0.0,
          episodes=float(self.episodes), window_full=float(len(self.recent) >= self.window))


  def save(self):
    """Snapshot the schedule and ordered success window under one lock."""
    with self._lock:
      return dict(mode=self.mode, window=self.window, lo=self.lo, hi=self.hi,
                  recent=list(self.recent), episodes=self.episodes, lam=self._lam)

  def load(self, state):
    # A resumed run must retain the schedule that produced this state.
    expected = (self.mode, self.window, self.lo, self.hi)
    actual = tuple(state[k] for k in ('mode', 'window', 'lo', 'hi'))
    if actual != expected:
      raise ValueError(f'qstar schedule mismatch: checkpoint={actual}, run={expected}')
    with self._lock:
      self.recent = collections.deque(state['recent'], maxlen=self.window)
      self.episodes = int(state['episodes'])
      self._lam = float(state['lam'])


def load_or_save_training_checkpoint(cp, qlam):
  """Register qstar state, tolerating old checkpoints which did not save it."""
  if qlam is None:
    cp.load_or_save()
    return
  cp.qlam = qlam
  path = cp.latest()
  if path is None:
    cp.save()
  elif (path / 'qlam.pkl').exists():
    cp.load()
  else:
    print('Legacy checkpoint without qlam: retain fresh lambda/window (old behavior).')
    cp.load(keys=['step', 'agent', 'replay'])
