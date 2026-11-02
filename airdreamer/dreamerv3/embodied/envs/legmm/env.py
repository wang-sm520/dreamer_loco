"""LegMMEnv: A2 Pro + AIRBOT Play nav -> pick -> carry -> place, depth only."""
import math

import elements
import embodied
import numpy as np

from . import goal as goal_lib
from . import interfaces as I
from . import scenes

# Extra log keys on top of interfaces.make_obs_space().
EXTRA_LOGS = {
    'log/skill': elements.Space(np.int32, (), 0, I.NUM_SKILLS),
    'log/skill_switch': elements.Space(np.float32),
    'log/grasp_attempt': elements.Space(np.float32),
    'log/failed_grasp': elements.Space(np.float32),
    'log/arm_collision': elements.Space(np.float32),
    'log/base_disp': elements.Space(np.float32),
    'log/seed': elements.Space(np.int32, (), 0, 2 ** 31 - 1),
}


class LegMMEnv(embodied.Env):

  def __init__(self, task='mm', backend='kin', spaces_only=False, split='train',
               seed=None, logdir=None, **cfg):
    del logdir
    if task != 'mm':
      raise NotImplementedError(f'legmm task variant {task!r}')
    self.task = task
    self.cfg = dict(cfg)
    self.split = split
    self.spaces_only = spaces_only
    self.max_steps = int(cfg.get('max_episode_steps', I.MAX_EPISODE_STEPS))
    self.max_failed = int(cfg.get('max_failed_grasps', 3))
    self.reward_cfg = I.RewardCfg(**{
        k: float(v) for k, v in dict(cfg.get('reward', {})).items()
        if k in I.RewardCfg.__dataclass_fields__})
    self.cap = float(cfg.get('manip_base_speed_cap', 0.15))
    self._obs_space = {**I.make_obs_space(), **EXTRA_LOGS}
    self.labeler = None      # optional callable(env) -> (aff (2,), mask (2,))
    if spaces_only:
      return
    if backend == 'kin':
      from .kin_backend import KinBackend
      self.backend = KinBackend(self.cfg)
    elif backend == 'isaac':
      raise NotImplementedError('Isaac backend is milestone M6.')
    else:
      raise NotImplementedError(backend)
    self.goal = goal_lib.make_goal_provider(cfg.get('goal_provider', {}))
    self.rng = np.random.default_rng(seed)
    self.forced_seed = None  # set to an int to pin the next episode's scene seed
    self._done = True
    self.scene = None
    self.executed = None

  @property
  def obs_space(self):
    return self._obs_space

  @property
  def act_space(self):
    return I.make_act_space()

  # ------------------------------------------------------------------ steps

  def step(self, action):
    if self.spaces_only:
      raise RuntimeError('LegMMEnv was built with spaces_only=True.')
    if action['reset'] or self._done:
      return self._reset()
    act = I.mask_inactive_np({k: action[k] for k in ('skill', 'nav', 'pick', 'place')})
    act['skill'] = np.int32(act['skill'])
    self.executed = act
    skill = I.Skill(int(act['skill']))
    key = I.SKILL_ACTION_KEY[skill]
    vec = np.clip(np.asarray(act[key], np.float32), -1, 1)
    cmd = self._command(skill, vec)
    ev = self.backend.step(cmd)
    delta = 0.0
    if self._prev_skill == skill and self._prev_vec is not None:
      delta = float(np.sum((vec - self._prev_vec) ** 2))
    cur = self.backend.step_state(ev, delta)
    reward = I.compute_reward(self._prev_state, cur, self.reward_cfg)
    switch = self._prev_skill is not None and self._prev_skill != skill
    self._n_switches += int(switch)
    self._prev_state, self._prev_skill, self._prev_vec = cur, skill, vec
    self._step += 1
    reason = self._termination(ev)
    terminal = reason not in ('none', 'timeout')
    self._done = reason != 'none'
    self._term_reason = reason
    logs = {
        'log/skill': np.int32(skill),
        'log/skill_switch': np.float32(switch),
        'log/grasp_attempt': np.float32(ev['grasp_attempt']),
        'log/failed_grasp': np.float32(ev['failed_grasp']),
        'log/arm_collision': np.float32(ev['arm_collision']),
        'log/base_disp': np.float32(ev['base_disp']),
    }
    return self._obs(reward, is_last=self._done, is_terminal=terminal, logs=logs)

  def _command(self, skill, vec):
    if skill == I.Skill.NAV:
      ph = I.scale_action(vec, I.NAV_LAYOUT)
      base = np.concatenate([ph['vx'], ph['vy'], ph['wz']])
      return dict(skill=int(skill), base=base, height=float(ph['height'][0]))
    ph = I.scale_action(vec, I.MANIP_LAYOUT)
    base = ph['base_adj'].astype(np.float64).copy()
    base[2] *= 2.0
    speed = math.hypot(base[0], base[1])
    if speed > self.cap:
      base[:2] *= self.cap / speed
    return dict(skill=int(skill), base=base, height=float(ph['height'][0]),
                dee_pos=ph['dee_pos'], dee_rot=ph['dee_rot'], grip=float(ph['grip'][0]))

  def _termination(self, ev):
    s = self.backend.state
    room = self.scene.room
    if ev['place_success']:
      return 'success'
    if ev['collision']:
      return 'collision'
    if ev['dropped']:
      return 'drop'
    if s.n_failed >= self.max_failed:
      return 'too_many_failed_grasps'
    if abs(s.x) > room[0] / 2 or abs(s.y) > room[1] / 2:
      return 'out_of_bounds'
    if not (np.isfinite(s.ee_b).all() and np.isfinite([s.x, s.y, s.yaw, s.h, s.pitch]).all()):
      return 'nan'
    if self._step >= self.max_steps:
      return 'timeout'
    return 'none'

  def _reset(self):
    seed = self.forced_seed
    if seed is None:
      seed = scenes.sample_seed(self.split, self.rng)
    self.scene = scenes.make_scene(self.split, seed, self.cfg)
    ep_rng = np.random.default_rng(self.rng.integers(2 ** 63))
    from .kin_backend import make_dr
    dr = make_dr(self.cfg, ep_rng)
    self.backend.reset(self.scene, ep_rng, dr)
    self.goal.reset(ep_rng)
    self._done = False
    self._step = 0
    self._n_switches = 0
    self._prev_skill = None
    self._prev_vec = None
    self._term_reason = 'none'
    self._prev_state = self.backend.step_state({})
    self.executed = None
    return self._obs(0.0, is_first=True)

  # ------------------------------------------------------------ observation

  def _goal_obs(self):
    b, s = self.backend, self.backend.state
    # Flat boxes: yaw of the thickness axis, encoded doubled (180 deg symmetric).
    # Cylinders have no orientation: report 0 relative yaw. Place zones: yaw 0.
    if self.scene.obj.shape == 'flat':
      pick = (s.obj_center, math.atan2(s.obj_R[1, 0], s.obj_R[0, 0]), True)
    else:
      pick = (s.obj_center, s.yaw, False)
    targets = [pick, (self.scene.place_point(self.scene.obj.half[2]), 0.0, False)]
    return self.goal(targets, b.base_pos(), b.R_body(), s.yaw)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False, logs=None):
    b = self.backend
    aff, mask = np.zeros(2, np.float32), np.zeros(2, np.float32)
    if self.labeler is not None and not is_last:
      aff, mask = self.labeler(self)
    obs = {
        'depth_head': b.render('head'),
        'depth_wrist': b.render('wrist'),
        'proprio': b.proprio(),
        'goal': self._goal_obs(),
        'label_aff': np.asarray(aff, np.float32),
        'label_aff_mask': np.asarray(mask, np.float32),
        'label_phase': np.int32(b.phase()),
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
        'log/term_reason': np.int32(I.TERM_REASONS.index(self._term_reason)),
        'log/success': np.float32(b.state.placed),
        'log/pick_success': np.float32(b.state.picked),
        'log/scene_type': np.int32(self.scene.scene_type),
        'log/seed': np.int32(self.scene.seed),
    }
    for k in EXTRA_LOGS:
      obs.setdefault(k, np.zeros((), EXTRA_LOGS[k].dtype))
    if logs:
      obs.update(logs)
    return obs

  # ------------------------------------------------- snapshots (labeler, M2)

  def snapshot(self):
    import copy
    return dict(
        backend=self.backend.snapshot(), goal=copy.deepcopy(self.goal.state()),
        env=copy.deepcopy(dict(
            done=self._done, step=self._step, n_switches=self._n_switches,
            prev_skill=self._prev_skill, prev_vec=self._prev_vec,
            term_reason=self._term_reason, prev_state=self._prev_state)))

  def restore(self, snap):
    import copy
    self.backend.restore(snap['backend'])
    self.goal.set_state(copy.deepcopy(snap['goal']))
    e = copy.deepcopy(snap['env'])
    self._done, self._step, self._n_switches = e['done'], e['step'], e['n_switches']
    self._prev_skill, self._prev_vec = e['prev_skill'], e['prev_vec']
    self._term_reason, self._prev_state = e['term_reason'], e['prev_state']

  def close(self):
    pass
