"""LegMMAgEnv: v4 active grasp, version legmm_ag_v1 (brief §3-§6).

Dreamer drives nav = (vx, vy, wz) with the arm held at STOW and the body at its standing height, and a
discrete trigger. trigger = 1 passes a safety gate (shoulder-object distance 0.45-0.70 m, bearing within
+-45 deg, object support height in the table range, base speed <= 0.3 m/s); a failed gate is an invalid
trigger (-0.1) and the step's nav executes normally. A passed gate settles the base with zero commands
(linear < 0.05 m/s and angular < 0.1 rad/s, at most 1.5 s) and re-checks the envelope: a base that slid
out is an invalid trigger and the episode goes on; otherwise the frozen-skill surrogate (ag_skill) runs
from the settled pose and the episode terminates. The observation of that transition is the settled
pre-grasp frame. Settle and skill take physical time that is logged but count as ONE decision step.

Reward parts per decision step (stored separately; the agent recombines them at training time):
  rew_task   = -0.01 (time) [-0.1 invalid] [-2 per contact START] [+20 success / -F failure] + distance shaping
  rew_qshape = q* shaping, F_q = gamma * Phi_q(s') - Phi_q(s), -Phi_q(s) at a real termination
  reward     = rew_task + 1.0 * rew_qshape (reference value; training uses rew_task + lambda * rew_qshape)
A 450-step timeout is a truncation (is_last, not is_terminal) and is counted as failure by the metrics only.
"""
import copy
import math
import time

import elements
import embodied
import numpy as np

from . import ag_scenes
from . import ag_contract as contract
from . import ag_shaping as shp
from . import ag_skill as skill
from . import interfaces as I
from .kin_backend import KinBackend, make_dr

TERM = I.AG_TERM_REASONS
STOW_REL_V4 = (0.36, 0.0, 0.0)


class PosGoal:
  """Noisy target position (sigma, per-episode latency, dropout holding the last value, valid flag).
  Position fields only: no yaw, no place block."""

  def __init__(self, pos_sigma=0.03, latency_steps=(0, 2), dropout=0.05, **kw):
    self.pos_sigma = float(pos_sigma)
    self.latency_steps = tuple(int(x) for x in latency_steps)
    self.dropout = float(dropout)
    self.rng = np.random.default_rng(0)
    self.latency, self.history, self.last = 0, [], None

  def reset(self, rng):
    self.rng = rng
    self.latency = int(rng.integers(self.latency_steps[0], self.latency_steps[1] + 1))
    self.history = []
    self.last = None

  def __call__(self, target_world, base_pos, R_body):
    """Observation vector. The world-frame estimate it is built from (latency / dropout applied) stays in
    `self.last` and is what the environment's safety gate reads."""
    noisy = np.asarray(target_world, np.float64) + self.rng.normal(0, self.pos_sigma, 3)
    self.history.append(noisy)
    if len(self.history) > self.latency + 1:
      self.history.pop(0)
    p = self.history[0]
    drop = self.rng.random() < self.dropout
    if drop and self.last is not None:
      p = self.last
    self.last = p
    d = p - np.asarray(base_pos, np.float64)
    n = float(np.linalg.norm(d))
    dir_b = R_body.T @ d / max(n, 1e-6)
    return np.array([*dir_b, math.hypot(d[0], d[1]), d[2], float(not drop)], np.float32)

  def state(self):
    return (self.latency, list(self.history), self.last)

  def set_state(self, st):
    self.latency, history, self.last = st
    self.history = list(history)


def _cfg_get(cfg, key, default):
  out = dict(default)
  out.update(dict(cfg.get(key, {})))
  return out


class LegMMAgEnv(embodied.Env):

  version = I.AG_VERSION

  def __init__(self, task='ag', backend='kin', spaces_only=False, split='train', seed=None,
               logdir=None, **cfg):
    del logdir
    if task != 'ag':
      raise NotImplementedError(f'legmm_ag task variant {task!r}')
    self.contract = str(cfg.get('contract', 'legacy'))
    if self.contract not in ('legacy', 'true_state'):
      raise ValueError(f'unknown contract: {self.contract}')
    self.task = task
    self.cfg = dict(cfg)
    self.split = split
    self.spaces_only = spaces_only
    self.max_steps = int(cfg.get('max_episode_steps', I.MAX_EPISODE_STEPS))
    self.gate_cfg = shp.gate_cfg(cfg.get('gate', {}))
    self.settle_cfg = _cfg_get(cfg, 'settle', dict(lin=0.05, ang=0.1, max_s=1.5))
    self.skill_cfg = skill.skill_cfg(cfg.get('skill', {}))
    if self.contract == 'true_state':
      self.skill_cfg.update(contract='true_state', contract_gate=self.gate_cfg)
    self.rew = _cfg_get(cfg, 'reward', dict(success=20.0, fail=5.0, invalid=0.1, collision=2.0, time=0.01,
                                            w_dist=2.0, w_q=5.0))
    self.rew['twist_deg'] = self.skill_cfg['twist_deg']
    self.label_cfg = _cfg_get(cfg, 'labels', dict(rate=0.25, terminal=False, seed=0))
    self.scene_filter = str(cfg.get('scene_filter', 'solvable'))
    if self.scene_filter not in ('none', 'solvable'):
      raise ValueError(f'unknown scene_filter: {self.scene_filter}')
    self.filter_max_attempts = int(cfg.get('scene_filter_max_attempts', 100))
    self.solvable_margin = float(cfg.get('solvable_margin', 0.01))      # 1 cm in rho, 3 deg in bearing
    self.record_solvable = bool(cfg.get('record_solvable', True))
    self.gamma = shp.gamma_from_horizon(cfg.get('horizon', 333))
    self.metadata = dict(version=self.version, task='legmm_ag', gamma=self.gamma, gate=self.gate_cfg,
                         settle=self.settle_cfg, skill=self.skill_cfg, reward=self.rew,
                         surrogate=dict(kind='kin', stochastic_grasp=self.skill_cfg['grasp_p'] < 1.0,
                                        grasp_p=self.skill_cfg['grasp_p']),
                         max_episode_steps=self.max_steps)
    dtype = np.uint8 if str(cfg.get('obs_depth_dtype', 'uint8')) == 'uint8' else np.float32
    self._obs_space = I.make_ag_obs_space(dtype)
    if spaces_only:
      return
    if backend != 'kin':
      raise NotImplementedError(backend)
    bcfg = dict(cfg)
    bcfg['stow_rel'] = tuple(cfg.get('stow_rel', STOW_REL_V4))
    bcfg['grasp_p'] = self.skill_cfg['grasp_p']
    self.bcfg = bcfg
    self.backend = KinBackend(bcfg)
    self._shadow_backend = None
    self.goal = PosGoal(**dict(cfg.get('goal_provider', {})))
    self.rng = np.random.default_rng(seed)
    self.label_rng = np.random.default_rng([int(self.label_cfg.get('seed', 0)), 0x1ABE1])
    self.forced_seed = None
    self._est = None                                     # world-frame target estimate of the current observation
    self.filter_stats = {}
    self.scene = None
    self.executed = None
    self._done = True
    self._contact = False                                 # collision flag of the previous decision step
    self._term_reason = 'none'
    self.last_info = {}

  @property
  def obs_space(self):
    return self._obs_space

  @property
  def act_space(self):
    return I.make_ag_act_space()

  # ------------------------------------------------------------------ helpers

  def _shadow(self):
    """A second kin backend that shares the scene but not the state or the RNG."""
    if self._shadow_backend is None:
      self._shadow_backend = KinBackend(self.bcfg)
    sh = self._shadow_backend
    sh.scene = self.scene
    sh._static = self.scene.boxes
    sh.dr = self.backend.dr
    return sh

  def _shoulder(self, s=None):
    s = s or self.backend.state
    return skill.shoulder_world(self.backend, (s.x, s.y, s.yaw))

  def _gate(self, s=None, est=None, true=False):
    """Safety gate. By default it reads the noisy target estimate the policy sees (the one produced for the
    current observation, kept in self._est; `est` overrides it, e.g. for a snapshot), never the true object
    position. `true=True` evaluates the true position (diagnostics only: the gate-vs-truth mismatch logs)."""
    s = s or self.backend.state
    if true:
      c = s.obj_center
    else:
      c = self._est if est is None else est
      if c is None:
        raise RuntimeError('no target estimate yet: the gate is only defined after reset()')
    support = float(c[2] - self.scene.obj.half[2])
    g = contract.gate_cfg(support, self.gate_cfg) if self.contract == 'true_state' else self.gate_cfg
    return shp.gate_report(self._shoulder(s), s.yaw, c, support,
                           float(math.hypot(s.v[0], s.v[1])), g)

  def _potentials(self, s=None):
    s = s or self.backend.state
    return shp.potentials(self._shoulder(s), s.yaw, self.scene.obj, s.obj_center, self.rew)

  def _nav_cmd(self, nav):
    ph = I.scale_action(np.clip(np.asarray(nav, np.float32), -1, 1), I.AG_NAV_LAYOUT)
    base = np.concatenate([ph['vx'], ph['vy'], ph['wz']]).astype(np.float64)
    return dict(skill=int(I.Skill.NAV), base=base, height=float(I.BASE_HEIGHT_RANGE[1]))

  def _settle(self, backend):
    """Zero-command steps until linear < lin and angular < ang, at most max_s. Returns (steps, collided)."""
    c = self.settle_cfg
    n_max = int(math.floor(float(c['max_s']) / backend.hl_dt + 1e-9))
    steps, collided = 0, False
    while steps < n_max:
      v = backend.state.v
      if math.hypot(v[0], v[1]) < float(c['lin']) and abs(v[2]) < float(c['ang']):
        break
      ev = backend.step(dict(skill=int(I.Skill.NAV), base=np.zeros(3), height=float(backend.state.h)))
      steps += 1
      collided |= bool(ev['collision'] or ev['arm_collision'])
    return steps, collided

  def _recheck(self, s, est):
    """Envelope re-check after settling: the (fixed, world-frame) estimate seen from the settled pose, no new
    noise is drawn. Speed is not part of the re-check."""
    c = self._est if est is None else est
    support = float(c[2] - self.scene.obj.half[2])
    g = contract.gate_cfg(support, self.gate_cfg) if self.contract == 'true_state' else self.gate_cfg
    return shp.gate_report(self._shoulder(s), s.yaw, c, support, 0.0, g)

  def _trigger_outcome(self, backend, est=None):
    """Gate -> settle -> envelope re-check -> skill, all on `backend` (a copy). The gate and the re-check
    read the estimate `est` (default: the current one); the skill runs from the true state. Returns a dict."""
    out = dict(gate=self._gate(backend.state, est))
    out['valid'] = bool(out['gate']['ok'])
    if not out['valid']:
      return out
    out['settle_steps'], out['settle_collided'] = self._settle(backend)
    g2 = self._recheck(backend.state, est)
    out['slid_out'] = self.contract == 'legacy' and not g2['envelope_ok']
    if out['slid_out']:
      return out
    t0 = time.perf_counter()
    out['skill'] = skill.run_skill(backend, self.skill_cfg)
    out['skill_wall_s'] = time.perf_counter() - t0
    return out

  def counterfactual(self, snap=None, rng=None):
    """Outcome of triggering now (or from `snap`, an AgEnv.snapshot()) on a copy of the backend: the live
    state, every RNG of the environment, the goal provider and the scene are not touched. `rng` seeds the
    copy's stochastic grasp check (default: a fixed seed, so the call is deterministic).
    Returns dict(valid, success, ...): valid = the gate passed (a label exists)."""
    sh = self._shadow()
    state = snap['backend'][0] if snap is not None else self.backend.state
    est = snap['est'] if snap is not None else self._est
    sh.state = copy.deepcopy(state)
    sh.rng = rng if rng is not None else np.random.default_rng(0)
    t0 = time.perf_counter()
    out = self._trigger_outcome(sh, est)
    out['success'] = bool(out.get('skill') is not None and out['skill'].success)
    out['wall_s'] = time.perf_counter() - t0
    return out

  # ---------------------------------------------------------------- snapshots

  def snapshot(self):
    return dict(
        backend=self.backend.snapshot(), goal=copy.deepcopy(self.goal.state()),
        env=copy.deepcopy(dict(done=self._done, step=self._step, phis=self._phis, contact=self._contact)),
        est=None if self._est is None else self._est.copy(),
        rngs=copy.deepcopy((self.rng.bit_generator.state, self.label_rng.bit_generator.state,
                            self.goal.rng.bit_generator.state)))

  def restore(self, snap):
    self.backend.restore(snap['backend'])
    self.goal.set_state(copy.deepcopy(snap['goal']))
    e = copy.deepcopy(snap['env'])
    self._done, self._step, self._phis = e['done'], e['step'], e['phis']
    self._contact = e['contact']
    self._est = None if snap['est'] is None else snap['est'].copy()
    r = copy.deepcopy(snap['rngs'])
    self.rng.bit_generator.state, self.label_rng.bit_generator.state = r[0], r[1]
    self.goal.rng.bit_generator.state = r[2]

  # -------------------------------------------------------------------- scenes

  def _sample_scene(self):
    self.scene_attempts, self.scene_rejected = 0, 0
    if self.forced_seed is not None:
      return self._finish_scene(ag_scenes.make_ag_scene(self.split, self.forced_seed, self.cfg))
    seed = int(self.rng.integers(*ag_scenes.AG_SPLITS[self.split]))
    shape, blocked, _ = ag_scenes.stratum_of(self.split, seed, self.cfg)
    if self.scene_filter != 'solvable' or self.split not in ('train', 'val'):
      return self._finish_scene(ag_scenes.make_ag_scene(self.split, seed, self.cfg))
    stats = self.filter_stats.setdefault((shape, blocked), dict(attempts=0, rejected=0))
    for attempt in range(self.filter_max_attempts):
      if attempt:
        seed = ag_scenes.sample_seed(self.split, self.rng, shape, blocked, self.cfg)
      scene = ag_scenes.make_ag_scene(self.split, seed, self.cfg)
      self.scene_attempts += 1
      stats['attempts'] += 1
      self.scene = scene
      if skill.solvable_scene(self._shadow(), scene, self.skill_cfg, margin=self.solvable_margin):
        scene.meta['solvable'] = True
        return scene
      self.scene_rejected += 1
      stats['rejected'] += 1
    raise RuntimeError(f'no solvable layout in {self.filter_max_attempts} attempts for {self.split}/{shape}/{blocked}')

  def _finish_scene(self, scene):
    scene.meta.setdefault('solvable', None)
    if scene.meta['solvable'] is None and self.record_solvable:
      self.scene = scene
      scene.meta['solvable'] = bool(skill.solvable_scene(self._shadow(), scene, self.skill_cfg,
                                                        margin=self.solvable_margin))
    return scene

  def _reset(self):
    self.scene = self._sample_scene()
    ep_rng = np.random.default_rng(self.rng.integers(2 ** 63))
    self.backend.reset(self.scene, ep_rng, make_dr(self.cfg, ep_rng))
    self.goal.reset(ep_rng)
    self._done = False
    self._contact = False
    self._step = 0
    self._phis = self._potentials()
    self.executed = None
    self.last_info = {}
    self._term_reason = 'none'
    return self._obs(dict(task=0.0, q=0.0), is_first=True)

  # --------------------------------------------------------------------- step

  def step(self, action):
    if self.spaces_only:
      raise RuntimeError('LegMMAgEnv was built with spaces_only=True.')
    if action['reset'] or self._done:
      return self._reset()
    nav = np.clip(np.asarray(action['nav'], np.float32), -1.0, 1.0)
    trig = int(action['trigger'])
    self.executed = dict(nav=nav, trigger=np.int32(trig))
    b, rw = self.backend, self.rew
    gate = self._gate() if trig == I.TRIGGER_GRASP else None       # reads the estimate the policy saw
    logs = dict(triggered=float(trig == I.TRIGGER_GRASP))
    if gate is not None:                                           # diagnostics: what the true gate would say
      true_ok = self._gate(true=True)['ok']
      logs.update(trig_gate_true_ok=float(true_ok), trig_gate_mismatch=float(true_ok != gate['ok']))
    r_task = -float(rw['time'])
    collided, invalid, terminal, success = False, False, False, False
    pre_xy = (b.state.x, b.state.y)
    phi_prev = self._phis
    res = None

    if gate is not None and gate['ok']:
      n, collided = self._settle(b)
      logs.update(settle_steps=float(n), settle_s=n * b.hl_dt)
      g2 = self._recheck(b.state, None)
      if self.contract == 'true_state' or g2['envelope_ok']:
        terminal = True
      else:
        invalid = True
        logs['slid_out'] = 1.0
    else:
      ev = b.step(self._nav_cmd(nav))
      collided = bool(ev['collision'] or ev['arm_collision'])
      invalid = gate is not None                         # trigger with a failed gate

    # The observation of a terminal transition is the settled pre-grasp frame: the skill runs on a copy.
    if terminal:
      sh = self._shadow()
      sh.state = copy.deepcopy(b.state)
      sh.rng = np.random.default_rng(self.rng.integers(2 ** 63))
      res = skill.run_skill(sh, self.skill_cfg)
      success = bool(res.success)
      logs.update(skill_steps=float(res.steps), skill_s=float(res.sim_s), skill_wall_s=float(res.wall_s),
                  no_plan=float(res.reason == 'no_plan'))
      if self.contract == 'true_state':
        logs['contract_violation'] = float(res.reason == 'contract_violation')
        for name in ('dist', 'bearing', 'height', 'clearance'):
          logs['contract_' + name] = float(not res.contract[name + '_ok'])
      r_task += float(rw['success']) if success else -float(rw['fail'])
    if invalid:
      r_task -= float(rw['invalid'])
      logs['invalid'] = 1.0
    contact_start = collided and not self._contact          # -2 once per contact, not per step
    self._contact = collided
    if collided:
      logs['collision'] = 1.0                                # contact steps (diagnostic)
    if contact_start:
      r_task -= float(rw['collision'])
      logs['collision_start'] = 1.0
    phis_next = self._potentials()
    pd = shp.shaping_diff(phi_prev[0], phis_next[0], self.gamma, terminal)
    pq = shp.shaping_diff(phi_prev[1], phis_next[1], self.gamma, terminal)
    r_task += pd
    self._phis = phis_next
    self._step += 1
    s = b.state
    finite = bool(np.isfinite(b.proprio()).all() and np.isfinite([s.x, s.y, s.yaw, s.h]).all())
    truncated = (not terminal) and (self._step >= self.max_steps or not finite)
    self._done = terminal or truncated
    if terminal:
      reason = 'success' if success else 'skill_fail'
    elif not finite:
      reason = 'nan'
    elif truncated:
      reason = 'timeout'
    else:
      reason = 'none'
    logs['success'] = float(success)
    logs['base_disp'] = float(math.hypot(s.x - pre_xy[0], s.y - pre_xy[1]))
    self._term_reason = reason
    self.last_info = dict(skill=res, reason=reason, gate=gate)
    if self.contract == 'true_state':
      self.last_info['contract'] = res.contract if res is not None else None
    return self._obs(dict(task=r_task, q=pq), is_last=self._done, is_terminal=terminal, logs=logs,
                     outcome=res)

  # ---------------------------------------------------------------- observation

  def _goal_obs(self):
    b, s = self.backend, self.backend.state
    out = self.goal(s.obj_center, b.base_pos(), b.R_body())
    self._est = np.array(self.goal.last, np.float64)             # the gate reads exactly this estimate
    return out

  def _label(self, is_last, outcome):
    """(label, mask): terminal frames reuse the executed outcome (opt-in); other frames are sampled
    inside the gate at `labels.rate` from a separate RNG, so the environment's own stream is unaffected."""
    lc = self.label_cfg
    if is_last:
      if outcome is not None and bool(lc.get('terminal', False)):
        return float(outcome.success), 1.0, 0.0
      return 0.0, 0.0, 0.0
    rate = float(lc.get('rate', 0.0))
    if rate <= 0.0 or not self._gate()['ok']:
      return 0.0, 0.0, 0.0
    if self.label_rng.random() >= rate:
      return 0.0, 0.0, 0.0
    cf = self.counterfactual(rng=np.random.default_rng(self.label_rng.integers(2 ** 63)))
    return float(cf['success']), 1.0, cf['wall_s']

  def _obs(self, parts, is_first=False, is_last=False, is_terminal=False, logs=None, outcome=None):
    b, s = self.backend, self.backend.state
    goal = self._goal_obs()                                      # draws the estimate once per observation
    label, mask, label_wall = self._label(is_last, outcome)
    rt, rq = float(parts['task']), float(parts['q'])
    gate = self._gate()
    gate_true = self._gate(true=True)
    meta = self.scene.meta
    obs = {
        'depth_head': b.render('head'),
        'depth_wrist': b.render('wrist'),
        'proprio': b.proprio(),
        'goal': goal,
        'label_feas': np.float32(label),
        'label_feas_mask': np.float32(mask),
        'rew_task': np.float32(rt),
        'rew_qshape': np.float32(rq),
        'reward': np.float32(rt + rq),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
        'log/term_reason': np.int32(0 if is_first else TERM.index(self._term_reason)),
        'log/gate_ok': np.float32(gate['ok']),
        'log/phi_dist': np.float32(self._phis[0]),
        'log/phi_q': np.float32(self._phis[1]),
        'log/q_star': np.float32(self._phis[2]),
        'log/gate_ok_true': np.float32(gate_true['ok']),
        'log/gate_mismatch': np.float32(gate['ok'] != gate_true['ok']),
        'log/shoulder_dist': np.float32(gate['dist']),              # from the estimate
        'log/shoulder_dist_true': np.float32(gate_true['dist']),
        'log/blocked': np.float32(meta['blocked']),
        'log/flat': np.float32(meta['shape'] == 'flat'),
        'log/solvable': np.float32(-1.0 if meta.get('solvable') is None else float(meta['solvable'])),
        'log/seed': np.int32(self.scene.seed),
        'log/label_wall_s': np.float32(label_wall),
    }
    for k, space in self._obs_space.items():
      obs.setdefault(k, np.zeros((), space.dtype))
    for k, v in (logs or {}).items():
      obs['log/' + k] = np.asarray(v, self._obs_space['log/' + k].dtype)
    return obs

  def close(self):
    pass
