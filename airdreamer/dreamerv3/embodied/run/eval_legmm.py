"""Small, CPU-friendly LegMM policy evaluator.

The policy API's ``mode`` argument is passed as ``mode='eval'`` below.  The
current :class:`dreamerv3.agent.Agent` samples its action distributions in
both modes (it does not take the mode), so this evaluator reports sampled
policy behavior rather than an argmax policy.

The B1 and B2 baselines replace only the switching decision. Navigation and
manipulation use the :class:`~embodied.envs.legmm.expert.ScriptedExpert`
controllers unchanged; the switch rule reads the observed ``obs['goal']``
(the same noisy ``GoalProvider`` observation seen by the agent), never a
privileged target. Their manipulation plan is anchored at the current base
pose, so the controller's own ``base_adj`` is the only allowed adjustment.
"""

import dataclasses
import json
import math
import pathlib
import time
from collections import Counter

import numpy as np
import elements

from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import expert
from embodied.envs.legmm import labeler as labeler_lib
from embodied.envs.legmm import planning
from embodied.envs.legmm import scenes
from embodied.envs.legmm import kin_backend as kb


def bootstrap_ci(values, n_resamples=1000, seed=0, confidence=0.95):
  """Return the percentile bootstrap confidence interval for a mean."""
  values = np.asarray(values, dtype=float).reshape(-1)
  if not len(values):
    return [float('nan'), float('nan')]
  rng = np.random.default_rng(seed)
  means = np.mean(values[rng.integers(0, len(values), (int(n_resamples), len(values)))], axis=1)
  alpha = (1.0 - confidence) / 2.0
  return [float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))]


def _resolve_scene_filter(split, scene_filter):
  scene_filter = str(scene_filter or 'auto').lower()
  if scene_filter not in ('auto', 'solvable', 'none'):
    raise ValueError(f'unknown eval.scene_filter: {scene_filter!r}')
  if scene_filter == 'auto':
    return 'solvable' if split in ('train', 'val') else 'none'
  return scene_filter


def select_seeds(split, scene_types=None, episodes=None, scene_filter='auto', cfg=None,
                 return_info=False):
  """Select split seeds in order, optionally retaining expert-solvable scenes."""
  if scene_types:
    parsed = parse_scene_types(scene_types)
    wanted = {int(x) for x in parsed}
  else:
    wanted = None
  lo, hi = I.SPLITS[split]['seeds']
  candidates = [seed for seed in range(lo, hi)
                if wanted is None or int(scenes.scene_type_of(split, seed)) in wanted]
  used_filter = _resolve_scene_filter(split, scene_filter)
  kept, rejected, considered = [], 0, 0
  for seed in candidates:
    considered += 1
    solvable = True
    if used_filter == 'solvable':
      scene = scenes.make_scene(split, seed, cfg or {})
      solvable = bool(expert.scene_solvable(scene, cfg or {}))
    if solvable:
      kept.append(seed)
    else:
      rejected += 1
    if episodes is not None and len(kept) >= int(episodes):
      break
  if not kept:
    raise ValueError(f'No seeds in split {split!r} match scene_types={scene_types!r}')
  selected = kept if episodes is None else [kept[i % len(kept)] for i in range(int(episodes))]
  info = {'scene_filter': used_filter, 'seeds_considered': considered,
          'seeds_rejected': rejected}
  return (selected, info) if return_info else selected


def parse_scene_types(value):
  if value is None or value == '' or value == ():
    return None
  if isinstance(value, (str, int)):
    value = str(value).split(',')
  names = {x.name: int(x) for x in scenes.SceneType}
  out = []
  for item in value:
    item = str(item).strip()
    if not item:
      continue
    out.append(names[item.upper()] if not item.lstrip('-').isdigit() else int(item))
  return tuple(out)


def count_false_triggers(skill_seq, aff_seq):
  """Evaluator's single counting path, kept beside the episode collector."""
  return I.false_trigger(np.asarray(skill_seq), np.asarray(aff_seq))


def _as_action(acts):
  return {key: np.asarray(value[0]) for key, value in acts.items()}


def _set_forced_seed(env, seed):
  """Set the LegMM field through the normal wrapper chain."""
  base = env
  while hasattr(base, 'env'):
    base = base.env
  base.forced_seed = int(seed)


class BaselinePolicy:
  """B1/B2 switch policy using the existing scripted manipulation controller."""

  def __init__(self, kind, env, heading_deg=30.0, switch_distance=0.75):
    kind = str(kind).lower()
    if kind not in ('b1', 'b2'):
      raise ValueError(f'unknown baseline policy: {kind!r}')
    self.kind = kind
    self.env = env
    self.heading_rad = np.deg2rad(float(heading_deg))
    self.switch_distance = float(switch_distance)
    self.expert = expert.ScriptedExpert(expert.ExpertConfig(diversity=False))
    self.ctrl = None

  def reset(self, seed=0):
    self.expert.reset(self.env.scene, np.random.default_rng(int(seed)), self.env.backend)
    self.ctrl = None
    self.planner = planning.GridPlanner(self.env.scene, scenes.NAV_INFLATE)
    self.path = None
    self.path_target = None

  @staticmethod
  def _goal_block(obs):
    # grip_effort is an observed proprioceptive signal: the backend reports
    # 0.8 while carrying and 0.2 for a closed but empty gripper.
    proprio = np.asarray(obs.get('proprio', ()), dtype=np.float32).reshape(-1)
    effort = (float(proprio[I.PROPRIO_LAYOUT['grip_effort'].start])
              if proprio.size > I.PROPRIO_LAYOUT['grip_effort'].start else 0.0)
    width = (float(proprio[I.PROPRIO_LAYOUT['grip_width'].start])
             if proprio.size > I.PROPRIO_LAYOUT['grip_width'].start else 1.0)
    # Deployment has no label_phase. Holding is the observed gripper state:
    # effort is high while carrying and the measured opening is below fully open.
    holding = effort >= 0.5 and width < 0.98
    return 8 if holding else 0, holding

  def _observed_target(self, obs):
    goal = np.asarray(obs['goal'], dtype=np.float32).reshape(-1)
    start, holding = self._goal_block(obs)
    direction = goal[start:start + 3]
    distance = float(goal[start + 3])
    bearing = float(np.arctan2(direction[1], direction[0]))
    return direction, distance, bearing, holding

  def should_switch(self, obs):
    """Return the exact strict threshold decision used by B1/B2."""
    _, distance, bearing, _ = self._observed_target(obs)
    if not (distance < self.switch_distance):
      return False
    return self.kind == 'b1' or abs(bearing) <= self.heading_rad

  def observed_goal_distance(self, obs):
    return float(self._observed_target(obs)[1])

  def _goal_xy(self, obs):
    direction, distance, _, _ = self._observed_target(obs)
    body_xy = np.asarray(direction[:2], np.float64)
    norm = float(np.linalg.norm(body_xy))
    if not np.isfinite(distance) or norm < 1e-6:
      return None
    rel_b = np.array([*(body_xy / norm * distance), 0.0])
    return self.env.backend.base_pos() + self.env.backend.R_body() @ rel_b

  def _nav(self, obs):
    target = self._goal_xy(obs)
    if target is None:
      return expert.nav_action()
    state = self.env.backend.state
    pos = np.array([state.x, state.y], np.float64)
    target_xy = np.asarray(target[:2], np.float64)
    # Reuse the expert's GridPlanner and pure-pursuit path follower. Only the
    # destination changes: it is the observed goal point, not a stance pose.
    if (self.path is None or self.path_target is None or
        np.linalg.norm(target_xy - self.path_target) > 0.08):
      self.planner.solve_from(pos)
      self.path = self.planner.path_to(target_xy)
      if self.path is None:
        # Object and place centers can lie inside furniture occupied cells.
        # Keep the observed goal as the navigation target and use the closest
        # reachable free cell only as the planner's final waypoint.
        candidates = []
        for radius in np.arange(0.05, 1.21, 0.05):
          for angle in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
            candidate = target_xy + radius * np.array([math.cos(angle), math.sin(angle)])
            if self.planner.free(candidate) and np.isfinite(self.planner.dist_to(candidate)):
              candidates.append((float(np.linalg.norm(candidate - target_xy)), candidate))
          if candidates:
            break
        if candidates:
          _, waypoint = min(candidates, key=lambda item: item[0])
          self.path = self.planner.path_to(waypoint)
      self.path_target = target_xy.copy()
    if self.path is None:
      return expert.nav_action()
    vx, wz, _ = planning.pure_pursuit(
        pos, state.yaw, self.path, lookahead=0.3, vmax=0.5)
    vx, vy, wz = self.expert._safe(vx, 0.0, wz)
    return expert.nav_action(vx=vx, vy=vy, wz=wz)

  def _plan_at_current_base(self, obs):
    direction, _, _, holding = self._observed_target(obs)
    backend, scene, state = self.env.backend, self.env.scene, self.env.backend.state
    base = np.array([state.x, state.y, state.yaw], np.float64)
    if holding:
      center = scene.place_point(scene.obj.half[2])
      R_w = kb.R_DEFAULT.copy()
      pre, goal, post = expert.place_targets(scene, center, R_w, state.obj_offset)
      h = expert.plan_at_base(backend, 'place', base, R_w, pre, goal, post,
                              True, state.obj_offset)
      kind, index = 'place', 0
    else:
      # The observed body-frame direction chooses the gripper approach axis;
      # no candidate/staging search is performed for either baseline.
      d = backend.R_body() @ np.asarray(direction, np.float64)
      d[2] = 0.0
      if np.linalg.norm(d) < 1e-6:
        d = np.array([math.cos(state.yaw), math.sin(state.yaw), 0.0])
      d /= np.linalg.norm(d)
      R_w = expert.grasp_frame(d, state.yaw)
      pre, goal, post = expert.pick_targets(state.obj_center, d, False)
      h = expert.plan_at_base(backend, 'pick', base, R_w, pre, goal, post)
      kind, index = 'pick', 0
    if h is None:
      h = float(state.h)
    return expert.ManipPlan(kind, base, h, R_w, pre, goal, post,
                            -R_w[:, 0][:2], index, way_w=())

  def __call__(self, obs):
    if self.ctrl is not None:
      action, status = self.ctrl.step(self.env.backend)
      if status == 'running':
        return action
      self.ctrl = None
      # A failed manipulation returns to NAV and retries at the same observed
      # threshold; the expert's failure recovery is intentionally not called.
      if status == 'done':
        return expert.nav_action()
    if self.should_switch(obs):
      self.ctrl = expert.ManipController(self._plan_at_current_base(obs))
      action, status = self.ctrl.step(self.env.backend)
      if status == 'running':
        return action
      self.ctrl = None
    return self._nav(obs)


def _summary(rows, split, scene_type, seed=0, affordance_source=None,
             scene_filter=None, seeds_considered=0, seeds_rejected=0):
  rows = list(rows)
  def vals(name):
    return np.asarray([getattr(row, name) for row in rows], dtype=float)
  metrics = {}
  for name in ('success', 'pick_success', 'n_false_triggers', 'n_skill_switches', 'steps'):
    x = vals(name)
    metrics[name] = {'mean': float(x.mean()) if len(x) else float('nan'),
                     'ci95': bootstrap_ci(x, seed=seed)}
  entry_counts = np.asarray([getattr(row, '_n_skill_entries', 0) for row in rows], dtype=float)
  rates = np.divide(vals('n_false_triggers'), entry_counts,
                    out=np.zeros(len(rows), dtype=float), where=entry_counts > 0)
  metrics['false_trigger_rate'] = {
      'mean': float(rates.mean()) if len(rates) else float('nan'),
      'ci95': bootstrap_ci(rates, seed=seed)}
  opportunities = int(entry_counts.sum())
  return {
      'split': split, 'scene_type': scene_type, 'count': len(rows),
      'affordance_source': affordance_source,
      'scene_filter': scene_filter,
      'seeds_considered': int(seeds_considered),
      'seeds_rejected': int(seeds_rejected),
      'metrics': metrics,
      'counts': {'success': int(sum(row.success for row in rows)),
                 'pick_success': int(sum(row.pick_success for row in rows)),
                 'false_triggers': int(sum(row.n_false_triggers for row in rows)),
                 'skill_switches': int(sum(row.n_skill_switches for row in rows)),
                 'steps': int(sum(row.steps for row in rows)),
                 'false_trigger_opportunities': int(opportunities)},
      'term_reason': dict(sorted(Counter(row.term_reason for row in rows).items())),
  }


def _write_json(path, value):
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _checkpoint_path(path):
  path = pathlib.Path(str(path))
  latest = path / 'latest'
  if latest.exists():
    return path / latest.read_text().strip()
  return path


def _episode(agent, env, seed, split, online_k, rng, baseline=None):
  _set_forced_seed(env, seed)
  reset = {key: np.zeros(space.shape, space.dtype) for key, space in env.act_space.items()}
  reset['reset'] = np.bool_(True)
  obs = env.step(reset)
  carry = agent.init_policy(1) if agent is not None else None
  if baseline is not None:
    baseline.reset(seed=int(rng.integers(2 ** 31)))
  skills, affs = [], []
  prev_skill = None
  path_length = 0.0
  switches = 0
  grasp_attempts = 0
  steps = 0
  min_observed_goal_distance = float('inf')
  switched_before_terminal = False
  labeler = labeler_lib.AffordanceLabeler(k_rollouts=online_k, seed=int(rng.integers(2 ** 31))) if online_k else None
  while True:
    if baseline is None:
      policy_obs = {key: np.asarray(value)[None] for key, value in obs.items()
                    if not key.startswith('log/')}
      carry, acts, _ = agent.policy(carry, policy_obs, mode='eval')
      action = _as_action(acts)
    else:
      action = baseline(obs)
      if not switched_before_terminal:
        min_observed_goal_distance = min(
            min_observed_goal_distance, baseline.observed_goal_distance(obs))
    action['reset'] = np.bool_(False)
    skill = int(action['skill'])
    skills.append(skill)
    entering = (prev_skill == int(I.Skill.NAV) and skill in (int(I.Skill.PICK), int(I.Skill.PLACE)))
    if entering and not switched_before_terminal:
      switched_before_terminal = True
    if entering:
      if labeler is not None:
        # rate() snapshots/restores the backend around its K rollout trials.
        state = env.backend.state
        valid_phase = ((skill == int(I.Skill.PICK) and not state.holding and not state.placed) or
                       (skill == int(I.Skill.PLACE) and state.holding and not state.placed))
        aff = float(labeler.rate(env, skill - 1)) if valid_phase else 0.0
      elif skill == int(I.Skill.PICK):
        aff = float(labeler_lib.aff_geom_pick(env.backend, env.scene))
      else:
        aff = float(labeler_lib.aff_geom_place(env.backend, env.scene))
    else:
      aff = 1.0
    affs.append(aff)
    switches += int(prev_skill is not None and skill != prev_skill)
    prev_skill = skill
    obs = env.step(action)
    steps += 1
    path_length += float(obs.get('log/base_disp', 0.0))
    grasp_attempts += int(round(float(obs.get('log/grasp_attempt', 0.0))))
    if obs['is_last']:
      term_id = int(obs['log/term_reason'])
      term_reason = I.TERM_REASONS[term_id] if 0 <= term_id < len(I.TERM_REASONS) else 'nan'
      row = I.EpisodeMetrics(
          split=split, scene_type=int(obs['log/scene_type']), seed=int(obs['log/seed']),
          success=bool(float(obs['log/success'])),
          pick_success=bool(float(obs['log/pick_success'])), steps=steps,
          path_length_m=float(path_length), n_skill_switches=switches,
          n_grasp_attempts=grasp_attempts,
          n_false_triggers=count_false_triggers(skills, affs),
          term_reason=term_reason)
      row._n_skill_entries = sum(
          prev == int(I.Skill.NAV) and cur in (int(I.Skill.PICK), int(I.Skill.PLACE))
          for prev, cur in zip(skills, skills[1:]))
      row._min_observed_goal_distance = (
          None if not np.isfinite(min_observed_goal_distance)
          else float(min_observed_goal_distance))
      row._switched_before_terminal = bool(switched_before_terminal)
      return row


def eval_legmm(make_agent, make_env, make_logger, args, config=None):
  """Evaluate sampled policy episodes and write EpisodeMetrics JSONL/summaries."""
  del make_logger
  if config is None:
    raise ValueError('eval_legmm requires the parsed configuration')
  if bool(config.agent.deliberation.enabled):
    raise NotImplementedError('LegMM deliberation is not implemented for eval_legmm yet')
  split = str(config.env.legmm.split)
  scene_types = parse_scene_types(getattr(config.eval, 'scene_types', ''))
  episodes = int(config.eval.episodes)
  online_k = int(config.eval.online_aff_rollouts)
  seeds, seed_info = select_seeds(
      split, scene_types, episodes, getattr(config.eval, 'scene_filter', 'auto'),
      config.env.legmm, return_info=True)
  logdir = pathlib.Path(str(config.logdir))
  logdir.mkdir(parents=True, exist_ok=True)
  rows_path = logdir / 'episodes.jsonl'
  policy_name = str(getattr(config.eval, 'policy', 'agent')).lower()
  if policy_name not in ('agent', 'b1', 'b2'):
    raise ValueError(f'unknown eval.policy: {policy_name!r}')
  agent = make_agent() if policy_name == 'agent' else None
  from_checkpoint = getattr(args, 'from_checkpoint', '')
  if from_checkpoint and agent is not None:
    def load_agent(data):
      agent.load(data, regex=getattr(args, 'from_checkpoint_regex', '') or None)
    elements.checkpoint.load(_checkpoint_path(from_checkpoint), {
        'agent': load_agent})
  env = make_env(0, split=split)
  baseline = (BaselinePolicy(
      policy_name, env, heading_deg=float(getattr(config.eval, 'baseline_heading_deg', 30.0)))
              if policy_name != 'agent' else None)
  rng = np.random.default_rng(int(config.seed))
  rows = []
  episode_seconds = []
  start = time.perf_counter()
  with rows_path.open('w') as stream:
    for seed in seeds:
      episode_start = time.perf_counter()
      row = _episode(agent, env, seed, split, online_k, rng, baseline=baseline)
      elapsed = time.perf_counter() - episode_start
      episode_seconds.append(float(elapsed))
      rows.append(row)
      row_json = dataclasses.asdict(row)
      row_json['affordance_source'] = 'online_labeler_rollouts' if online_k else 'aff_geom'
      row_json['min_observed_goal_distance'] = row._min_observed_goal_distance
      row_json['switched_before_terminal'] = row._switched_before_terminal
      row_json['below_switch_distance'] = bool(
          row._min_observed_goal_distance is not None and
          row._min_observed_goal_distance < (baseline.switch_distance if baseline else 0.75))
      stream.write(json.dumps(row_json, sort_keys=True) + '\n')
      stream.flush()
  env.close()
  groups = {}
  for row in rows:
    groups.setdefault(row.scene_type, []).append(row)
  for stype, group in groups.items():
    name = scenes.SceneType(stype).name
    _write_json(logdir / f'summary_{split}_{name}.json', _summary(
        group, split, name,
        affordance_source='online_labeler_rollouts' if online_k else 'aff_geom',
        **seed_info))
  overall = _summary(rows, split, 'overall',
                     affordance_source='online_labeler_rollouts' if online_k else 'aff_geom',
                     **seed_info)
  _write_json(logdir / f'summary_{split}_overall.json', overall)
  _write_json(logdir / 'summary.json', overall)
  _write_json(logdir / 'timing.json', {
      'episodes': len(episode_seconds), 'wall_seconds_per_episode': episode_seconds,
      'mean_wall_seconds_per_episode': float(np.mean(episode_seconds)) if episode_seconds else float('nan'),
  })
  for i, elapsed in enumerate(episode_seconds):
    print(f'eval_legmm episode={i} wall_seconds={elapsed:.6f}')
  print(f'eval_legmm episodes={len(rows)} wall_seconds={time.perf_counter() - start:.3f}')
  return rows

