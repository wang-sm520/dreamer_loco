"""Small, CPU-friendly LegMM policy evaluator.

The policy API's ``mode`` argument is passed as ``mode='eval'`` below.  The
current :class:`dreamerv3.agent.Agent` samples its action distributions in
both modes (it does not take the mode), so this evaluator reports sampled
policy behavior rather than an argmax policy.
"""

import dataclasses
import json
import pathlib
import time
from collections import Counter

import numpy as np
import elements

from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import expert
from embodied.envs.legmm import labeler as labeler_lib
from embodied.envs.legmm import scenes


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


def _episode(agent, env, seed, split, online_k, rng):
  _set_forced_seed(env, seed)
  reset = {key: np.zeros(space.shape, space.dtype) for key, space in env.act_space.items()}
  reset['reset'] = np.bool_(True)
  obs = env.step(reset)
  carry = agent.init_policy(1)
  skills, affs = [], []
  prev_skill = None
  path_length = 0.0
  switches = 0
  grasp_attempts = 0
  steps = 0
  labeler = labeler_lib.AffordanceLabeler(k_rollouts=online_k, seed=int(rng.integers(2 ** 31))) if online_k else None
  while True:
    policy_obs = {key: np.asarray(value)[None] for key, value in obs.items()
                  if not key.startswith('log/')}
    carry, acts, _ = agent.policy(carry, policy_obs, mode='eval')
    action = _as_action(acts)
    action['reset'] = np.bool_(False)
    skill = int(action['skill'])
    skills.append(skill)
    entering = (prev_skill == int(I.Skill.NAV) and skill in (int(I.Skill.PICK), int(I.Skill.PLACE)))
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
  agent = make_agent()
  from_checkpoint = getattr(args, 'from_checkpoint', '')
  if from_checkpoint:
    def load_agent(data):
      # WM checkpoints created before the return-normalizer state was added
      # legitimately lack these two runtime scalars. Keep the fresh values
      # while retaining strict shape/key checks for model parameters.
      params = dict(data['params'])
      for key in set(agent.params) - set(params):
        if key.startswith('retnorm/'):
          params[key] = np.zeros_like(np.asarray(agent.params[key]))
      agent.load({**data, 'params': params},
                 regex=getattr(args, 'from_checkpoint_regex', '') or None)
    elements.checkpoint.load(_checkpoint_path(from_checkpoint), {
        'agent': load_agent})
  env = make_env(0, split=split)
  rng = np.random.default_rng(int(config.seed))
  rows = []
  episode_seconds = []
  start = time.perf_counter()
  with rows_path.open('w') as stream:
    for seed in seeds:
      episode_start = time.perf_counter()
      row = _episode(agent, env, seed, split, online_k, rng)
      elapsed = time.perf_counter() - episode_start
      episode_seconds.append(float(elapsed))
      rows.append(row)
      row_json = dataclasses.asdict(row)
      row_json['affordance_source'] = 'online_labeler_rollouts' if online_k else 'aff_geom'
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

