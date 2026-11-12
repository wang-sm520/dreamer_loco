"""Evaluate a trained legmm_ag Dreamer checkpoint on the calibration test seeds (v4, brief W7/W9).

Runs the policy on `LegMMAgEnv` (split 'test', unfiltered scenes, forced seeds 20000..) and writes the same
statistics as calibration_v3/calibration.json (`unfiltered`, `filtered` = scene solvable with the 1 cm / 3 deg
margin, by shape / blocked / folded orientation, Wilson 95% intervals) plus a comparison with the three
calibration policies. The seed -> (alpha, solvable_nomargin) metadata is taken from calibration_episodes.jsonl
(scene metadata does not depend on the policy); shape / blocked / solvable are cross-checked against the env logs.

Action selection. dreamerv3.agent.Agent.policy always SAMPLES its action distributions and has no mode switch
(the `mode` argument is ignored). This tool therefore rebinds `dreamerv3.agent.sample` (module global read when the
policy is traced; agent.py itself is untouched) so that
  --nav   mode    continuous nav head -> Normal mean   (default; or `sample`)
  --trigger argmax discrete trigger head -> argmax      (or `sample`, the training-time behaviour; default)
Which choice was used is written to meta.action_selection.

  python tools/legmm_ag_eval.py --checkpoint ~/logdir/legmm/v4/pilot/ag_anneal_s0 --n-episodes 300 \
      --out ~/logdir/legmm/v4/pilot/eval_s0_nav-mode_trig-argmax.json
"""
import argparse
import collections
import json
import os
import pathlib
import sys
import time
from functools import partial as bind

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np  # noqa: E402

DEFAULT_CALIB = '~/logdir/legmm/v4/env/calibration_v3'


def parse_args(argv=None):
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', required=True, help='logdir (uses its ckpt/latest) or a checkpoint folder')
  p.add_argument('--configs', nargs='+', default=['legmm_ag'])
  p.add_argument('--n-episodes', type=int, default=700, help='test seeds 20000..20000+n-1 (>= 300 requested)')
  p.add_argument('--split', choices=['test', 'val'], default='test')
  p.add_argument('--seed-start', type=int, default=None)
  p.add_argument('--envs', type=int, default=8, help='parallel env worker processes (<= 8)')
  p.add_argument('--nav', choices=['mode', 'sample'], default='mode')
  p.add_argument('--trigger', choices=['argmax', 'sample'], default='argmax')
  p.add_argument('--platform', choices=['cuda', 'cpu'], default='cuda')
  p.add_argument('--prealloc', action='store_true', help='reserve the XLA memory fraction for GPU resource trials')
  p.add_argument('--calibration', default=DEFAULT_CALIB)
  p.add_argument('--out', required=True)
  p.add_argument('--max-steps-wall-s', type=float, default=6 * 3600, help='abort if the evaluation takes longer')
  return p.parse_known_args(argv)


def load_config(args, extra, outdir):
  import elements
  import ruamel.yaml as yaml
  raw = yaml.YAML(typ='safe').load((ROOT / 'dreamerv3' / 'configs.yaml').read_text())
  cfg = elements.Config(raw['defaults'])
  for name in args.configs:
    cfg = cfg.update(raw[name])
  cfg = elements.Flags(cfg).parse(list(extra))
  return cfg.update(
      script='eval_only', logdir=str(outdir),
      jax=dict(cfg.jax, platform=args.platform, prealloc=args.prealloc),
      env=dict(cfg.env, legmm_ag=dict(cfg.env.legmm_ag, split=args.split, scene_filter='none')))


def patch_action_selection(nav, trigger):
  """Rebind agent.sample: modes for the chosen heads. Returns a description."""
  import dreamerv3.agent as agent_mod
  import jax
  import ninjax as nj

  def pick(k, x):
    det = (nav == 'mode') if k == 'nav' else (trigger == 'argmax')
    return x.pred() if det else x.sample(nj.seed())

  def sample(xs):
    if isinstance(xs, dict):
      return {k: pick(k, x) for k, x in xs.items()}
    return jax.tree.map(lambda x: x.sample(nj.seed()), xs)

  agent_mod.sample = sample
  return dict(nav=nav, trigger=trigger,
              note='Agent.policy has no deterministic mode; dreamerv3.agent.sample rebound by this tool')


class SeedEnv:
  """Wraps a (normalised) legmm_ag env: every reset uses the next forced test seed of this worker."""

  def __init__(self, env, seeds):
    self.env, self.seeds, self.i = env, list(seeds), 0
    from embodied.envs.legmm.ag_env import LegMMAgEnv
    base = env
    while not isinstance(base, LegMMAgEnv):     # embodied wrappers forward attribute reads, not writes
      base = base.env
    self.base = base
    self.contract_seeds = set()
    self.contract_file = None

  obs_space = property(lambda s: s.env.obs_space)
  act_space = property(lambda s: s.env.act_space)

  def step(self, action):
    if action['reset'] or self.base._done:
      seed = int(self.seeds[self.i % len(self.seeds)])         # past the quota: repeat (ignored by the collector)
      self.i += 1
      self.base.forced_seed = seed
      self.base.rng = np.random.default_rng([seed, 0xE7A1])
    obs = self.env.step(action)
    if self.contract_file is not None and obs['is_last'] and self.base.scene.seed not in self.contract_seeds:
      self.contract_seeds.add(self.base.scene.seed)
      info = self.base.last_info
      res = info.get('skill')
      with open(self.contract_file, 'a') as stream:
        stream.write(json.dumps(dict(seed=self.base.scene.seed, contract=info.get('contract'),
                         skill_reason=res.reason if res else '', success=bool(res and res.success))) + '\n')
    return obs

  def close(self):
    self.env.close()


def make_worker_env(cfg, index, seeds):
  import dreamerv3.main as main
  env = SeedEnv(main.make_env(cfg, index, split=cfg.env.legmm_ag.split), seeds)
  if cfg.env.legmm_ag.contract == 'true_state':
    env.contract_file = str(pathlib.Path(cfg.logdir) / f'contract_worker_{index}.jsonl')
  return env


class Collector:
  """Per-episode accumulation from the driver's transitions (log/* keys are the env's own logs)."""

  def __init__(self, n_workers, term_names):
    self.acc = [None] * n_workers
    self.rows = {}
    self.term_names = term_names

  def __call__(self, tran, worker, **kw):
    if tran['is_first']:
      self.acc[worker] = collections.defaultdict(float, steps=0, first_trigger=None, min_dist=9.0, seed=None)
    a = self.acc[worker]
    if tran['is_first']:
      a['seed'] = int(tran['log/seed'])
      a['dist0'] = float(tran['log/shoulder_dist_true'])
      return
    a['steps'] += 1
    for k in ('triggered', 'invalid', 'collision', 'collision_start', 'gate_ok', 'gate_ok_true',
              'slid_out', 'skill_steps', 'skill_s', 'settle_s', 'trig_gate_mismatch'):
      a[k] += float(tran['log/' + k])
    a['min_dist'] = min(a['min_dist'], float(tran['log/shoulder_dist_true']))
    a['reward'] += float(tran['reward'])
    if float(tran['log/triggered']) and a['first_trigger'] is None:
      a['first_trigger'] = a['steps']
    if tran['is_last']:
      seed = a['seed']
      if seed in self.rows:                                    # worker ran past its quota
        return
      steps = max(a['steps'], 1)
      self.rows[seed] = dict(
          seed=seed, success=bool(float(tran['log/success'])),
          reason=self.term_names[int(tran['log/term_reason'])], steps=int(a['steps']),
          flat=bool(float(tran['log/flat'])), blocked=bool(float(tran['log/blocked'])),
          solvable_env=float(tran['log/solvable']),
          triggers=int(a['triggered']), invalid=int(a['invalid']), valid_triggers=int(a['triggered'] - a['invalid']),
          collision_steps=int(a['collision']), collision_starts=int(a['collision_start']),
          gate_ok_frac=a['gate_ok'] / steps, gate_ok_true_frac=a['gate_ok_true'] / steps,
          first_trigger=a['first_trigger'], min_shoulder_dist=a['min_dist'], start_shoulder_dist=a['dist0'],
          slid_out=int(a['slid_out']), trig_gate_mismatch=int(a['trig_gate_mismatch']),
          terminal=bool(tran['is_terminal']), return_=a['reward'])


def scene_seeds(args):
  from embodied.envs.legmm.ag_scenes import AG_SPLITS
  lo, hi = AG_SPLITS[args.split]
  start = lo if args.seed_start is None else args.seed_start
  if args.n_episodes <= 0 or start < lo or start + args.n_episodes > hi:
    raise ValueError(f'seeds outside {args.split}: [{lo}, {hi})')
  return list(range(start, start + args.n_episodes))


def scene_metadata(cfg, seeds):
  """Compute labels with the selected contract, independently of policy and old calibration."""
  from embodied.envs.legmm.ag_env import LegMMAgEnv
  from embodied.envs.legmm import ag_scenes, ag_skill
  ecfg = dict(cfg.env.legmm_ag)
  for key in ('split', 'backend', 'spaces_only', 'use_seed'):
    ecfg.pop(key, None)
  env = LegMMAgEnv(split=cfg.env.legmm_ag.split, **ecfg)
  meta = {}
  for seed in seeds:
    scene = ag_scenes.make_ag_scene(env.split, seed, env.cfg)
    env.scene = scene
    robust = ag_skill.solvable_scene(env._shadow(), scene, env.skill_cfg, margin=env.solvable_margin)
    nomargin = ag_skill.solvable_scene(env._shadow(), scene, env.skill_cfg, margin=0.)
    m = scene.meta
    meta[seed] = dict(seed=seed, shape=m['shape'], blocked=m['blocked'], alpha=m['alpha_fold_deg'],
                      solvable=robust, solvable_margin=robust, solvable_nomargin=nomargin)
  env.close()
  return meta


def load_calibration(path, seeds):
  path = pathlib.Path(path).expanduser()
  meta = {}
  for line in open(path / 'calibration_episodes.jsonl'):
    r = json.loads(line)
    if r['policy'] == 'privileged':
      meta[r['seed']] = r
  missing = [s for s in seeds if s not in meta]
  assert not missing, f'seeds not in calibration: {missing[:5]}'
  return meta, json.load(open(path / 'calibration.json'))


def summarize(rows, calib_meta, calib, calib_dir):
  import legmm_ag_calibrate as cal
  full = []
  mismatch = collections.Counter()
  for r in rows:
    m = calib_meta[r['seed']]
    mismatch['shape'] += int((m['shape'] == 'flat') != r['flat'])
    mismatch['blocked'] += int(bool(m['blocked']) != r['blocked'])
    mismatch['solvable'] += int(float(m['solvable']) != r['solvable_env'])
    full.append(dict(r, shape=m['shape'], blocked=bool(m['blocked']), alpha=m['alpha'],
                     solvable=bool(m['solvable']), solvable_margin=bool(m['solvable_margin']),
                     solvable_nomargin=bool(m['solvable_nomargin'])))
  n = len(full)
  filt = [r for r in full if r['solvable']]
  mean = lambda xs: round(float(np.mean(xs)), 4) if len(xs) else None
  def extra(rs):
    k = max(len(rs), 1)
    return dict(
        contract_violation_rate=sum(r.get('contract_violation', False) for r in rs) / k,
        contract_counts={name: sum(bool(r.get('contract')) and not r['contract'][name + '_ok'] for r in rs)
                         for name in ('dist', 'bearing', 'height', 'clearance')},
        n=len(rs), mean_steps=mean([r['steps'] for r in rs]),
        timeout_rate=round(sum(r['reason'] == 'timeout' for r in rs) / k, 4),
        reason_counts=dict(collections.Counter(r['reason'] for r in rs)),
        triggered_episode_rate=round(sum(r['triggers'] > 0 for r in rs) / k, 4),
        triggers_per_episode=mean([r['triggers'] for r in rs]),
        invalid_triggers_per_episode=mean([r['invalid'] for r in rs]),
        valid_triggers_per_episode=mean([r['valid_triggers'] for r in rs]),
        collision_steps_per_episode=mean([r['collision_steps'] for r in rs]),
        collision_starts_per_episode=mean([r['collision_starts'] for r in rs]),
        gate_ok_step_fraction=mean([r['gate_ok_frac'] for r in rs]),
        gate_ok_true_step_fraction=mean([r['gate_ok_true_frac'] for r in rs]),
        episodes_ever_gate_ok_true=round(sum(r['gate_ok_true_frac'] > 0 for r in rs) / k, 4),
        mean_min_shoulder_dist_true=mean([r['min_shoulder_dist'] for r in rs]),
        mean_start_shoulder_dist_true=mean([r['start_shoulder_dist'] for r in rs]),
        mean_return=mean([r['return_'] for r in rs]),
        trigger_gate_mismatch=dict(mismatch=sum(r['trig_gate_mismatch'] for r in rs),
                                   triggers=sum(r['triggers'] for r in rs)))
  out = dict(
      policy_eval=dict(unfiltered=cal.tables(full), filtered=cal.tables(filt),
                       filtered_nomargin=cal.tables([r for r in full if r['solvable_nomargin']]),
                       extra_unfiltered=extra(full), extra_filtered=extra(filt)),
      env_vs_calibration_metadata_mismatch=dict(mismatch, of=n))
  # Comparison with the calibration policies on the SAME seeds (calibration policies were run on all 700).
  if calib is None:
    return out, full
  seeds = {r['seed'] for r in full}
  comp = {}
  rate = lambda rs: (sum(r['success'] for r in rs), len(rs))
  cal_eps = collections.defaultdict(dict)
  for line in open(pathlib.Path(calib_dir).expanduser() / 'calibration_episodes.jsonl'):
    r = json.loads(line)
    if r['seed'] in seeds:
      cal_eps[r['policy']][r['seed']] = r
  for name, eps in cal_eps.items():
    for sel_name, sel in (('unfiltered', lambda r: True), ('filtered', lambda r: r['solvable'])):
      ka, na = rate([r for r in full if sel(r)])
      kb, nb = rate([r for r in eps.values() if sel(r)])
      comp[f'agent_vs_{name}_{sel_name}'] = dict(
          n=na, agent=round(ka / na, 4) if na else None, baseline=round(kb / nb, 4) if nb else None,
          diff=round(ka / na - kb / nb, 4) if na else None, diff_ci95_newcombe_unpaired=cal.newcombe_diff(ka, na, kb, nb))
  out['comparison_same_seeds'] = comp
  return out, full


def main(argv=None):
  args, extra = parse_args(argv)
  assert args.envs <= 8, 'at most 8 CPU workers'
  out = pathlib.Path(args.out).expanduser()
  out.parent.mkdir(parents=True, exist_ok=True)
  seeds = scene_seeds(args)
  from embodied.envs.legmm import ag_scenes, interfaces as I


  import elements
  import embodied
  import jax
  import dreamerv3.main as main_mod
  selection = patch_action_selection(args.nav, args.trigger)
  cfg = load_config(args, extra, out.parent)
  if args.split == 'val' or cfg.env.legmm_ag.contract == 'true_state':
    calib_meta, calib = scene_metadata(cfg, seeds), None
  else:
    calib_meta, calib = load_calibration(args.calibration, seeds)
  out.with_name('scene_metadata.json').write_text(json.dumps(calib_meta, indent=1))
  agent_cfg = cfg.update(env=dict(cfg.env, legmm_ag=dict(cfg.env.legmm_ag, spaces_only=True)))
  agent = main_mod.make_agent(agent_cfg)      # spaces from interfaces: no simulator in the main process
  cp = pathlib.Path(args.checkpoint).expanduser()
  if (cp / 'ckpt').exists():
    cp = cp / 'ckpt'
  if (cp / 'latest').exists():
    cp = cp / (cp / 'latest').read_text().strip()
  ck = elements.Checkpoint()
  ck.agent = agent
  ck.load(str(cp), keys=['agent'])
  print('loaded', cp, flush=True)

  W = min(args.envs, len(seeds))
  fns = [bind(make_worker_env, cfg, i, seeds[i::W]) for i in range(W)]
  driver = embodied.Driver(fns, parallel=True, mode='eval')
  col = Collector(W, I.AG_TERM_REASONS)
  driver.on_step(col)
  driver.reset(agent.init_policy)
  t0, last = time.time(), 0
  nsteps = 0
  while len(col.rows) < len(seeds):
    nsteps, _ = driver._step(agent.policy, nsteps, 0)
    if len(col.rows) - last >= 25:
      last = len(col.rows)
      print(f'{last}/{len(seeds)} episodes {time.time() - t0:.0f}s', flush=True)
    if time.time() - t0 > args.max_steps_wall_s:
      raise RuntimeError('wall-clock limit')
  driver.close()
  rows = [col.rows[s] for s in seeds]
  contracts = {}
  for path in out.parent.glob('contract_worker_*.jsonl'):
    for line in path.read_text().splitlines():
      r = json.loads(line)
      contracts[r['seed']] = r
  for row in rows:
    r = contracts.get(row['seed'], {})
    row['contract_violation'] = r.get('skill_reason') == 'contract_violation'
    row['contract'] = r.get('contract')
  summary, full = summarize(rows, calib_meta, calib, args.calibration)
  if calib is not None:
    summary['calibration_baselines_700_scenes'] = {
      k: dict(unfiltered=v['unfiltered']['overall'], filtered=v['filtered']['overall'])
      for k, v in calib['policies'].items()}
  summary['meta'] = dict(
      version=I.AG_VERSION, split=args.split, seeds=[seeds[0], seeds[-1]], n_episodes=len(seeds),
      checkpoint=str(cp), checkpoint_counters=None, configs=args.configs, extra_flags=list(extra),
      action_selection=selection, platform=args.platform, wall_s=round(time.time() - t0, 1),
      env_steps=int(nsteps),
      note='unfiltered = all selected split seeds; filtered = scenes solvable with the 1 cm / 3 deg margin (env meta.solvable)')
  out.write_text(json.dumps(summary, indent=1))
  with open(out.with_name(out.stem + '_episodes.jsonl'), 'w') as f:
    for r in full:
      f.write(json.dumps(r) + '\n')
  print('written', out, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
  main()
