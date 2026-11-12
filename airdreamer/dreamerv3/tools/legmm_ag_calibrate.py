"""v4 calibration without learning (brief W7): scripted policies on the real legmm_ag env + surrogate skill.

  nearest_edge       drive to the object's nearest table edge ignoring orientation, stand head-on, trigger
  nearest_free_edge  the same, but knows which edge is blocked (first free edge with a feasible head-on stance)
  privileged         true pose: best feasible stance of the fixed-base skill (planner), then trigger

All three navigate with embodied.envs.legmm.planning (GridPlanner + pure_pursuit). Test scenes are
unfiltered; each episode records whether its scene is solvable (some stance with a skill plan), and the
tables report the unfiltered set, the solvable ("filtered") subset and strata. The surrogate's success is
not a real grasp success rate.

  python tools/legmm_ag_calibrate.py --n-scenes 700 --workers 28 --out ~/logdir/legmm/v4/env/calibration.json
"""
import argparse
import collections
import json
import math
import multiprocessing
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied.envs.legmm import ag_contract as contract
from embodied.envs.legmm import ag_env  # noqa: E402
from embodied.envs.legmm import ag_scenes  # noqa: E402
from embodied.envs.legmm import ag_skill as sk  # noqa: E402
from embodied.envs.legmm import interfaces as I  # noqa: E402
from embodied.envs.legmm import planning  # noqa: E402
from embodied.envs.legmm import scenes as sc  # noqa: E402

POLICIES = ('nearest_edge', 'nearest_free_edge', 'privileged')
# Shoulder-object distances to try, interior first (a stance 5 mm outside the gate cannot be used).
RHOS = (0.575, 0.55, 0.60, 0.525, 0.625, 0.50, 0.65, 0.475, 0.675, 0.46, 0.69)
STAGE_BACK = (0.4, 0.25, 0.6)
ALPHA_BUCKETS = ((0, 30), (30, 60), (60, 90.0001))


def load_cfg():
  text = (ROOT / 'dreamerv3/configs.yaml').read_text()
  cfg = dict(yaml.safe_load(text)['defaults']['env']['legmm_ag'])
  cfg['labels'] = dict(cfg['labels'], rate=0.0)          # no counterfactual labels needed here
  for k in ('use_seed', 'split', 'backend', 'spaces_only'):
    cfg.pop(k, None)
  return cfg


def wilson(k, n, z=1.959964):
  if n == 0:
    return [None, None]
  p = k / n
  den = 1 + z * z / n
  centre = (p + z * z / (2 * n)) / den
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(k, n):
  return dict(k=int(k), n=int(n), rate=round(k / n, 4) if n else None, ci95=wilson(k, n))


def newcombe_diff(k1, n1, k2, n2):
  """95% CI of p1 - p2 (Newcombe hybrid score method)."""
  if n1 == 0 or n2 == 0:
    return [None, None]
  l1, u1 = wilson(k1, n1)
  l2, u2 = wilson(k2, n2)
  p1, p2 = k1 / n1, k2 / n2
  lo = p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
  hi = p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
  return [round(lo, 4), round(hi, 4)]


# ------------------------------------------------------------------------------------------ policies

def nav_action(vx=0.0, vy=0.0, wz=0.0, trigger=0):
  vec = I.unscale_action({'vx': vx, 'vy': vy, 'wz': wz}, I.AG_NAV_LAYOUT, I.AG_NAV_DIM)
  return {'reset': False, 'nav': vec.astype(np.float32), 'trigger': np.int32(trigger)}


def base_from_shoulder(env, shoulder_xy, yaw):
  dx, dy = env.backend.shoulder_b[:2]
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([shoulder_xy[0] - c * dx + s * dy, shoulder_xy[1] - s * dx - c * dy, yaw])


def edge_distances(scene):
  """Distance from the object center to each table edge (edge index as in ag_scenes.EDGE_NORMALS)."""
  tc = np.asarray(scene.meta['table_center'])
  sx, sy, _ = scene.meta['table']
  rel = scene.obj.center[:2] - tc
  out = []
  for n in ag_scenes.EDGE_NORMALS:
    half = sy / 2 if n[0] == 0 else sx / 2
    out.append(float(half - rel @ n))
  return out


class StancePolicy:

  def __init__(self, kind):
    assert kind in POLICIES
    self.kind = kind

  # ---- choosing the stance
  def head_on(self, env, edge):
    n = ag_scenes.EDGE_NORMALS[edge]
    yaw = math.atan2(-n[1], -n[0])
    out = []
    rhos = contract.radius_grid(env.scene.obj.support_z) if env.contract == 'true_state' else RHOS
    for rho in rhos:
      base = base_from_shoulder(env, env.scene.obj.center[:2] + rho * n, yaw)
      if sk.inside_room(env.backend, base) and sc.footprint_free(
          env.scene.boxes, base[0], base[1], base[2], sk.BODY_MARGIN):
        out.append(base)
    return out

  def oblique(self, env, edge):
    """Fallback of nearest_free_edge when no head-on stance fits (the far edges are >= 0.35 m from the
    object, a head-on stance would put the body into the table): facing the object from the free edge's
    side, as close to head-on as the footprint allows. Object orientation is still ignored."""
    n = ag_scenes.EDGE_NORMALS[edge]
    az0 = math.degrees(math.atan2(n[1], n[0]))
    sh = env._shadow()
    sh.reset(env.scene, np.random.default_rng(0))
    out = []
    for off in sorted(range(-75, 76, 5), key=abs):
      for bearing in (0, 15, -15, 30, -30, 45, -45):   # body turned up to 45 deg off the object (gate limit)
        rhos = contract.radius_grid(env.scene.obj.support_z) if env.contract == 'true_state' else RHOS
        for rho in rhos:
          base = sk.stance_pose(sh, env.scene.obj.center[:2], rho, az0 + off, bearing)
          if sk.inside_room(sh, base) and sc.footprint_free(
              env.scene.boxes, base[0], base[1], base[2], sk.BODY_MARGIN):
            out.append(base)
    return out

  def candidates(self, env):
    scene = env.scene
    d = edge_distances(scene)
    near = int(scene.meta['edge'])
    if self.kind == 'nearest_edge':
      return [self.head_on(env, near)]
    if self.kind == 'nearest_free_edge':
      blocked = near if scene.meta['blocked'] else None
      edges = [e for e in np.argsort(d) if e != blocked]
      return [self.head_on(env, int(e)) or self.oblique(env, int(e)) for e in edges]
    sh = env._shadow()
    sh.reset(scene, np.random.default_rng(0))
    fs = sk.feasible_stances(sh, scene, env.skill_cfg)
    # Grid stances on the envelope boundary (rho 0.70, bearing 45) cannot be reached to the millimetre:
    # also try variants moved inside, re-verified with the skill planner, and prefer the most interior one.
    out = []
    for f in fs:
      for d_rho in (0.0, 0.01, 0.02):
        for d_b in (0, 3, 6):
          rho = f['rho'] - d_rho if f['rho'] > 0.575 else f['rho'] + d_rho
          bearing = f['bearing'] - math.copysign(d_b, f['bearing']) if f['bearing'] else 0
          base = sk.stance_pose(sh, scene.obj.center[:2], rho, f['az'], bearing)
          if not sc.footprint_free(sh._static, base[0], base[1], base[2], sk.BODY_MARGIN):
            continue
          if sk.plan_skill(sh, env.skill_cfg, base=base, h=sh.state.h)[0] is None:
            continue
          if env.contract == 'true_state' and not contract.report(sh, base, env.gate_cfg)['ok']:
            continue
          hi = contract.reach_limit(scene.obj.support_z) if env.contract == 'true_state' else .70
          margin = min(rho - 0.45, hi - rho, (45 - abs(bearing)) / 100.0)
          out.append((-min(margin, 0.02), abs(rho - 0.575) + 0.004 * abs(bearing), base))
    robust = [o for o in out if o[0] <= -0.0095]       # >= ~1 cm inside the gate (margin capped at 2 cm)
    out = sorted(robust or out, key=lambda o: o[:2])
    return [[o[2] for o in out]]

  # ---- navigation plan
  def _line_free(self, env, a, b, yaw):
    n = max(2, int(np.linalg.norm(b - a) / 0.01) + 1)       # 1 cm steps: narrow gaps must not be skipped
    return all(sc.footprint_free(env.scene.boxes, *(a * (1 - t) + b * t), yaw, 0.005)
               for t in np.linspace(0, 1, n))

  def plan_to(self, env, base, planners):
    yaw = base[2]
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    for planner in planners:
      for back in STAGE_BACK:
        st = base[:2] - back * fwd
        if planner.free(st) and self._line_free(env, st, base[:2], yaw):
          path = planner.path_to(st)
          if path is not None and len(path) > 1:     # the body (not a point) must also fit at the end of the path
            lead = path[-1] - path[-2]
            if not sc.footprint_free(env.scene.boxes, st[0], st[1], math.atan2(lead[1], lead[0]), 0.03):
              continue
          if path is not None:
            return path, st, planner.dist_to(st) + back
    return None

  def reset(self, env):
    s = env.backend.state
    start = np.array([s.x, s.y])
    planners = [planning.GridPlanner(env.scene, 0.45), planning.GridPlanner(env.scene, sc.NAV_INFLATE),
                planning.GridPlanner(env.scene, 0.2)]      # last resort next to the table / blocker
    for p in planners:
      p.solve_from(start)
    self.mode, self.tries, self.timer = 'idle', 0, 0
    self.back, self.unstick, self.hist = 0, 0, []
    self.stance, self.path, self.found = None, None, False
    for group in self.candidates(env):
      best = None
      for base in (group[:120] if self.kind == 'privileged' else group):
        plan = self.plan_to(env, base, planners)
        if plan is not None and (best is None or plan[2] < best[0][2]):
          best = (plan, base)
        if plan is not None and self.kind != 'privileged':
          break
      if best is not None:
        (self.path, self.stage, _), self.stance = best
        self.found = True
        self.mode = 'nav'
        return
    # No reachable stance: drive to the reachable point closest to the nominal head-on stance.
    near = int(env.scene.meta['edge'])
    n = ag_scenes.EDGE_NORMALS[near]
    target = env.scene.obj.center[:2] + 0.6 * n
    p = planners[1]
    dist = p._dist.reshape(p.shape)
    ii, jj = np.where(np.isfinite(dist))
    cells = np.stack([ii, jj], 1)
    world = np.array([p.world(c) for c in cells[::5]])
    k = int(np.argmin(np.linalg.norm(world - target, axis=1)))
    path = p.path_to(world[k])
    if path is not None:
      self.path, self.stage = path, world[k]
      self.stance = np.array([*world[k], math.atan2(-n[1], -n[0])])
      self.mode = 'nav'

  # ---- control
  def _safe(self, env, vx, vy, wz, horizon=0.3):
    s = env.backend.state
    for cand in ((vx, vy, wz), (0.0, 0.0, wz), (-0.15, 0.0, 0.0), (0.0, 0.0, 0.0)):
      cvx, cvy, cwz = cand
      ok = True
      for frac in (0.5, 1.0):
        t = horizon * frac
        yaw = s.yaw + cwz * t
        c, sn = math.cos(s.yaw + 0.5 * cwz * t), math.sin(s.yaw + 0.5 * cwz * t)
        x = s.x + (c * cvx - sn * cvy) * t
        y = s.y + (sn * cvx + c * cvy) * t
        if not sc.footprint_free(env.scene.boxes, x, y, yaw, 0.005):
          ok = False
          break
      if ok:
        return cand
    return (0.0, 0.0, 0.0)

  def act(self, env, obs):
    s = env.backend.state
    pos = np.array([s.x, s.y])
    self.timer += 1
    if self.mode == 'idle' or self.stance is None:
      return nav_action()
    if self.back > 0:                                   # unstick: reverse a few steps
      self.back -= 1
      return nav_action(vx=self._safe(env, -0.25, 0.0, 0.0)[0])
    self.hist.append(pos.copy())
    if self.mode in ('nav', 'line') and len(self.hist) > 20 and np.linalg.norm(self.hist[-1] - self.hist[-20]) < 0.02 \
        and self.unstick < 6:
      self.unstick += 1
      self.back = 5
      self.hist.clear()
    if self.mode == 'nav':
      vx, wz, rem = planning.pure_pursuit(pos, s.yaw, self.path, lookahead=0.3, vmax=0.5)
      if rem < 0.1:
        self.mode = 'align'
      else:
        vx, _, wz = self._safe(env, vx, 0.0, wz)
        return nav_action(vx=vx, wz=wz)
    if self.mode == 'align':
      err = planning.wrap(self.stance[2] - s.yaw)
      if abs(err) < 0.08:
        self.mode = 'line'
      else:
        _, _, wz = self._safe(env, 0.0, 0.0, float(np.clip(2.0 * err, -1, 1)))
        return nav_action(wz=wz)
    if self.mode == 'line':
      c, sn = math.cos(s.yaw), math.sin(s.yaw)
      d = self.stance[:2] - pos
      fx, fy = c * d[0] + sn * d[1], -sn * d[0] + c * d[1]
      err = planning.wrap(self.stance[2] - s.yaw)
      if math.hypot(fx, fy) < 0.01 and abs(err) < 0.02:
        self.mode = 'stop'
      else:
        vx, vy, wz = self._safe(env, float(np.clip(1.2 * fx, -0.3, 0.3)), float(np.clip(1.2 * fy, -0.2, 0.2)),
                                float(np.clip(2.0 * err, -0.6, 0.6)))
        return nav_action(vx=vx, vy=vy, wz=wz)
    if self.mode == 'stop':
      if env.contract == 'true_state':
        # Check the predicted settled pose too: trigger safety uses an estimate.
        sh = env._shadow()
        import copy
        sh.state = copy.deepcopy(env.backend.state)
        env._settle(sh)
        if not contract.report(sh, cfg=env.gate_cfg)['ok']:
          self.mode = 'line'
          return nav_action()
      if obs.get('log/invalid', 0.0) > 0 and self.timer > 1:
        self.tries += 1
        self.mode = 'line' if self.tries <= 20 else 'idle'
        return nav_action()
      return nav_action(trigger=1)
    return nav_action()


# ------------------------------------------------------------------------------------------ episodes

_ENV = {}


def get_env(cfg):
  if 'env' not in _ENV:
    _ENV['env'] = ag_env.LegMMAgEnv('ag', split='test', seed=0, **cfg)
  return _ENV['env']


def run_episode(job):
  kind, seed, cfg = job
  env = get_env(cfg)
  t0 = time.perf_counter()
  env.forced_seed = int(seed)
  obs = env.step({'reset': True, 'nav': np.zeros(3, np.float32), 'trigger': np.int32(0)})
  env.rng = np.random.default_rng([seed, POLICIES.index(kind)])
  policy = StancePolicy(kind)
  policy.reset(env)
  steps, invalid, collisions, path = 0, 0, 0, 0.0
  skill_steps = skill_s = settle_s = 0.0
  first_trigger = None
  trig_mismatch = trig_total = mismatch_steps = 0
  while True:
    obs = env.step(policy.act(env, obs))
    steps += 1
    trig_total += int(obs['log/triggered'])
    trig_mismatch += int(obs['log/trig_gate_mismatch'])
    mismatch_steps += int(obs['log/gate_mismatch'])
    invalid += int(obs['log/invalid'])
    collisions += int(obs['log/collision'])
    path += float(obs['log/base_disp'])
    if obs['log/triggered'] and first_trigger is None:
      first_trigger = steps
    if obs['is_last']:
      break
  m = env.scene.meta
  info = env.last_info
  return dict(
      policy=kind, seed=int(seed), shape=m['shape'], blocked=bool(m['blocked']), blocker=m['blocker'],
      solvable=bool(m['solvable']), solvable_margin=bool(sk.solvable_scene(env._shadow(), env.scene, env.skill_cfg, margin=0.01)),
      solvable_nomargin=bool(sk.solvable_scene(env._shadow(), env.scene, env.skill_cfg, margin=0.0)),
      trig_mismatch=int(trig_mismatch), trig_total=int(trig_total), gate_mismatch_steps=int(mismatch_steps), alpha=round(m['alpha_fold_deg'], 2), edge_dist=round(m['edge_dist'], 3),
      success=bool(obs['log/success']), reason=I.AG_TERM_REASONS[int(obs['log/term_reason'])],
      skill_reason=(info['skill'].reason if info.get('skill') is not None else ''),
      contract=info.get('contract'),
      steps=steps, invalid=invalid, collisions=collisions, path_m=round(path, 3), found_stance=policy.found,
      first_trigger=first_trigger, settle_s=float(obs['log/settle_s']), skill_steps=float(obs['log/skill_steps']),
      skill_s=float(obs['log/skill_s']), skill_wall_s=float(obs['log/skill_wall_s']),
      wall_s=round(time.perf_counter() - t0, 3))


# ------------------------------------------------------------------------------------------ statistics

def bucket_of(alpha):
  for lo, hi in ALPHA_BUCKETS:
    if lo <= alpha < hi:
      return f'{lo}-{int(hi)}'
  return f'{ALPHA_BUCKETS[-1][0]}-90'


def tables(rows):
  """Success rate (Wilson 95%) overall and stratified by shape, blocked, folded orientation, solvable."""
  def group(keyfn, sel=lambda r: True):
    g = collections.defaultdict(list)
    for r in rows:
      if sel(r):
        g[keyfn(r)].append(r['success'])
    return {str(k): rate(sum(v), len(v)) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}
  flat = lambda r: r['shape'] == 'flat'
  return dict(
      overall=rate(sum(r['success'] for r in rows), len(rows)),
      by_shape=group(lambda r: r['shape']),
      by_blocked=group(lambda r: 'blocked' if r['blocked'] else 'free'),
      by_shape_blocked=group(lambda r: (r['shape'], 'blocked' if r['blocked'] else 'free')),
      by_alpha_folded_flat=group(lambda r: bucket_of(r['alpha']), flat),
      by_alpha_folded_flat_blocked=group(lambda r: (bucket_of(r['alpha']), 'blocked' if r['blocked'] else 'free'), flat),
      by_solvable=group(lambda r: 'solvable' if r['solvable'] else 'unsolvable'),
      by_robust_solvable_1cm=group(lambda r: 'robust' if r['solvable_margin'] else ('knife_edge' if r['solvable_nomargin'] else 'unsolvable')))


def summarize(results, n_scenes):
  rows_by = collections.defaultdict(list)
  for r in results:
    rows_by[r['policy']].append(r)
  out = dict(policies={})
  for kind in POLICIES:
    rows = sorted(rows_by[kind], key=lambda r: r['seed'])
    filt = [r for r in rows if r['solvable']]
    n = max(len(rows), 1)
    out['policies'][kind] = dict(
        unfiltered=tables(rows), filtered=tables(filt), filtered_nomargin=tables([r for r in rows if r['solvable_nomargin']]),
        extra=dict(
            mean_steps=round(float(np.mean([r['steps'] for r in rows])), 1),
            timeout_rate=round(sum(r['reason'] == 'timeout' for r in rows) / n, 4),
            no_stance_found_rate=round(sum(not r['found_stance'] for r in rows) / n, 4),
            invalid_triggers_per_episode=round(float(np.mean([r['invalid'] for r in rows])), 3),
            trigger_gate_mismatch=dict(mismatch=sum(r['trig_mismatch'] for r in rows), triggers=sum(r['trig_total'] for r in rows)),
            step_gate_mismatch_rate=round(sum(r['gate_mismatch_steps'] for r in rows) / max(sum(r['steps'] for r in rows), 1), 5),
            collision_steps_per_episode=round(float(np.mean([r['collisions'] for r in rows])), 3),
            skill_fail_reasons=dict(collections.Counter(r['skill_reason'] for r in rows if r['reason'] == 'skill_fail')),
            mean_path_m=round(float(np.mean([r['path_m'] for r in rows])), 2)))
  # Scene statistics (identical scenes for every policy).
  base = sorted(rows_by[POLICIES[0]], key=lambda r: r['seed'])
  strata = collections.defaultdict(lambda: [0, 0, 0])
  for r in base:
    for key in (('all',), (r['shape'],), ('blocked' if r['blocked'] else 'free',),
                (r['shape'], 'blocked' if r['blocked'] else 'free'),
                (r['shape'], 'blocked' if r['blocked'] else 'free', r['blocker'])):
      strata['/'.join(key)][0] += 1
      strata['/'.join(key)][1] += int(r['solvable'])
      strata['/'.join(key)][2] += int(r['solvable_nomargin'])
  out['scene_solvability'] = {k: dict(n=v[0], solvable=v[1], rate=round(v[1] / v[0], 4), ci95=wilson(v[1], v[0]),
                                   solvable_without_margin=v[2], rate_without_margin=round(v[2] / v[0], 4))
                              for k, v in sorted(strata.items())}
  # Pairwise differences on the unfiltered and the solvable set.
  comp = {}
  for a, b in (('nearest_free_edge', 'privileged'), ('nearest_edge', 'nearest_free_edge'), ('nearest_edge', 'privileged')):
    for sel_name, sel in (('unfiltered', lambda r: True), ('filtered', lambda r: r['solvable'])):
      ra = {r['seed']: r for r in rows_by[a] if sel(r)}
      rb = {r['seed']: r for r in rows_by[b] if sel(r)}
      ka, kb_, n = sum(r['success'] for r in ra.values()), sum(r['success'] for r in rb.values()), len(ra)
      comp[f'{a}_minus_{b}_{sel_name}'] = dict(
          n=n, a=round(ka / n, 4) if n else None, b=round(kb_ / n, 4) if n else None,
          diff=round((ka - kb_) / n, 4) if n else None, diff_ci95_newcombe=newcombe_diff(ka, n, kb_, n))
  out['comparison'] = comp
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--contract', choices=['legacy', 'true_state'], default='legacy')
  parser.add_argument('--n-scenes', type=int, default=700, help='test seeds 20000.. (unfiltered)')
  parser.add_argument('--workers', type=int, default=28)
  parser.add_argument('--out', default='~/logdir/legmm/v4/env/calibration.json')
  parser.add_argument('--policies', default=','.join(POLICIES))
  args = parser.parse_args()
  out = pathlib.Path(args.out).expanduser()
  out.parent.mkdir(parents=True, exist_ok=True)
  cfg = load_cfg()
  cfg['contract'] = args.contract
  lo = ag_scenes.AG_SPLITS['test'][0]
  seeds = list(range(lo, lo + args.n_scenes))
  assert seeds[-1] < ag_scenes.AG_SPLITS['test'][1]
  kinds = args.policies.split(',')
  jobs = [(k, s, cfg) for s in seeds for k in kinds]
  t0 = time.time()
  results = []
  with multiprocessing.Pool(args.workers) as pool:
    for i, r in enumerate(pool.imap_unordered(run_episode, jobs, chunksize=4)):
      results.append(r)
      if (i + 1) % 200 == 0:
        print(i + 1, len(jobs), f'{time.time() - t0:.0f}s', flush=True)
  summary = summarize(results, len(seeds))
  summary['meta'] = dict(
      version=I.AG_VERSION, split='test', seeds=[seeds[0], seeds[-1]], n_scenes=len(seeds), policies=kinds,
      wall_s=round(time.time() - t0, 1), skill=cfg['skill'], gate=cfg['gate'], reward=cfg['reward'],
      surrogate='kin fixed-base skill; success is not a real grasp success rate',
      stochastic_grasp=float(cfg['skill']['grasp_p']) < 1.0)
  out.write_text(json.dumps(summary, indent=1))
  with open(out.with_name(out.stem + '_episodes.jsonl'), 'w') as f:
    for r in sorted(results, key=lambda r: (r['policy'], r['seed'])):
      f.write(json.dumps(r) + '\n')
  print('written', out, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
  main()
