"""Scripted expert success rate per scene type (M2 acceptance).

python tools/legmm_expert_eval.py --episodes 100 --diversity False --out docs/legmm/m2_expert_eval.json
"""
import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests' / 'legmm'))

import numpy as np


def run_episodes(args):
  split, seeds, diversity, wid, overrides, *selection = args
  filtered = bool(selection and selection[0])
  from legmm_utils import make_env, zero_action
  from embodied.envs.legmm import expert as ex
  from embodied.envs.legmm import interfaces as I
  env = make_env(split, scene_filter='solvable' if filtered else 'none')
  out = []
  for seed in seeds:
    rng = np.random.default_rng([int(seed), 0xE1A1])
    env.rng = np.random.default_rng(rng.integers(2 ** 63))
    env.forced_seed = None
    scene = env._sample_scene(seed=int(seed))
    attempts, rejected = env.scene_filter_attempts, env.scene_filter_rejected
    env.forced_seed = scene.seed
    obs = env.step(zero_action(reset=True))
    expert = ex.ScriptedExpert(ex.ExpertConfig(diversity=diversity, **overrides))
    expert.reset(env.scene, np.random.default_rng(rng.integers(2 ** 63)), env.backend)
    ret, steps, last_skill, holding_before = 0.0, 0, 0, False
    t0 = time.perf_counter()
    diag = {}
    while not obs['is_last']:
      act = expert.act()
      last_skill = int(act['skill'])
      holding_before = bool(env.backend.state.holding)
      sub = getattr(expert.ctrl, 'sub', None) if expert.mode == 'manip' else expert.mode
      obs = env.step({**act, 'reset': False})
      if obs['is_last'] and I.TERM_REASONS[int(obs['log/term_reason'])] == 'drop':
        diag = _drop_diag(env, sub)
      if obs['is_last'] and I.TERM_REASONS[int(obs['log/term_reason'])] == 'collision':
        diag = dict(mode=sub, explore=expert.explore_left > 0)
      ret += float(obs['reward'])
      steps += 1
    out.append(dict(seed=env.scene.seed, requested_seed=int(seed),
                    object_shape=env.scene.obj.shape, object_size=(2 * env.scene.obj.half).tolist(),
                    object_yaw=env.scene.obj.yaw, filter_attempts=attempts, filter_rejected=rejected,
                    scene_type=int(obs['log/scene_type']),
                    success=bool(obs['log/success']), pick_success=bool(obs['log/pick_success']),
                    term_reason=I.TERM_REASONS[int(obs['log/term_reason'])], steps=steps,
                    ret=ret, seconds=time.perf_counter() - t0, last_skill=last_skill,
                    holding_before=holding_before, drop_diag=diag, **expert.stats))
  return out


def _drop_diag(env, sub):
  import copy
  from embodied.envs.legmm import geometry as geo, kin_backend as kb
  b = env.backend
  s = copy.deepcopy(b.state)
  s.holding = True
  pts = b.arm_points(s)
  pts_b = (pts - b.base_pos(s)) @ b.R_body(s)
  hp = b.obj_corners(s)
  return dict(
      sub=sub, reason=b.drop_reason, h=round(float(s.h), 3), latency=int(b.dr.arm_latency),
      arm_static=[int(i) for i in np.where(geo.points_in_boxes(pts, b._static, kb.LINK_RADIUS).any(1))[0]],
      arm_body=[int(i) + 1 for i in np.where(geo.points_in_boxes(pts_b[1:], kb.BODY_BOX[None], 0.01).any(1))[0]],
      held_static=bool(geo.points_in_boxes(hp, b._static, 0.0).any()),
      held_bottom_z=round(float(hp[:, 2].min()), 3), ee_b=[round(float(x), 3) for x in s.ee_b])


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--episodes', type=int, default=100, help='per scene type')
  p.add_argument('--diversity', default='False')
  p.add_argument('--workers', type=int, default=16)
  p.add_argument('--out', default='')
  p.add_argument('--cfg', default='', help='ExpertConfig overrides, e.g. explore_frac=0,p_dart=0')
  p.add_argument('--types', default='', help='comma separated scene type ids to run')
  p.add_argument('--solvable', default='False', choices=['True', 'False'],
                 help='True: filter train/val like collection; test splits always remain raw')
  a = p.parse_args()
  diversity = a.diversity == 'True'
  overrides = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in a.cfg.split(',') if kv}
  types = {int(t) for t in a.types.split(',') if t}
  from embodied.envs.legmm import interfaces as I
  from embodied.envs.legmm import scenes
  jobs = []
  for split in ('train', 'test_constrained'):
    lo, hi = I.SPLITS[split]['seeds']
    wanted = set(I.SPLITS[split]['types'])
    if split == 'test_constrained':
      wanted &= {4, 5}
    if types:
      wanted &= types
    if not wanted:
      continue
    counts = collections.Counter()
    for seed in range(lo, hi):
      t = int(scenes.scene_type_of(split, seed))
      if t not in wanted or counts[t] >= a.episodes:
        continue
      jobs.append((split, [seed], diversity, len(jobs), overrides, a.solvable == 'True'))
      counts[t] += 1
      if all(counts[t] >= a.episodes for t in wanted):
        break
    if any(counts[t] < a.episodes for t in wanted):
      raise RuntimeError(f'not enough seeds in {split}: {dict(counts)}')
  rows = []
  with mp.Pool(a.workers) as pool:
    for chunk in pool.imap_unordered(run_episodes, jobs):
      rows.extend(chunk)
      if len(rows) % 25 == 0 or len(rows) == len(jobs):
        print(f'evaluated {len(rows)}/{len(jobs)} episodes', flush=True)
  rows.sort(key=lambda r: (r['scene_type'], r['requested_seed']))
  names = [t.name for t in I.SceneType]
  summary = {}
  for t in sorted({r['scene_type'] for r in rows}):
    rs = [r for r in rows if r['scene_type'] == t]
    summary[names[t]] = dict(
        n=len(rs), success=float(np.mean([r['success'] for r in rs])),
        pick_success=float(np.mean([r['pick_success'] for r in rs])),
        mean_steps=float(np.mean([r['steps'] for r in rs])),
        term_reasons=dict(collections.Counter(r['term_reason'] for r in rs)),
        failure_causes=dict(sum((collections.Counter({k: v for k, v in r.items() if k.startswith('fail_')})
                                 for r in rs if not r['success']), collections.Counter())),
        drop_skill=dict(collections.Counter(['NAV', 'PICK', 'PLACE'][r['last_skill']]
                                            for r in rs if r['term_reason'] == 'drop')),
        collision_skill=dict(collections.Counter(['NAV', 'PICK', 'PLACE'][r['last_skill']]
                                                 for r in rs if r['term_reason'] == 'collision')),
        collision_mode=dict(collections.Counter(str(r['drop_diag'].get('mode'))
                                                for r in rs if r['term_reason'] == 'collision')))
  for name, stats in summary.items():
    rs = [r for r in rows if names[r['scene_type']] == name]
    attempts = sum(r['filter_attempts'] for r in rs)
    rejected = sum(r['filter_rejected'] for r in rs)
    stats['planner_filter'] = dict(attempts=attempts, rejected=rejected,
                                   rejection_fraction=rejected / attempts if attempts else None)
    stats['per_object_shape'] = {
        shape: dict(n=len(sub), success=float(np.mean([r['success'] for r in sub])),
                    pick_success=float(np.mean([r['pick_success'] for r in sub])))
        for shape in sorted({r['object_shape'] for r in rs})
        if (sub := [r for r in rs if r['object_shape'] == shape])}
  res = dict(diversity=diversity, selection='filtered_train_val' if a.solvable == 'True' else 'raw',
             per_type=summary, mean_return=float(np.mean([r['ret'] for r in rows])),
             sec_per_episode=float(np.mean([r['seconds'] for r in rows])))
  print(json.dumps(res, indent=1))
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(dict(summary=res, rows=rows), indent=1))


if __name__ == '__main__':
  main()
