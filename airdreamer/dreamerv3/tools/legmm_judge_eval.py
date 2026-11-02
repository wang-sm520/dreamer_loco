"""Who is right on reviewed affordance labels? Compares, on samples that the QA
filter selects (pick labels), against a stronger simulator oracle:
stored label, labeler re-run, aff_geom, a rule judge, Codex with the original
prompt (v1) and with geometry + rules (v2).

python tools/legmm_judge_eval.py --data ~/data/legmm_v2/train --n 60 --gpt True
"""
import argparse
import collections
import concurrent.futures
import json
import math
import multiprocessing as mp
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np


def reconstruct(split, row):
  from embodied.envs.legmm import collect, kin_backend as kb
  from embodied.envs.legmm.env import LegMMEnv
  cfg = collect.load_env_cfg()
  cfg.pop('use_seed', None)
  cfg['split'] = split
  cfg['domain_rand'] = dict(cfg['domain_rand'], enabled=False)
  env = LegMMEnv('mm', seed=0, **cfg)
  env.forced_seed = int(row['seed'])
  env.step({'reset': True, **{k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items() if k != 'reset'}})
  s = env.backend.state
  s.x, s.y, s.yaw, s.h = (float(v) for v in row['base'])
  s.obj_center = np.asarray(row['obj_center'], np.float64)
  s.ee_b = env.backend.shoulder_b + kb.STOW_REL
  s.R_ee_b = kb.R_STOW.copy()
  return env


def features(env):
  from embodied.envs.legmm import expert as ex, scenes
  b, s, sc = env.backend, env.backend.state, env.scene
  obj = s.obj_center
  bearing = math.degrees(math.atan2(math.sin(math.atan2(obj[1] - s.y, obj[0] - s.x) - s.yaw),
                                    math.cos(math.atan2(obj[1] - s.y, obj[0] - s.x) - s.yaw)))
  sh = b.to_world(b.shoulder_b)
  to_obj = math.atan2(obj[1] - s.y, obj[0] - s.x)
  free = []
  for i in range(scenes.N_DIRS):
    if sc.approach[i]:
      d = scenes.APPROACH_DIRS[i]
      # Direction the gripper travels (-d) relative to the robot->object direction.
      ang = math.degrees(math.atan2(math.sin(math.atan2(-d[1], -d[0]) - to_obj),
                                    math.cos(math.atan2(-d[1], -d[0]) - to_obj)))
      free.append(round(ang))
  reach_ok = False
  for i, _, axes in ex.approach_options(sc, s.yaw):
    for _, a in axes:
      R_w = ex.grasp_frame(a, s.yaw)
      pre, goal, _ = ex.pick_targets(obj, a, i == scenes.N_DIRS)
      if ex.height_range(b, (s.x, s.y, s.yaw), (pre, goal), R_w, 0.10) is not None:
        reach_ok = True
  near = [k for bx, k in zip(sc.boxes, sc.kinds)
          if k != 'wall' and np.all(obj[:2] >= bx[:2] - 0.3) and np.all(obj[:2] <= bx[3:5] + 0.3)]
  return dict(bearing_deg=round(bearing, 1), shoulder_to_object_xy_m=round(math.hypot(obj[0] - sh[0], obj[1] - sh[1]), 3),
              object_height_m=round(float(obj[2]), 3), body_height_m=round(float(s.h), 3),
              reach_ok=bool(reach_ok), top_down_free=bool(sc.approach[scenes.N_DIRS]),
              free_approach_bearings_deg=free, furniture_within_0_3m=sorted(set(near)))


def rule_judge(f):
  """My judgment from the same information, written down as rules."""
  if not f['reach_ok'] or abs(f['bearing_deg']) > 60 + 17:
    return 'infeasible'
  if not (f['top_down_free'] or any(abs(a) <= 60 for a in f['free_approach_bearings_deg'])):
    return 'infeasible'
  if abs(f['bearing_deg']) > 45 or f['shoulder_to_object_xy_m'] > 0.85:
    return 'uncertain'
  return 'feasible'


def audit(args):
  split, row, seed = args
  from embodied.envs.legmm import labeler as lb, oracle
  env = reconstruct(split, row)
  f = features(env)
  weak = lb.AffordanceLabeler(seed=seed).rate(env, 0)
  weak45 = lb.AffordanceLabeler(seed=seed, horizon=45).rate(env, 0)
  weak60 = lb.AffordanceLabeler(seed=seed, horizon=60).rate(env, 0)
  orc, info = oracle.oracle_pick_rate(env, seed=seed)
  return dict(sample_id=int(row['sample_id']), label=float(row['label']), geom_stored=float(row['geom']),
              geom_rerun=lb.aff_geom_pick(env.backend, env.scene), weak_rerun=weak, weak_h45=weak45,
              weak_h60=weak60, oracle=orc,
              oracle_info=info, features=f, rule=rule_judge(f))


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--data', required=True)
  p.add_argument('--n', type=int, default=60)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--gpt', default='True')
  p.add_argument('--workers', type=int, default=16)
  p.add_argument('--gpt_workers', type=int, default=8)
  p.add_argument('--out', default='')
  a = p.parse_args()
  import label_qa
  from embodied.envs.legmm import dataset_stats
  data = pathlib.Path(a.data).expanduser()
  side, eps = dataset_stats.load_meta(data)
  split = eps[0]['split']
  rng = np.random.default_rng(a.seed)
  sel = [(r, w) for r, w in label_qa.select(side) if w == 0]
  picks = [sel[i] for i in sorted(rng.choice(len(sel), min(a.n, len(sel)), replace=False))]
  jobs = [(split, dict(sample_id=side['sample_id'][r], seed=side['seed'][r], base=side['base'][r].tolist(),
                       obj_center=side['obj_center'][r].tolist(), label=side['label_aff'][r, 0],
                       geom=side['aff_geom'][r, 0]), a.seed + i) for i, (r, _) in enumerate(picks)]
  with mp.get_context('spawn').Pool(a.workers) as pool:
    res = pool.map(audit, jobs)
  if a.gpt == 'True':
    import legmm_judge_codex as jc
    index = label_qa.ChunkIndex(data)
    pdir = data / 'qa' / 'judge_eval_packets'
    packets = []
    for (r, w), rr in zip(picks, res):
      pk = label_qa.make_packet(side, r, w, index, split, pdir)
      pk['images'] = [str(pathlib.Path(pk['dir']) / n) for n in ('topdown.png', 'depth_head.png', 'depth_wrist.png')]
      packets.append(pk)
    def call(i):
      pk = packets[i]
      v1 = jc.judge(pk)
      pk2 = dict(pk, state=dict(pk['state'], geometry=res[i]['features']))
      v2 = jc.judge_v2(pk2)
      return v1, v2
    with concurrent.futures.ThreadPoolExecutor(a.gpt_workers) as ex_:
      outs = list(ex_.map(call, range(len(packets))))
    for rr, (v1, v2) in zip(res, outs):
      rr['gpt_v1'], rr['gpt_v2'] = v1, v2
  truth = lambda r: r['oracle'] >= 0.5
  summary = {'n': len(res), 'oracle_positive_frac': float(np.mean([truth(r) for r in res])),
             'stored_label_agrees_with_oracle': float(np.mean([(r['label'] >= 0.5) == truth(r) for r in res])),
             'labeler_rerun_agrees_with_oracle': float(np.mean([(r['weak_rerun'] >= 0.5) == truth(r) for r in res])),
             'labeler_h45_agrees_with_oracle': float(np.mean([(r['weak_h45'] >= 0.5) == truth(r) for r in res])),
             'labeler_h60_agrees_with_oracle': float(np.mean([(r['weak_h60'] >= 0.5) == truth(r) for r in res])),
             'aff_geom_agrees_with_oracle': float(np.mean([(r['geom_rerun'] >= 0.5) == truth(r) for r in res]))}
  def judge_stats(name, get):
    dec = [(get(r), r) for r in res if get(r) in ('feasible', 'infeasible')]
    out = {'decided': len(dec), 'uncertain': len(res) - len(dec),
           'accuracy_vs_oracle': float(np.mean([(v == 'feasible') == truth(r) for v, r in dec])) if dec else float('nan'),
           'says_feasible_frac': float(np.mean([v == 'feasible' for v, _ in dec])) if dec else float('nan')}
    return out
  summary['rule_judge'] = judge_stats('rule', lambda r: r['rule'])
  for name in ('gpt_v1', 'gpt_v2'):
    if res and name in res[0]:
      summary[name] = judge_stats(name, lambda r, n=name: r[n]['verdict'])
      # The QA rule: reject when the judge contradicts the label with confidence >= 0.8.
      rej = [r for r in res if r[name]['verdict'] in ('feasible', 'infeasible') and r[name]['confidence'] >= 0.8
             and (r[name]['verdict'] == 'feasible') != (r['label'] >= 0.5)]
      summary[name]['would_reject'] = len(rej)
      summary[name]['rejections_correct'] = int(sum((r['label'] >= 0.5) != truth(r) for r in rej))
  print(json.dumps(summary, indent=1))
  out = pathlib.Path(a.out or data / 'qa' / 'judge_eval.json')
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(dict(summary=summary, rows=res), indent=1, default=float))


if __name__ == '__main__':
  main()
