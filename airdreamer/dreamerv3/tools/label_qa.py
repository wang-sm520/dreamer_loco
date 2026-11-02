"""AI review of affordance labels (brief §7.3).

Single machine:
  python tools/label_qa.py --data ~/data/legmm_v1/train --judge <module:function> [--apply True]
Split across machines (packets/apply where the data is, judge where the API is reachable):
  python tools/label_qa.py --stage packets --data ~/data/legmm_v1/train [--max_packets N]
  python tools/label_qa.py --stage judge --packets <data>/qa/packets --judge legmm_judge_sonnet:judge \
      --verdicts <data>/qa/verdicts.jsonl
  python tools/label_qa.py --stage apply --data ~/data/legmm_v1/train --verdicts <...>/verdicts.jsonl --apply True

1. Filter labeled samples with |aff_rollout - aff_geom| > 0.5 or 0.25 < aff_rollout < 0.75.
2. Build review packets: top-down sketch PNG, head and wrist depth PNGs, state JSON.
3. Call judge_fn(packet) -> {verdict: feasible|infeasible|uncertain, confidence, reason}.
   The model behind judge_fn is chosen by the user; the default 'none' judge
   answers 'uncertain' for everything (dry run).
4. If the verdict contradicts the rollout label with confidence >= 0.8, set
   label_aff_mask of that sample to 0 (drop it). Label values are never
   overwritten. Every decision goes to qa_log.jsonl.
5. Report: filtered fraction, rejected fraction, 20 random examples.
"""
import argparse
import importlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from embodied.envs.legmm import dataset_stats
from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import scenes, viz

CONF_THRESHOLD = 0.8


def none_judge(packet):
  return {'verdict': 'uncertain', 'confidence': 0.0, 'reason': 'no judge configured (dry run)'}


def load_judge(spec):
  if spec in ('', 'none'):
    return none_judge
  module, fn = spec.split(':')
  return getattr(importlib.import_module(module), fn)


def select(side, both=True):
  """Indices (row, which) of labeled samples that need review."""
  rows = []
  for which in (0, 1):
    m = side['label_aff_mask'][:, which] > 0
    y = side['label_aff'][:, which]
    g = side['aff_geom'][:, which]
    sel = m & ((np.abs(y - g) > 0.5) | ((y > 0.25) & (y < 0.75)))
    rows += [(int(i), which) for i in np.where(sel)[0]]
  return rows


class ChunkIndex:
  """sample_id -> (chunk file, index)."""

  def __init__(self, data):
    self.files = sorted(pathlib.Path(data).glob('*.npz'))
    self.where = {}
    for f in self.files:
      ids = np.load(f)['sample_id']
      for i, sid in enumerate(ids):
        self.where[int(sid)] = (f, i)

  def step(self, sid, keys=('depth_head', 'depth_wrist')):
    f, i = self.where[int(sid)]
    d = np.load(f)
    return {k: d[k][i] for k in keys}


class _PoseView:
  def __init__(self, base, obj_center):
    self.x, self.y, self.yaw = float(base[0]), float(base[1]), float(base[2])
    self.obj_center = np.asarray(obj_center)


def make_packet(side, row, which, index, split, outdir, images=True):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  seed = int(side['seed'][row])
  scene = scenes.make_scene(split, seed, {})
  pose = _PoseView(side['base'][row], side['obj_center'][row])
  sid = int(side['sample_id'][row])
  pdir = pathlib.Path(outdir) / f'{sid}_{which}'
  pdir.mkdir(parents=True, exist_ok=True)
  if not images:
    scene = None
  fig, ax = (plt.subplots(figsize=(5, 5)) if images else (None, None))
  if images:
    viz.topdown(ax, scene, pose, None, title=f'{split}/{seed} {["pick", "place"][which]}')
    fig.savefig(pdir / 'topdown.png', dpi=80)
    plt.close(fig)
    depth = index.step(sid)
    for k, v in depth.items():
      plt.imsave(pdir / f'{k}.png', viz.depth_rgb(v))
  else:
    scene = scenes.make_scene(split, seed, {})
  state = dict(
      sample_id=sid, which=['pick', 'place'][which], split=split, seed=seed,
      scene_type=I.SceneType(int(side['scene_type'][row])).name,
      phase=I.Phase(int(side['phase'][row])).name,
      aff_rollout=float(side['label_aff'][row, which]), aff_geom=float(side['aff_geom'][row, which]),
      base_xy_yaw_h=[round(float(x), 3) for x in side['base'][row]],
      object_center=[round(float(x), 3) for x in side['obj_center'][row]],
      place_zone=dict(center=scene.place.center.round(3).tolist(), support_z=round(scene.place.support_z, 3)),
      approach_dirs_free=scene.approach.astype(int).tolist(),
      reconstruct={k: np.asarray(side[k][row]).tolist() for k in (
          'ee_b', 'R_ee_b', 'v', 'grip', 'obj_R', 'obj_offset', 'obj_R_rel', 'dr_vel_scale',
          'dr_latency', 'holding') if k in side},
      question=('Can the robot complete the ' + ['grasp', 'placement'][which] +
                ' from this base pose using only small base adjustments (no re-navigation)?'))
  (pdir / 'state.json').write_text(json.dumps(state, indent=1))
  return dict(dir=str(pdir), images=[str(pdir / 'topdown.png'), str(pdir / 'depth_head.png'),
                                     str(pdir / 'depth_wrist.png')], state=state)


def contradicts(verdict, aff):
  return (verdict == 'feasible' and aff < 0.5) or (verdict == 'infeasible' and aff >= 0.5)


def drop_masks(data, index, rejects):
  """Sets label_aff_mask[which] = 0 in the replay chunks for rejected samples."""
  by_file = {}
  for sid, which in rejects:
    f, i = index.where[int(sid)]
    by_file.setdefault(f, []).append((i, which))
  for f, items in by_file.items():
    d = dict(np.load(f))
    for i, which in items:
      d['label_aff_mask'][i, which] = 0.0
    tmp = f.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **d)
    tmp.replace(f)


def _packets_stage(a, data, side, eps, rng):
  split = a.split or (eps[0]['split'] if eps else 'train')
  rows = select(side)
  if a.max_packets and len(rows) > a.max_packets:
    rows = [rows[i] for i in sorted(rng.choice(len(rows), a.max_packets, replace=False))]
  index = ChunkIndex(data)
  pdir = data / 'qa' / 'packets'
  pdir.mkdir(parents=True, exist_ok=True)
  with open(pdir / 'index.jsonl', 'w') as f:
    for row, which in rows:
      packet = make_packet(side, row, which, index, split, pdir, a.images == 'True')
      f.write(json.dumps({'dir': pathlib.Path(packet['dir']).name, 'state': packet['state']}) + '\n')
  return len(rows)


def _load_packets(pdir):
  pdir = pathlib.Path(pdir).expanduser()
  for line in (pdir / 'index.jsonl').read_text().splitlines():
    entry = json.loads(line)
    d = pdir / entry['dir']
    yield dict(dir=str(d), state=entry['state'],
               images=[str(d / n) for n in ('topdown.png', 'depth_head.png', 'depth_wrist.png')])


def _judge_stage(a):
  """Judges every packet without a verdict yet (resumable), a.workers calls in parallel."""
  import concurrent.futures
  import threading
  judge = load_judge(a.judge)
  vpath = pathlib.Path(a.verdicts).expanduser()
  done = set()
  if vpath.exists():
    done = {(v['sample_id'], v['which']) for v in map(json.loads, vpath.read_text().splitlines())}
  todo = [p for p in _load_packets(a.packets)
          if (p['state']['sample_id'], p['state']['which']) not in done]
  lock = threading.Lock()
  n = 0
  with open(vpath, 'a') as f:
    def work(packet):
      out = judge(packet)
      rec = {'sample_id': packet['state']['sample_id'], 'which': packet['state']['which'],
             'judge': a.judge, **out}
      with lock:
        f.write(json.dumps(rec) + '\n')
        f.flush()
    with concurrent.futures.ThreadPoolExecutor(max(1, a.workers)) as pool:
      for _ in pool.map(work, todo):
        n += 1
  return n


def _apply_stage(a, data, side, verdicts):
  labeled = int((side['label_aff_mask'] > 0).sum())
  rng = np.random.default_rng(a.seed)
  states = {(e['state']['sample_id'], e['state']['which']): e['state']
            for e in _load_packets(data / 'qa' / 'packets')}
  rejects, log = [], []
  for v in verdicts:
    st = states.get((v['sample_id'], v['which']))
    if st is None:
      continue
    aff = st['aff_rollout']
    reject = (v.get('verdict') in ('feasible', 'infeasible') and contradicts(v['verdict'], aff)
              and float(v.get('confidence', 0.0)) >= CONF_THRESHOLD)
    log.append(dict(sample_id=st['sample_id'], which=st['which'], aff_rollout=aff,
                    aff_geom=st['aff_geom'], judge=v.get('judge'), model=v.get('model'),
                    verdict=v.get('verdict'), confidence=v.get('confidence'), reason=v.get('reason'),
                    rejected=bool(reject), applied=a.apply == 'True'))
    if reject:
      rejects.append((st['sample_id'], 0 if st['which'] == 'pick' else 1))
  qa_dir = data / 'qa'
  with open(qa_dir / 'qa_log.jsonl', 'a') as f:
    for e in log:
      f.write(json.dumps(e) + '\n')
  if a.apply == 'True' and rejects:
    drop_masks(data, ChunkIndex(data), rejects)
  examples = [log[i] for i in rng.choice(len(log), min(20, len(log)), replace=False)] if log else []
  selected = len(select(side))
  report = dict(
      data=str(data), labeled=labeled, selected=selected, selected_frac=selected / max(labeled, 1),
      reviewed=len(log), rejected=len(rejects), rejected_frac_of_labeled=len(rejects) / max(labeled, 1),
      rejected_frac_of_reviewed=len(rejects) / max(len(log), 1), applied=a.apply == 'True',
      verdicts={k: sum(e['verdict'] == k for e in log) for k in ('feasible', 'infeasible', 'uncertain')},
      agree_with_rollout=sum((e['verdict'] == 'feasible') == (e['aff_rollout'] >= 0.5)
                             for e in log if e['verdict'] != 'uncertain'),
      examples=examples)
  (qa_dir / 'qa_report.json').write_text(json.dumps(report, indent=1))
  return report


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--stage', default='all', choices=['all', 'packets', 'judge', 'apply'])
  p.add_argument('--data', default='')
  p.add_argument('--packets', default='')
  p.add_argument('--verdicts', default='')
  p.add_argument('--judge', default='none')
  p.add_argument('--split', default='')
  p.add_argument('--max_packets', type=int, default=0, help='0 = all selected samples')
  p.add_argument('--apply', default='False')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--workers', type=int, default=1, help='parallel judge calls')
  p.add_argument('--images', default='True', help='False: state-only packets (oracle judge)')
  a = p.parse_args()
  if a.stage == 'judge':
    print(json.dumps({'judged': _judge_stage(a), 'verdicts': a.verdicts}))
    return
  data = pathlib.Path(a.data).expanduser()
  side, eps = dataset_stats.load_meta(data)
  rng = np.random.default_rng(a.seed)
  if a.stage in ('all', 'packets'):
    n = _packets_stage(a, data, side, eps, rng)
    print(json.dumps({'packets': n, 'selected': len(select(side)),
                      'labeled': int((side['label_aff_mask'] > 0).sum())}))
    if a.stage == 'packets':
      return
    a.packets = str(data / 'qa' / 'packets')
    a.verdicts = a.verdicts or str(data / 'qa' / 'verdicts.jsonl')
    _judge_stage(a)
  verdicts = [json.loads(l) for l in pathlib.Path(a.verdicts).expanduser().read_text().splitlines()]
  report = _apply_stage(a, data, side, verdicts)
  print(json.dumps({k: v for k, v in report.items() if k != 'examples'}, indent=1))


if __name__ == '__main__':
  main()
