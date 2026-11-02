"""AI review of affordance labels (brief §7.3).

python tools/label_qa.py --data ~/data/legmm_v1/train --judge <module:function> [--apply True]

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


def make_packet(side, row, which, index, split, outdir):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  seed = int(side['seed'][row])
  scene = scenes.make_scene(split, seed, {})
  pose = _PoseView(side['base'][row], side['obj_center'][row])
  sid = int(side['sample_id'][row])
  pdir = pathlib.Path(outdir) / f'{sid}_{which}'
  pdir.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(5, 5))
  viz.topdown(ax, scene, pose, None, title=f'{split}/{seed} {["pick", "place"][which]}')
  fig.savefig(pdir / 'topdown.png', dpi=80)
  plt.close(fig)
  depth = index.step(sid)
  for k, v in depth.items():
    plt.imsave(pdir / f'{k}.png', viz.depth_rgb(v))
  state = dict(
      sample_id=sid, which=['pick', 'place'][which], split=split, seed=seed,
      scene_type=I.SceneType(int(side['scene_type'][row])).name,
      phase=I.Phase(int(side['phase'][row])).name,
      aff_rollout=float(side['label_aff'][row, which]), aff_geom=float(side['aff_geom'][row, which]),
      base_xy_yaw_h=[round(float(x), 3) for x in side['base'][row]],
      object_center=[round(float(x), 3) for x in side['obj_center'][row]],
      place_zone=dict(center=scene.place.center.round(3).tolist(), support_z=round(scene.place.support_z, 3)),
      approach_dirs_free=scene.approach.astype(int).tolist(),
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


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--data', required=True)
  p.add_argument('--judge', default='none')
  p.add_argument('--split', default='')
  p.add_argument('--max_packets', type=int, default=0, help='0 = all selected samples')
  p.add_argument('--apply', default='False')
  p.add_argument('--seed', type=int, default=0)
  a = p.parse_args()
  data = pathlib.Path(a.data).expanduser()
  side, eps = dataset_stats.load_meta(data)
  split = a.split or (eps[0]['split'] if eps else 'train')
  rng = np.random.default_rng(a.seed)
  judge = load_judge(a.judge)
  labeled = int((side['label_aff_mask'] > 0).sum())
  rows = select(side)
  if a.max_packets and len(rows) > a.max_packets:
    rows = [rows[i] for i in sorted(rng.choice(len(rows), a.max_packets, replace=False))]
  index = ChunkIndex(data)
  qa_dir = data / 'qa'
  qa_dir.mkdir(exist_ok=True)
  rejects, log = [], []
  for row, which in rows:
    packet = make_packet(side, row, which, index, split, qa_dir / 'packets')
    out = judge(packet)
    aff = packet['state']['aff_rollout']
    reject = (out.get('verdict') in ('feasible', 'infeasible') and contradicts(out['verdict'], aff)
              and float(out.get('confidence', 0.0)) >= CONF_THRESHOLD)
    entry = dict(sample_id=packet['state']['sample_id'], which=packet['state']['which'],
                 aff_rollout=aff, aff_geom=packet['state']['aff_geom'], judge=a.judge,
                 verdict=out.get('verdict'), confidence=out.get('confidence'),
                 reason=out.get('reason'), rejected=bool(reject), applied=a.apply == 'True')
    log.append(entry)
    if reject:
      rejects.append((packet['state']['sample_id'], which))
  with open(qa_dir / 'qa_log.jsonl', 'a') as f:
    for e in log:
      f.write(json.dumps(e) + '\n')
  if a.apply == 'True' and rejects:
    drop_masks(data, index, rejects)
  examples = [log[i] for i in rng.choice(len(log), min(20, len(log)), replace=False)] if log else []
  report = dict(
      data=str(data), judge=a.judge, labeled=labeled, selected=len(select(side)),
      selected_frac=len(select(side)) / max(labeled, 1), reviewed=len(log), rejected=len(rejects),
      rejected_frac_of_labeled=len(rejects) / max(labeled, 1), applied=a.apply == 'True',
      verdicts={v: sum(e['verdict'] == v for e in log) for v in ('feasible', 'infeasible', 'uncertain')},
      examples=examples)
  (qa_dir / 'qa_report.json').write_text(json.dumps(report, indent=1))
  print(json.dumps({k: v for k, v in report.items() if k != 'examples'}, indent=1))


if __name__ == '__main__':
  main()
