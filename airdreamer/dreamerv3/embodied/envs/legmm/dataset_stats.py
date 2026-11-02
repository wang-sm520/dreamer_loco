"""dataset_stats.json and figures for a collected legmm dataset (brief §7.4)."""
import collections
import json
import math
import pathlib

import numpy as np

from . import interfaces as I


def load_meta(out):
  out = pathlib.Path(out)
  side = collections.defaultdict(list)
  for f in sorted((out / 'meta').glob('side_*.npz')):
    d = np.load(f)
    for k in d.files:
      side[k].append(d[k])
  side = {k: np.concatenate(v, 0) for k, v in side.items()}
  episodes = []
  for f in sorted((out / 'meta').glob('episodes_*.jsonl')):
    episodes += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
  return side, episodes


def task_failure(e):
  """Episode ends in failure or contains a failed manipulation attempt
  (failed grasp, pick/place controller failure). Navigation retries of the
  expert (stuck, alignment timeouts) do not count."""
  manip = sum(v for k, v in e.items()
              if k.startswith(('expert_fail_pick_', 'expert_fail_place_', 'fail_pick_', 'fail_place_')))
  return (not e['success']) or e.get('failed_grasps', 0) > 0 or manip > 0


def summarize(out, figs=True):
  out = pathlib.Path(out)
  side, eps = load_meta(out)
  names = [t.name for t in I.SceneType]
  n = len(side['sample_id'])
  skills = np.bincount(side['skill'], minlength=3) / max(n, 1)
  succ = np.array([e['success'] for e in eps])
  with_failure = np.array([task_failure(e) for e in eps])
  per_type = {}
  for t in sorted({e['scene_type'] for e in eps}):
    es = [e for e in eps if e['scene_type'] == t]
    per_type[names[t]] = dict(
        episodes=len(es), success=float(np.mean([e['success'] for e in es])),
        pick_success=float(np.mean([e['pick_success'] for e in es])))
  labels = {}
  for i, name in ((0, 'pick'), (1, 'place')):
    m = side['label_aff_mask'][:, i] > 0
    y = side['label_aff'][m, i]
    g = side['aff_geom'][m, i]
    labels[name] = dict(
        labeled=int(m.sum()), frac_positive=float(np.mean(y > 0.5)) if m.any() else float('nan'),
        hist=np.histogram(y, bins=[0, 0.01, 0.26, 0.51, 0.76, 1.01])[0].tolist(),
        aff_geom_auroc=I.auroc(g, y > 0.5) if m.any() else float('nan'),
        aff_geom_ece=I.ece(g, y) if m.any() else float('nan'))
  chunks = list(out.glob('*.npz'))
  summary = dict(
      steps=int(n), hours=n / I.HL_HZ / 3600, episodes=len(eps),
      skill_frac=dict(zip(('NAV', 'PICK', 'PLACE'), skills.round(4).tolist())),
      success=float(succ.mean()) if len(eps) else float('nan'),
      episodes_with_failure=float(with_failure.mean()) if len(eps) else float('nan'),
      episodes_ending_in_failure=float(1 - succ.mean()) if len(eps) else float('nan'),
      term_reasons=dict(collections.Counter(e['term_reason'] for e in eps)),
      per_scene_type=per_type, labels=labels,
      explore_frac=float(side['explore'].mean()), dart_episode_frac=float(np.mean([e['dart'] for e in eps])),
      bad_pose_episodes=int(sum(e.get('expert_bad_pose', 0) > 0 for e in eps)),
      early_switch_episodes=int(sum(e.get('expert_early_switch', 0) > 0 for e in eps)),
      mean_episode_steps=float(np.mean([e['steps'] for e in eps])) if eps else float('nan'),
      mean_skill_switches=float(np.mean([e['n_skill_switches'] for e in eps])) if eps else float('nan'),
      chunk_files=len(chunks), disk_gb=sum(f.stat().st_size for f in chunks) / 1e9)
  stats = dict(summary=summary)
  (out / 'dataset_stats.json').write_text(json.dumps(stats, indent=1))
  if figs:
    _figures(out, side, eps, summary)
  return stats


def _figures(out, side, eps, summary):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fdir = out / 'figs'
  fdir.mkdir(exist_ok=True)
  fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
  ax[0].bar(list(summary['skill_frac']), list(summary['skill_frac'].values()))
  ax[0].set_title('skill fraction (steps)')
  pt = summary['per_scene_type']
  ax[1].bar(list(pt), [v['success'] for v in pt.values()], label='success')
  ax[1].bar(list(pt), [v['pick_success'] for v in pt.values()], alpha=0.4, label='pick')
  ax[1].set_title('expert success by scene type (with diversity)')
  ax[1].legend(fontsize=7)
  ax[1].tick_params(axis='x', labelsize=7)
  for i, name in ((0, 'pick'), (1, 'place')):
    m = side['label_aff_mask'][:, i] > 0
    ax[2].hist(side['label_aff'][m, i], bins=np.linspace(0, 1, 6), alpha=0.6, label=name)
  ax[2].set_title('label_aff distribution')
  ax[2].legend()
  fig.tight_layout()
  fig.savefig(fdir / 'summary.png', dpi=90)
  plt.close(fig)
  # Base pose coverage around the object (object-centred frame), coloured by aff.
  m = side['label_aff_mask'][:, 0] > 0
  rel = side['base'][m, :2] - side['obj_center'][m, :2]
  fig, ax = plt.subplots(figsize=(5.5, 5))
  sc = ax.scatter(rel[:, 0], rel[:, 1], c=side['label_aff'][m, 0], s=2, cmap='viridis', vmin=0, vmax=1)
  ax.set_aspect('equal')
  ax.set_xlim(-2.1, 2.1)
  ax.set_ylim(-2.1, 2.1)
  ax.plot(0, 0, 'r*')
  plt.colorbar(sc, label='aff_pick (rollout)')
  ax.set_title('base positions around the object (labeled TO_PICK states)')
  fig.savefig(fdir / 'coverage_pick.png', dpi=90)
  plt.close(fig)
  fig, ax = plt.subplots(figsize=(5, 3.5))
  ax.hist([e['steps'] for e in eps], bins=30)
  ax.set_title('episode length (high-level steps)')
  fig.savefig(fdir / 'episode_length.png', dpi=90)
  plt.close(fig)
