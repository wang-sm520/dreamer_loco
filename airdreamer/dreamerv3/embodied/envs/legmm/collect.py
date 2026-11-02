"""Offline dataset collection with the scripted expert (brief §7.4).

python -m embodied.envs.legmm.collect --split train --hours 30 --workers 20 --out ~/data/legmm_v1/train

Each worker runs LegMMEnv + ScriptedExpert (+ AffordanceLabeler) and writes
replay chunks with embodied.replay.Replay(directory=out).add(step, worker).
The transition layout matches embodied.core.Driver: step t holds obs_t and the
action taken after seeing obs_t; on is_last the action is zeroed and the next
step is a reset. Extra key `sample_id` (int64) links replay steps to the side
table in out/meta/ used by tools/label_qa.py; train_offline drops it.
"""
import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]   # .../airdreamer/dreamerv3

SIDE_KEYS = ('sample_id', 'episode', 't', 'seed', 'scene_type', 'phase', 'skill', 'label_aff',
             'label_aff_mask', 'aff_geom', 'base', 'obj_center', 'holding', 'explore', 'dart')


def load_env_cfg(presets=('legmm',), overrides=()):
  import elements
  import ruamel.yaml as yaml
  configs = yaml.YAML(typ='safe').load((ROOT / 'dreamerv3' / 'configs.yaml').read_text())
  config = elements.Config(configs['defaults'])
  for name in presets:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(list(overrides))
  return dict(config.env.legmm)


def _sample_id(worker, counter):
  return np.int64((int(worker) << 40) | int(counter))


def worker_main(args):
  (wid, split, n_steps, out, seed, labels, diversity, chunksize, overrides) = args
  sys.path.insert(0, str(ROOT))
  import embodied
  from embodied.envs.legmm import expert as ex
  from embodied.envs.legmm import interfaces as I
  from embodied.envs.legmm import labeler as lb
  from embodied.envs.legmm.env import LegMMEnv

  cfg = load_env_cfg(overrides=overrides)
  cfg.pop('use_seed', None)
  cfg['split'] = split
  env = LegMMEnv('mm', seed=seed * 1000 + wid, **cfg)
  if labels:
    lcfg = dict(cfg.get('labels', {}))
    env.labeler = lb.AffordanceLabeler(**lcfg, seed=seed * 1000 + wid)
  rng = np.random.default_rng([seed, wid])
  out = pathlib.Path(out)
  replay = embodied.replay.Replay(
      length=1, capacity=3 * chunksize, directory=out, chunksize=chunksize, save_wait=True)
  side = collections.defaultdict(list)
  episodes = []
  zero = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items() if k != 'reset'}

  def new_episode():
    obs = env.step({'reset': True, **zero})
    expert = ex.ScriptedExpert(ex.ExpertConfig(diversity=diversity))
    expert.reset(env.scene, np.random.default_rng(rng.integers(2 ** 63)), env.backend)
    return obs, expert

  obs, expert = new_episode()
  ep, t, counter = 0, 0, 0
  ep_stats = collections.Counter()
  t0 = time.perf_counter()
  while counter < n_steps:
    if obs['is_last']:
      act = dict(zero)
    else:
      act = expert.act()
    explore = expert.explore_left > 0
    sid = _sample_id(wid, counter)
    step = {k: v for k, v in obs.items() if not k.startswith('log/')}
    step.update(act)
    step['sample_id'] = sid
    replay.add(step, wid)
    counter += 1
    if counter % chunksize == 0:
      replay.save()
    s = env.backend.state
    last = env.labeler.last if env.labeler is not None else None
    side['sample_id'].append(sid)
    side['episode'].append(ep)
    side['t'].append(t)
    side['seed'].append(int(obs['log/seed']))
    side['scene_type'].append(int(obs['log/scene_type']))
    side['phase'].append(int(obs['label_phase']))
    side['skill'].append(int(act['skill']))
    side['label_aff'].append(np.asarray(obs['label_aff'], np.float32))
    side['label_aff_mask'].append(np.asarray(obs['label_aff_mask'], np.float32))
    side['aff_geom'].append(np.asarray(last['geom'] if last else np.zeros(2), np.float32))
    side['base'].append(np.array([s.x, s.y, s.yaw, s.h], np.float32))
    side['obj_center'].append(np.asarray(s.obj_center, np.float32))
    side['holding'].append(bool(s.holding))
    side['explore'].append(bool(explore))
    side['dart'].append(bool(expert.dart))
    ep_stats['grasp_attempts'] += int(obs.get('log/grasp_attempt', 0))
    ep_stats['failed_grasps'] += int(obs.get('log/failed_grasp', 0))
    ep_stats['arm_collisions'] += int(obs.get('log/arm_collision', 0))
    ep_stats['path_length'] += float(obs.get('log/base_disp', 0.0))
    if obs['is_last']:
      episodes.append(dict(
          worker=wid, episode=ep, split=split, seed=int(obs['log/seed']),
          scene_type=int(obs['log/scene_type']), success=bool(obs['log/success']),
          pick_success=bool(obs['log/pick_success']), steps=t,
          term_reason=I.TERM_REASONS[int(obs['log/term_reason'])],
          n_skill_switches=int(env._n_switches), dart=bool(expert.dart),
          **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in ep_stats.items()},
          **{f'expert_{k}': int(v) for k, v in expert.stats.items()}))
      obs, expert = new_episode()
      ep, t = ep + 1, 0
      ep_stats = collections.Counter()
    else:
      obs = env.step({**act, 'reset': False})
      t += 1
  replay.save()
  meta = out / 'meta'
  meta.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(meta / f'side_{wid:03d}.npz', **{k: np.asarray(v) for k, v in side.items()})
  with open(meta / f'episodes_{wid:03d}.jsonl', 'w') as f:
    for e in episodes:
      f.write(json.dumps(e) + '\n')
  return dict(worker=wid, steps=counter, episodes=len(episodes), seconds=time.perf_counter() - t0,
              label_calls=getattr(env.labeler, 'calls', 0), rollouts=getattr(env.labeler, 'rollouts', 0))


def collect(split, hours, workers, out, seed=0, labels=True, diversity=True, chunksize=1024,
            overrides=()):
  from embodied.envs.legmm import interfaces as I
  out = pathlib.Path(out).expanduser()
  out.mkdir(parents=True, exist_ok=True)
  total = int(round(hours * 3600 * I.HL_HZ))
  per = [total // workers + (1 if i < total % workers else 0) for i in range(workers)]
  jobs = [(w, split, per[w], str(out), seed, labels, diversity, chunksize, tuple(overrides))
          for w in range(workers)]
  t0 = time.perf_counter()
  if workers == 1:
    results = [worker_main(jobs[0])]
  else:
    with mp.get_context('spawn').Pool(workers) as pool:
      results = pool.map(worker_main, jobs)
  info = dict(split=split, hours=hours, total_steps=total, workers=workers, seed=seed,
              labels=labels, diversity=diversity, wall_seconds=time.perf_counter() - t0,
              workers_info=results)
  (out / 'meta' / 'collect_info.json').write_text(json.dumps(info, indent=1))
  from embodied.envs.legmm import dataset_stats
  stats = dataset_stats.summarize(out)
  return info, stats


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--split', default='train')
  p.add_argument('--hours', type=float, default=30.0)
  p.add_argument('--workers', type=int, default=20)
  p.add_argument('--out', required=True)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--labels', default='True')
  p.add_argument('--diversity', default='True')
  p.add_argument('--chunksize', type=int, default=1024)
  a, rest = p.parse_known_args()
  info, stats = collect(a.split, a.hours, a.workers, a.out, a.seed, a.labels == 'True',
                        a.diversity == 'True', a.chunksize, rest)
  print(json.dumps({k: v for k, v in info.items() if k != 'workers_info'}, indent=1))
  print(json.dumps(stats['summary'], indent=1))


if __name__ == '__main__':
  main()
