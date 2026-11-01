"""Random-policy throughput and robustness check for LegMMEnv (M1).

python tools/legmm_bench.py --steps 5000 --split train --out bench.json
"""
import argparse
import collections
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'dreamerv3'))

import numpy as np


def make_env(split, seed, overrides=()):
  sys.path.insert(0, str(ROOT / 'tests' / 'legmm'))
  from drone_signature import load_config
  import main as dmain
  config = load_config(['legmm'], [f'--env.legmm.split={split}', f'--seed={seed}',
                                   '--logdir=/tmp/legmm_bench', *overrides])
  return dmain.make_env(config, 0)


def random_action(env, rng, skill_probs=(0.6, 0.25, 0.15)):
  skill = int(rng.choice(3, p=skill_probs))
  act = {'reset': False, 'skill': np.int32(skill)}
  for k in ('nav', 'pick', 'place'):
    act[k] = rng.uniform(-1, 1, env.act_space[k].shape).astype(np.float32)
  return act


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--steps', type=int, default=5000)
  p.add_argument('--split', default='train')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', default='')
  a = p.parse_args()
  env = make_env(a.split, a.seed)
  rng = np.random.default_rng(a.seed)
  obs = env.step({'reset': True, **{k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items() if k != 'reset'}})
  reasons = collections.Counter()
  scene_types = collections.Counter()
  n_eps, n_nan = 0, 0
  t0 = time.perf_counter()
  for _ in range(a.steps):
    obs = env.step(random_action(env, rng))
    if not all(np.isfinite(obs[k]).all() for k in ('proprio', 'goal', 'reward')):
      n_nan += 1
    if obs['is_last']:
      from embodied.envs.legmm.interfaces import TERM_REASONS
      reasons[TERM_REASONS[int(obs['log/term_reason'])]] += 1
      scene_types[int(obs['log/scene_type'])] += 1
      n_eps += 1
  dt = time.perf_counter() - t0
  res = dict(split=a.split, steps=a.steps, seconds=round(dt, 3), steps_per_s=round(a.steps / dt, 1),
             episodes=n_eps, nonfinite_steps=n_nan, term_reasons=dict(reasons),
             scene_types=dict(scene_types))
  print(json.dumps(res))
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
  main()
