import json
import pathlib
import sys

import numpy as np

from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import expert, scenes
from embodied.run.eval_legmm import (
    bootstrap_ci, count_false_triggers, eval_legmm, parse_scene_types,
    select_seeds,
)
from drone_signature import load_config


def test_bootstrap_ci_edges_and_mean():
  assert bootstrap_ci(np.ones(5)) == [1.0, 1.0]
  data = np.arange(5, dtype=float)
  lo, hi = bootstrap_ci(data, seed=7)
  assert lo <= data.mean() <= hi


def test_open_seed_filter_is_type_only_and_repeatable():
  a = select_seeds('train', parse_scene_types('OPEN'), 12)
  b = select_seeds('train', parse_scene_types('OPEN'), 12)
  assert a == b
  assert all(scenes.scene_type_of('train', seed) == scenes.SceneType.OPEN for seed in a)


def test_scene_filter_soluble_and_none_are_deterministic():
  cfg = dict(load_config(['legmm', 'sizetest']).env.legmm)
  kept, info = select_seeds('train', 'OPEN', 8, 'solvable', cfg, return_info=True)
  again, again_info = select_seeds('train', 'OPEN', 8, 'solvable', cfg, return_info=True)
  assert kept == again and info == again_info
  assert all(expert.scene_solvable(scenes.make_scene('train', seed, cfg), cfg)
             for seed in kept)
  assert info['scene_filter'] == 'solvable' and info['seeds_rejected'] >= 1
  raw = select_seeds('train', 'OPEN', 8, 'none', cfg)
  assert raw == [4, 5, 7, 8, 10, 11, 15, 19]
  assert 10 in raw and 10 not in kept


def test_false_triggers_use_skill_entries():
  skills = [int(I.Skill.NAV), int(I.Skill.NAV), int(I.Skill.PICK),
            int(I.Skill.PICK), int(I.Skill.NAV), int(I.Skill.PLACE)]
  aff = [1.0, 1.0, 0.1, 0.0, 1.0, 0.1]
  assert count_false_triggers(skills, aff) == 2


def test_eval_legmm_two_open_episodes(tmp_path):
  # sizetest + random_agent keeps this an actual policy/environment run while
  # avoiding a checkpoint dependency in the unit suite.
  root = pathlib.Path(__file__).resolve().parents[2]
  sys.path.insert(0, str(root / 'dreamerv3'))
  import main as dmain
  config = load_config(['legmm', 'sizetest', 'legmm_debug'], [
      '--script=eval_legmm', '--random_agent=True', '--jax.platform=cpu',
      '--jax.policy_devices=0', '--jax.train_devices=0', '--jax.prealloc=False',
      '--env.legmm.split=train', '--env.legmm.max_episode_steps=2',
      '--eval.episodes=2', '--eval.scene_types=OPEN', '--eval.online_aff_rollouts=0',
      f'--logdir={tmp_path}',
  ])
  args = config.run
  rows = eval_legmm(lambda: dmain.make_agent(config),
                    lambda index, **kw: dmain.make_env(config, index, **kw),
                    lambda: None, args, config=config)
  assert len(rows) == 2
  data = [json.loads(line) for line in (tmp_path / 'episodes.jsonl').read_text().splitlines()]
  fields = {f.name for f in I.EpisodeMetrics.__dataclass_fields__.values()}
  assert all(fields <= set(row) for row in data)
  assert all(row['affordance_source'] == 'aff_geom' for row in data)
  assert all(isinstance(row['steps'], int) and row['term_reason'] in I.TERM_REASONS for row in data)
  assert all(row['scene_type'] == int(scenes.SceneType.OPEN) for row in data)
