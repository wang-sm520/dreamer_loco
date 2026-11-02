"""The collection filter must use the expert planner without changing strata."""
import numpy as np
import pytest

from embodied.envs.legmm import expert, kin_backend, scenes
from legmm_utils import env_cfg, make_env, zero_action


def test_filter_accepts_scene_the_expert_can_plan():
  cfg = env_cfg()
  cfg['object'] = dict(cfg['object'], flat_frac=0.0)
  scene = scenes.make_scene('train', 99, cfg)
  backend = kin_backend.KinBackend(cfg)
  backend.reset(scene, np.random.default_rng(0))
  policy = expert.ScriptedExpert(expert.ExpertConfig(diversity=False))
  policy.reset(scene, np.random.default_rng(0), backend)
  assert policy._plan(), 'fixture must have an actual expert pick plan'
  assert expert.scene_solvable(scene, cfg)


def test_retry_keeps_requested_object_and_scene_type(monkeypatch):
  env = make_env('train', seed=41)
  attempted = []
  def accept_third(scene, cfg):
    attempted.append((scene.scene_type, scene.obj.shape))
    return len(attempted) == 3
  monkeypatch.setattr(expert, 'scene_solvable', accept_third)
  env.step(zero_action(reset=True))
  assert len(attempted) == 3
  assert len(set(attempted)) == 1, attempted


def test_filter_exhaustion_never_returns_rejected_scene(monkeypatch):
  env = make_env('train', seed=41, scene_filter_max_attempts=2)
  monkeypatch.setattr(expert, 'scene_solvable', lambda scene, cfg: False)
  with pytest.raises(RuntimeError, match='planner rejected'):
    env.step(zero_action(reset=True))


def test_test_splits_keep_raw_layouts(monkeypatch):
  env = make_env('test_constrained', seed=41)
  def unexpected(*args):
    pytest.fail('held-out evaluation must not filter on expert ability')
  monkeypatch.setattr(expert, 'scene_solvable', unexpected)
  env.step(zero_action(reset=True))


def test_object_shape_is_stable_across_layout_retries():
  for seed in range(20):
    shapes = {scenes._Builder('train', seed, {}, attempt).make_object(0, 0, 0).shape
              for attempt in range(10)}
    assert len(shapes) == 1, (seed, shapes)
