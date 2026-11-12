import copy
import math

import numpy as np

from embodied.envs.legmm import ag_scenes, scenes as sc
from ag_utils import act, make_ag_env

SEEDS = range(20000, 20200)


def test_scene_determinism():
  for seed in list(SEEDS)[:30]:
    a, b = ag_scenes.make_ag_scene('test', seed), ag_scenes.make_ag_scene('test', seed)
    assert a.fingerprint() == b.fingerprint()
    np.testing.assert_array_equal(a.boxes, b.boxes)
    assert a.meta == b.meta
  prints = {ag_scenes.make_ag_scene('test', s).fingerprint() for s in list(SEEDS)[:30]}
  assert len(prints) == 30                                  # seeds differ
  assert ag_scenes.make_ag_scene('train', 5).fingerprint() != ag_scenes.make_ag_scene('val', 5).fingerprint()


def test_stratum_matches_scene_and_fractions():
  for seed in SEEDS:
    shape, blocked, kind = ag_scenes.stratum_of('test', seed)
    m = ag_scenes.make_ag_scene('test', seed).meta
    assert (m['shape'], m['blocked'], m['blocker']) == (shape, blocked, kind)
  strata = [ag_scenes.stratum_of('test', s) for s in range(20000, 21000)]
  assert abs(np.mean([s[1] for s in strata]) - 0.5) < 0.05
  assert abs(np.mean([s[0] == 'flat' for s in strata]) - 0.7) < 0.05


def test_filter_keeps_stratum_and_returns_solvable():
  env = make_ag_env(split='train', scene_filter='solvable', seed=3)
  rejected = 0
  for _ in range(40):
    r = copy.deepcopy(env.rng)
    seed0 = int(r.integers(*ag_scenes.AG_SPLITS['train']))
    scene = env._sample_scene()
    want = ag_scenes.stratum_of('train', seed0)[:2]
    assert (scene.meta['shape'], scene.meta['blocked']) == want
    assert scene.meta['solvable'] is True
    rejected += env.scene_rejected
  assert rejected > 0                                       # the filter did something
  # the test split is not filtered and records solvability, including unsolvable scenes
  test = make_ag_env(split='test', seed=3)
  flags = []
  for s in range(20000, 20060):
    test.forced_seed = s
    obs = test.step(act(reset=True))
    flags.append(float(obs['log/solvable']))
  assert set(flags) == {0.0, 1.0}


def test_scene_ranges():
  yaws = []
  for seed in range(20000, 20400):
    sce = ag_scenes.make_ag_scene('test', seed)
    m = sce.meta
    sx, sy, th = m['table']
    assert 0.6 <= sx <= 1.2 and 0.6 <= sy <= 1.0 and 0.50 <= th <= 0.65
    assert 0.05 <= m['edge_dist'] <= 0.25
    tc = np.array(m['table_center'])
    rel = sce.obj.center[:2] - tc
    dists = [sy / 2 - rel[1], sx / 2 - rel[0], sy / 2 + rel[1], sx / 2 + rel[0]]
    assert abs(min(dists) - m['edge_dist']) < 1e-9 and int(np.argmin(dists)) == m['edge']
    assert min(dists) > 0
    if sce.obj.shape == 'flat':
      yaws.append(sce.obj.yaw % math.pi)
      assert 0.03 <= 2 * sce.obj.half[0] <= 0.05 and 0.10 <= 2 * sce.obj.half[1] <= 0.14
    assert sum(k == 'table' for k in sce.kinds) == 1          # one target on one table
  hist = np.histogram(yaws, bins=6, range=(0, math.pi))[0]
  assert hist.min() > 0.5 * hist.mean()                       # orientation covers 0-180 deg


def test_blocked_edge_blocks_footprint_and_camera():
  for seed in range(20000, 20150):
    sce = ag_scenes.make_ag_scene('test', seed)
    m = sce.meta
    base = ag_scenes.head_on_stance(sce, 0.70)
    hit = sc.geo.obb_hits_boxes(base[:2], base[2], (sc.FOOT_HALF[0], sc.FOOT_HALF[1]), sce.boxes)
    kinds_hit = {sce.kinds[i] for i in np.where(hit)[0]}
    if m['blocked']:
      # a stance behind a room wall lies outside the room: unusable as well
      c, s_ = math.cos(base[2]), math.sin(base[2])
      ex = abs(c) * sc.FOOT_HALF[0] + abs(s_) * sc.FOOT_HALF[1]
      ey = abs(s_) * sc.FOOT_HALF[0] + abs(c) * sc.FOOT_HALF[1]
      outside = abs(base[0]) + ex > sce.room[0] / 2 or abs(base[1]) + ey > sce.room[1] / 2
      assert (kinds_hit & {'blocker', 'wall'}) or outside, (seed, kinds_hit)
      if m['blocker'] == 'furniture':
        assert 'blocker' in kinds_hit, (seed, kinds_hit)
      assert ('blocker' in sce.kinds) == (m['blocker'] == 'furniture')
    else:
      assert not kinds_hit and 'blocker' not in sce.kinds, (seed, kinds_hit)
  # the cameras see the blocker: a head camera looking at it reports its face distance
  env = make_ag_env()
  for seed in range(20000, 20100):
    env.forced_seed = seed
    env.step(act(reset=True))
    if env.scene.meta['blocker'] == 'furniture':
      break
  b, n = env.scene.boxes[env.scene.kinds.index('blocker')], np.asarray(env.scene.meta['normal'])
  tc = np.array(env.scene.meta['table_center'])
  face = np.where(n != 0, np.where(n > 0, b[3:5], -b[:2]), 0.0)
  dist_out = 1.2
  pos = tc + n * (np.abs(face).max() + dist_out)
  s = env.backend.state
  s.x, s.y, s.yaw = pos[0], pos[1], math.atan2(-n[1], -n[0])
  z = env.backend.render_true('head')
  centre = z[:, 36:44]
  expect = dist_out - env.backend.cams['head']['mount'][0]
  assert abs(np.min(centre) - expect) < 0.03, (np.min(centre), expect)


def test_start_is_free_and_target_out_of_reach():
  env = make_ag_env()
  for seed in range(20000, 20120):
    env.forced_seed = seed
    env.step(act(reset=True))
    s = env.backend.state
    assert not sc.geo.obb_hits_boxes((s.x, s.y), s.yaw, sc.FOOT_HALF, env.scene.boxes).any()
    g = env._gate()
    assert g['dist'] > 1.0 and not g['ok']


def test_knife_edge_scene_is_rejected_with_margin_and_accepted_without():
  # Seed 20148 (test split, unblocked flat box) has a skill plan only at stances on the very edge of the gate:
  # solvable with margin 0, not with the 1 cm / 3 deg margin. The environment records and filters with the margin.
  from embodied.envs.legmm import ag_skill as sk
  env = make_ag_env(split='test')
  scene = ag_scenes.make_ag_scene('test', 20148, env.cfg)
  env.scene = scene
  assert sk.solvable_scene(env._shadow(), scene, env.skill_cfg, margin=0.0)
  assert not sk.solvable_scene(env._shadow(), scene, env.skill_cfg, margin=0.01)
  assert env.solvable_margin == 0.01
  # the stances found without margin sit on the boundary (rho 0.45/0.70 or bearing 45)
  fs = sk.feasible_stances(env._shadow(), scene, env.skill_cfg, margin=0.0)
  assert fs and all(f['rho'] in (0.45, 0.70) or abs(f['bearing']) == 45 for f in fs)
  # recorded solvability uses the margin ...
  env.forced_seed = 20148
  env.step(act(reset=True))
  assert env.scene.meta['solvable'] is False
  # ... and so does the train filter (the scene is resampled), unlike an env with margin 0
  train = make_ag_env(split='train', scene_filter='solvable', seed=1, solvable_margin=0.0)
  assert train.solvable_margin == 0.0
  strict = make_ag_env(split='train', scene_filter='solvable', seed=1)
  for e, want in ((train, 0.0), (strict, 0.01)):
    for _ in range(30):
      e.step(act(reset=True))
      sc_ = e.scene
      assert sk.solvable_scene(e._shadow(), sc_, e.skill_cfg, margin=want)
