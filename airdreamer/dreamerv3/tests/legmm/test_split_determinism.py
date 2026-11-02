import numpy as np

from embodied.envs.legmm import interfaces as I
from embodied.envs.legmm import scenes


def test_same_split_seed_same_scene():
  for split in I.SPLITS:
    lo, _ = I.SPLITS[split]['seeds']
    for seed in range(lo, lo + 5):
      a = scenes.make_scene(split, seed, {})
      b = scenes.make_scene(split, seed, {})
      assert a.fingerprint() == b.fingerprint()
      np.testing.assert_array_equal(a.boxes, b.boxes)
      np.testing.assert_array_equal(a.occupancy, b.occupancy)


def test_different_seeds_differ():
  fps = {scenes.make_scene('train', s, {}).fingerprint() for s in range(20)}
  assert len(fps) == 20


def test_held_out_types():
  held = {int(I.SceneType.CORNER), int(I.SceneType.CUBBY)}
  for split in ('train', 'val', 'test_iid'):
    lo, _ = I.SPLITS[split]['seeds']
    types = {scenes.make_scene(split, s, {}).scene_type for s in range(lo, lo + 150)}
    assert not (types & held), (split, types)
  lo, _ = I.SPLITS['test_constrained']['seeds']
  types = {scenes.make_scene('test_constrained', s, {}).scene_type for s in range(lo, lo + 150)}
  assert held <= types and types <= set(I.SPLITS['test_constrained']['types'])


def test_cubby_and_flat_objects():
  """Top-down grasps are disabled; every scene has at least one graspable free
  direction; for flat boxes only the two narrow faces (+-y of the object) can
  be graspable."""
  lo, _ = I.SPLITS['test_constrained']['seeds']
  n_flat = 0
  for s in range(lo, lo + 60):
    sc = scenes.make_scene('test_constrained', s, {})
    assert not sc.approach[scenes.N_DIRS]
    assert sc.approach.any()
    if sc.obj.shape == 'flat':
      n_flat += 1
      wide = sc.obj.R[:, 1]
      for i in np.where(sc.approach)[0]:
        assert abs(sc.dirs[i] @ wide) > 0.99, (s, i)
  assert 25 <= n_flat <= 55


def test_env_episode_seed_from_split():
  from legmm_utils import make_env, zero_action
  env = make_env('val', seed=3)
  lo, hi = I.SPLITS['val']['seeds']
  for _ in range(3):
    obs = env.step(zero_action(reset=True))
    assert lo <= int(obs['log/seed']) < hi
    assert env.scene.fingerprint() == scenes.make_scene('val', int(obs['log/seed']), env.cfg).fingerprint()


def test_layout_not_tied_to_scene_type():
  """Regression: the layout RNG once shared its stream with the scene-type RNG,
  so e.g. every CUBBY stood against the +x wall."""
  import collections
  sides = collections.defaultdict(collections.Counter)
  for split, stype in (('test_constrained', I.SceneType.CUBBY), ('train', I.SceneType.WALL_BACK)):
    lo, hi = I.SPLITS[split]['seeds']
    n = 0
    for seed in range(lo, hi):
      if scenes.scene_type_of(split, seed) != stype:
        continue
      sc = scenes.make_scene(split, seed, {})
      furn = np.array([b for b, k in zip(sc.boxes, sc.kinds) if k in ('cubby', 'table')])
      c = (furn[:, :2].min(0) + furn[:, 3:5].max(0)) / 2
      side = ('x+' if c[0] > 0 else 'x-') if abs(c[0]) > abs(c[1]) else ('y+' if c[1] > 0 else 'y-')
      sides[stype.name][side] += 1
      n += 1
      if n >= 60:
        break
    assert len(sides[stype.name]) == 4, dict(sides)
    assert max(sides[stype.name].values()) < 0.5 * n, dict(sides)
