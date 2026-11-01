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


def test_cubby_blocks_top_and_opens_to_the_room():
  lo, _ = I.SPLITS['test_constrained']['seeds']
  n_cubby = 0
  for s in range(lo, lo + 60):
    sc = scenes.make_scene('test_constrained', s, {})
    if sc.scene_type != I.SceneType.CUBBY:
      continue
    n_cubby += 1
    assert not sc.approach[scenes.N_DIRS], 'cubby must block top-down access'
    # Cubbies stand against the nearest wall and open away from it; the bin
    # pointing out of the opening must be free.
    # (Other bins may be formally free too: the brief's 0.25 m corridor rule
    # does not reach the side/back panels of a deep or wide cubby.)
    near_wall = np.argmax(np.abs(sc.obj.center[:2]))
    n = np.zeros(2)
    n[near_wall] = -np.sign(sc.obj.center[near_wall])
    i = int(np.argmax(scenes.APPROACH_DIRS[:scenes.N_DIRS, :2] @ n))
    assert sc.approach[i], (s, sc.approach)
  assert n_cubby >= 5


def test_env_episode_seed_from_split():
  from legmm_utils import make_env, zero_action
  env = make_env('val', seed=3)
  lo, hi = I.SPLITS['val']['seeds']
  for _ in range(3):
    obs = env.step(zero_action(reset=True))
    assert lo <= int(obs['log/seed']) < hi
    assert env.scene.fingerprint() == scenes.make_scene('val', int(obs['log/seed']), env.cfg).fingerprint()
