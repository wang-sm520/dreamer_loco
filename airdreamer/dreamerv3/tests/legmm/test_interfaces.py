"""Checks for embodied/envs/legmm/interfaces.py (the single source of truth).

The interfaces file ships without a self-test block, so these tests exercise
each documented contract directly.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from embodied.envs.legmm import interfaces as I


def _covers(layout, dim):
  idx = []
  for v in layout.values():
    sl = v[0] if isinstance(v, tuple) else v
    idx += list(range(sl.start, sl.stop))
  return sorted(idx) == list(range(dim))


def test_layouts_are_contiguous():
  assert _covers(I.NAV_LAYOUT, I.NAV_DIM)
  assert _covers(I.MANIP_LAYOUT, I.MANIP_DIM)
  assert _covers(I.PROPRIO_LAYOUT, I.PROPRIO_DIM)
  assert _covers(I.GOAL_LAYOUT, I.GOAL_DIM)
  assert I.DECIMATION == 40 and I.HL_DT == pytest.approx(0.2)
  assert [I.SKILL_ACTION_KEY[s] for s in I.Skill] == ['nav', 'pick', 'place']


def test_spaces():
  obs = I.make_obs_space()
  act = I.make_act_space()
  assert obs['depth_head'].shape == obs['depth_wrist'].shape == (48, 80, 1)
  assert obs['depth_head'].dtype == obs['depth_wrist'].dtype == np.uint8
  assert obs['proprio'].shape == (I.PROPRIO_DIM,)
  assert obs['goal'].shape == (I.GOAL_DIM,)
  for k in I.LABEL_KEYS:
    assert k in obs and not k.startswith('log/')
  assert act['skill'].discrete and act['skill'].high == I.NUM_SKILLS
  for k, d in I.ACTION_DIMS.items():
    assert act[k].shape == (d,)


def test_scale_unscale_roundtrip():
  rng = np.random.default_rng(0)
  for layout, dim in ((I.NAV_LAYOUT, I.NAV_DIM), (I.MANIP_LAYOUT, I.MANIP_DIM)):
    vec = rng.uniform(-1, 1, dim).astype(np.float32)
    phys = I.scale_action(vec, layout)
    for name, (sl, lo, hi) in layout.items():
      assert np.all(phys[name] >= lo - 1e-6) and np.all(phys[name] <= hi + 1e-6)
    back = I.unscale_action(phys, layout, dim)
    np.testing.assert_allclose(back, vec, atol=1e-5)
  # Missing fields map to the middle of the range, i.e. 0 in normalized units.
  np.testing.assert_allclose(I.unscale_action({}, I.NAV_LAYOUT, I.NAV_DIM), 0, atol=1e-6)
  # Out-of-range inputs are clipped.
  assert I.scale_action(np.full(5, 3.0), I.NAV_LAYOUT)['vx'] == pytest.approx(0.8)


def test_mask_inactive_np_batched():
  act = {
      'skill': np.array([0, 1, 2, 1]),
      'nav': np.ones((4, I.NAV_DIM)),
      'pick': np.ones((4, I.MANIP_DIM)),
      'place': np.ones((4, I.MANIP_DIM)),
  }
  out = I.mask_inactive_np(act)
  np.testing.assert_array_equal(out['nav'][:, 0], [1, 0, 0, 0])
  np.testing.assert_array_equal(out['pick'][:, 0], [0, 1, 0, 1])
  np.testing.assert_array_equal(out['place'][:, 0], [0, 0, 1, 0])
  assert out['nav'].dtype == np.float32
  # Idempotent and does not touch the input.
  again = I.mask_inactive_np(out)
  for k in ('nav', 'pick', 'place'):
    np.testing.assert_array_equal(again[k], out[k])
  assert act['nav'].sum() == 4 * I.NAV_DIM


def test_depth_noise_and_quantization():
  cam = I.D435iModel(hole_frac=(0.0, 0.0), edge_hole_prob=0.0)
  rng = np.random.default_rng(0)
  z = np.full((I.DEPTH_H, I.DEPTH_W), 2.0)
  z[:, :10] = 0.1          # closer than min_z -> invalid
  z[:, -10:] = np.inf      # no hit -> max_z
  zn = I.apply_depth_noise(z, rng, cam, a=0.0)
  assert np.all(zn[:, :10] == 0)
  assert np.all(zn[:, -10:] == cam.max_z)
  np.testing.assert_allclose(zn[:, 10:-10], 2.0)
  q = I.depth_to_uint8(zn, cam)
  assert q.shape == (I.DEPTH_H, I.DEPTH_W, 1) and q.dtype == np.uint8
  assert np.all(q[:, :10, 0] == 0) and np.all(q[:, -10:, 0] == 255)
  # Round trip error is at most one quantization step.
  zs = np.linspace(0.01, cam.max_z, 1000)
  back = I.depth_to_uint8(zs[None], cam)[0, :, 0] / 255.0 * cam.max_z
  assert np.max(np.abs(back - zs)) <= cam.max_z / 255 + 1e-9
  assert np.all(I.depth_to_uint8(np.array([[1e-4]]), cam) == 1)
  # Noise grows with distance.
  a = 0.004
  near = I.apply_depth_noise(np.full((200, 200), 1.0), np.random.default_rng(1), cam, a=a)
  far = I.apply_depth_noise(np.full((200, 200), 4.0), np.random.default_rng(1), cam, a=a)
  assert np.std(near) == pytest.approx(a * 1.0, rel=0.1)
  assert np.std(far) == pytest.approx(a * 16.0, rel=0.1)


def test_depth_noise_far_pixels_stay_far():
  # A scene with no hits at all must not produce edge holes at the borders.
  cam = I.D435iModel(hole_frac=(0.0, 0.0), edge_hole_prob=1.0)
  zn = I.apply_depth_noise(np.full((8, 8), np.inf), np.random.default_rng(0), cam, a=0.0)
  assert np.all(zn == cam.max_z)


def test_reward_potential_and_phase_switch():
  cfg = I.RewardCfg()
  prev = I.StepState(phase=0, d_xy_goal=2.0, d_ee_goal=2.5, roll=0.0)
  cur = I.StepState(phase=0, d_xy_goal=1.5, d_ee_goal=2.0, roll=0.0)
  r = I.compute_reward(prev, cur, cfg)
  assert r == pytest.approx(cfg.r_time + cfg.w_nav * 0.5 + cfg.w_ee * 0.5)
  # Phase switch: the goal jumps, the potential difference must be ignored.
  jump = I.StepState(phase=1, d_xy_goal=5.0, d_ee_goal=5.0, roll=0.0, grasp_success=True)
  assert I.compute_reward(cur, jump, cfg) == pytest.approx(cfg.r_time + cfg.r_grasp)
  tilt = I.StepState(phase=0, d_xy_goal=1.5, d_ee_goal=2.0, roll=0.2, collision=True)
  assert I.compute_reward(cur, tilt, cfg) == pytest.approx(
      cfg.r_time + cfg.r_collision - cfg.w_tilt * 0.04)


def test_skill_prior():
  a1 = np.array([0.0, 0.9, 0.9, 0.2])
  a2 = np.array([0.5, 0.5, 0.1, 0.8])
  h = np.array([0.0, 0.0, 1.0, 1.0])
  p = I.skill_prior(a1, a2, h, eps=0.0)
  np.testing.assert_allclose(p.sum(-1), 1.0)
  np.testing.assert_allclose(p[1], [0.1, 0.9, 0.0])
  np.testing.assert_allclose(p[3], [0.2, 0.0, 0.8])
  pe = I.skill_prior(a1, a2, h, eps=0.05)
  assert np.all(pe >= 0.05 / 3 - 1e-9)
  np.testing.assert_allclose(pe.sum(-1), 1.0)


def test_metrics():
  y = np.array([0, 0, 1, 1])
  assert I.auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
  assert I.auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
  assert I.auroc(np.full(4, 0.5), y) == 0.5
  assert np.isnan(I.auroc(np.ones(3), np.ones(3)))
  p = np.array([0.05, 0.05, 0.95, 0.95])
  assert I.ece(p, p) == pytest.approx(0.0)
  assert I.ece(np.full(4, 0.9), np.zeros(4)) == pytest.approx(0.9)
  r = np.zeros(30); rp = np.zeros(30); rp[10] = 2.0
  err = I.reward_head_event_error(r, rp, [10, 2, 28], window=5)
  assert err.shape == (11,) and err[5] == 2.0 and err.sum() == 2.0
  assert np.all(np.isnan(I.reward_head_event_error(r, rp, [1], window=5)))
  skills = np.array([0, 0, 1, 1, 0, 1, 0, 2])
  aff = np.array([0, 0, 0.1, 0, 0, 0.9, 0, 0.05])
  assert I.false_trigger(skills, aff) == 2


def test_splits_hold_out_constrained_types():
  train_types = set(I.SPLITS['train']['types'])
  assert {int(I.SceneType.CORNER), int(I.SceneType.CUBBY)}.isdisjoint(train_types)
  assert {int(I.SceneType.CORNER), int(I.SceneType.CUBBY)} <= set(
      I.SPLITS['test_constrained']['types'])
  ranges = sorted(v['seeds'] for v in I.SPLITS.values())
  for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
    assert a1 <= b0, 'seed ranges of different splits overlap'
