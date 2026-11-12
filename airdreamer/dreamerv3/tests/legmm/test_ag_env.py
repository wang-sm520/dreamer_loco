import copy
import math

import numpy as np
import pytest

from embodied.envs.legmm import ag_shaping as shp, ag_skill as sk, interfaces as I
from ag_utils import act, env_at_stance, good_seed, make_ag_env, place, set_est, shoulder_dist
from embodied.envs.legmm import ag_scenes


def _with_shoulder_dist(env, rho):
  place(env, ag_scenes.head_on_stance(env.scene, rho))


@pytest.mark.parametrize('rho,ok', [(0.4499, False), (0.4501, True), (0.6999, True), (0.7001, False)])
def test_gate_distance_boundaries(rho, ok):
  env, _ = env_at_stance()
  _with_shoulder_dist(env, rho)
  g = env._gate()                                                   # est pinned to the true centre by env_at_stance
  assert abs(g['dist'] - rho) < 1e-6 and g['dist_ok'] == ok and g['ok'] == ok


@pytest.mark.parametrize('deg,ok', [(44.0, True), (46.0, False), (-44.0, True), (-46.0, False)])
def test_gate_bearing_boundaries(deg, ok):
  env, _ = env_at_stance()
  s = env.backend.state
  sh = env._shoulder()
  yaw = s.yaw + math.radians(deg)                                   # turn about the shoulder, keep its position
  dx, dy = env.backend.shoulder_b[:2]
  place(env, [sh[0] - math.cos(yaw) * dx + math.sin(yaw) * dy, sh[1] - math.sin(yaw) * dx - math.cos(yaw) * dy, yaw])
  g = env._gate()
  assert abs(math.degrees(abs(g['bearing'])) - abs(deg)) < 1e-6 and g['bearing_ok'] == ok and g['ok'] == ok


@pytest.mark.parametrize('support,ok', [(0.49, False), (0.51, True), (0.64, True), (0.66, False)])
def test_gate_height_boundaries(support, ok):
  env, _ = env_at_stance()
  set_est(env, offset=(0.0, 0.0, support + env.scene.obj.half[2] - env.backend.state.obj_center[2]))
  g = env._gate()                                                   # the gate reads the ESTIMATED height
  assert g['height_ok'] == ok and g['ok'] == ok and abs(g['support_z'] - support) < 1e-9


@pytest.mark.parametrize('speed,ok', [(0.29, True), (0.31, False)])
def test_gate_speed_boundaries(speed, ok):
  env, _ = env_at_stance(speed=speed)
  g = env._gate()
  assert g['speed_ok'] == ok and g['ok'] == ok


def test_invalid_trigger_continues_and_applies_nav():
  env = make_ag_env()
  env.forced_seed = good_seed()
  env.step(act(reset=True))
  s = env.backend.state
  x0, y0 = s.x, s.y
  assert not env._gate()['ok']
  obs = env.step(act(nav=(1.0, 0.0, 0.0), trigger=1))
  assert not obs['is_last'] and not obs['is_terminal'] and obs['log/invalid'] == 1 and obs['log/triggered'] == 1
  assert math.hypot(env.backend.state.x - x0, env.backend.state.y - y0) > 0.01      # nav was executed
  # same step without the trigger differs only by the -0.1 invalid penalty
  twin = make_ag_env()
  twin.forced_seed = good_seed()
  twin.step(act(reset=True))
  ref = twin.step(act(nav=(1.0, 0.0, 0.0), trigger=0))
  assert abs((ref['rew_task'] - obs['rew_task']) - 0.1) < 1e-5


def test_valid_trigger_terminates_with_success_and_pre_grasp_frame():
  env, _ = env_at_stance()
  obs = env.step(act(trigger=1))
  assert obs['is_last'] and obs['is_terminal'] and obs['log/success'] == 1
  assert I.AG_TERM_REASONS[int(obs['log/term_reason'])] == 'success'
  assert obs['log/skill_steps'] > 0 and obs['log/skill_s'] > 0
  s = env.backend.state
  assert not s.holding and s.ee_b is not None                      # the frame is the pre-grasp one
  np.testing.assert_allclose(s.ee_b, env.backend.shoulder_b + env.backend.stow_rel, atol=1e-6)
  assert obs['rew_task'] > 10                                       # +20 minus time and shaping
  nxt = env.step(act())                                             # the next call starts a new episode
  assert nxt['is_first']


def test_skill_failure_is_minus_F_and_terminal():
  env, _ = env_at_stance()
  s = env.backend.state
  # Face the wide side of the box: gate passes (distance/bearing), but no plan exists.
  R = env.scene.obj.R
  wide_dir = R[:2, 0]                                               # thickness axis points at a wide face
  c = env.scene.obj.center[:2]
  sh = c + 0.55 * wide_dir
  yaw = math.atan2(-wide_dir[1], -wide_dir[0])
  dx, dy = env.backend.shoulder_b[:2]
  base = [sh[0] - math.cos(yaw) * dx + math.sin(yaw) * dy, sh[1] - math.sin(yaw) * dx - math.cos(yaw) * dy, yaw]
  place(env, base)
  assert env._gate()['ok']
  obs = env.step(act(trigger=1))
  assert obs['is_last'] and obs['is_terminal'] and obs['log/success'] == 0
  assert I.AG_TERM_REASONS[int(obs['log/term_reason'])] == 'skill_fail' and obs['log/no_plan'] == 1
  assert obs['rew_task'] < -4.9 + 0.5 and obs['rew_task'] > -5.2 - 1.0


def test_slid_out_after_settle_is_invalid_and_episode_continues():
  # Moving away from the object at 0.28 m/s while the shoulder is 0.69 m away: the gate passes (speed <= 0.3,
  # distance <= 0.70) and settle carries the base out of the envelope.
  env, _ = env_at_stance(rho=0.69, speed=-0.28)
  assert env._gate()['ok']
  obs = env.step(act(trigger=1))
  assert obs['log/slid_out'] == 1 and obs['log/invalid'] == 1 and obs['log/settle_steps'] > 0
  assert not obs['is_last'] and not obs['is_terminal']
  assert shoulder_dist(env) > 0.70
  assert not env.step(act())['is_first']                            # the episode goes on
  # control: the same speed towards the object stays inside and terminates
  env2, _ = env_at_stance(rho=0.60, speed=0.28)
  obs2 = env2.step(act(trigger=1))
  assert obs2['is_terminal'] and obs2['log/slid_out'] == 0 and obs2['log/settle_steps'] > 0


def test_settle_stops_the_base_and_is_one_decision_step():
  env, _ = env_at_stance(rho=0.60, speed=0.28)
  step0 = env._step
  obs = env.step(act(trigger=1))
  assert env._step == step0 + 1
  assert abs(obs['log/settle_s'] - obs['log/settle_steps'] * 0.2) < 1e-6 and obs['log/settle_s'] <= 1.5
  assert math.hypot(*env.backend.state.v[:2]) < 0.05 and abs(env.backend.state.v[2]) < 0.1


def test_truncation_flags_and_no_failure_reward():
  env = make_ag_env(max_episode_steps=4)
  env.forced_seed = good_seed()
  env.step(act(reset=True))
  for i in range(4):
    obs = env.step(act())
    assert obs['is_last'] == (i == 3)
  assert obs['is_last'] and not obs['is_terminal']
  assert I.AG_TERM_REASONS[int(obs['log/term_reason'])] == 'timeout' and obs['log/success'] == 0
  # truncation: only the time cost plus potential shaping, no -F
  assert obs['rew_task'] > -0.5


def test_observation_has_no_yaw_place_or_privileged_keys_and_ignores_true_yaw():
  env, obs = env_at_stance()
  keys = set(env.obs_space)
  assert not [k for k in keys if any(w in k for w in ('yaw', 'place', 'priv', 'q_star'))
              if k != 'log/q_star']
  assert I.make_ag_obs_space()['goal'].shape == (I.AG_GOAL_DIM,) == (6,)
  snap = env.snapshot()
  a = env.step(act(nav=(0.2, 0.0, 0.1)))
  q_a = float(a['log/phi_q'])
  env.restore(snap)
  env.scene.obj.yaw += math.radians(60)
  env.backend.state.obj_R = env.scene.obj.R.copy()
  b = env.step(act(nav=(0.2, 0.0, 0.1)))
  np.testing.assert_array_equal(a['goal'], b['goal'])
  np.testing.assert_array_equal(a['proprio'], b['proprio'])
  assert abs(float(b['log/phi_q']) - q_a) > 1e-3                    # the privileged q* did react


def test_skill_cannot_move_base_or_height():
  env, _ = env_at_stance()
  sh = env._shadow()
  import copy as _c
  sh.state = _c.deepcopy(env.backend.state)
  sh.rng = np.random.default_rng(0)
  sh.state.v = np.array([0.3, 0.1, 0.5])                           # residual velocity must not leak
  cmds = []
  orig = sh.step
  sh.step = lambda cmd: (cmds.append(cmd), orig(cmd))[1]
  pose0 = (sh.state.x, sh.state.y, sh.state.yaw, sh.state.h)
  res = sk.run_skill(sh, env.skill_cfg)
  assert res.success and cmds
  assert all(not np.any(c['base']) and c['height'] == pose0[3] for c in cmds)
  s = sh.state
  assert abs(s.x - pose0[0]) < 1e-9 and abs(s.y - pose0[1]) < 1e-9 and abs(s.h - pose0[3]) < 1e-12
  assert abs(math.atan2(math.sin(s.yaw - pose0[2]), math.cos(s.yaw - pose0[2]))) < 1e-9
  # a backend that moves the base is caught
  sh.state = copy.deepcopy(env.backend.state)
  plan, _ = sk.plan_skill(sh, env.skill_cfg)
  ex = sk.SkillExecutor(sh, plan, env.skill_cfg)
  sh.state.x += 0.01
  assert not ex.base_fixed()


def test_feasible_stance_succeeds_and_wide_face_fails_in_execution():
  env, _ = env_at_stance()
  sh = env._shadow()
  sh.state = copy.deepcopy(env.backend.state)
  sh.rng = np.random.default_rng(0)
  plan, _ = sk.plan_skill(sh, env.skill_cfg)
  assert plan is not None
  rot = copy.deepcopy(sh.state)
  # Execute the same plan against a box turned by 90 deg: the wide face is now between the fingers.
  sh.state.obj_R = sh.state.obj_R @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
  res = sk.SkillExecutor(sh, plan, env.skill_cfg).run()
  assert not res.success and res.reason == 'grasp' and res.executed
  sh.state = rot
  assert sk.SkillExecutor(sh, plan, env.skill_cfg).run().success
  # and planning from a wide-face stance finds nothing
  sh.state = copy.deepcopy(env.backend.state)
  sh.state.obj_R = sh.state.obj_R @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
  assert sk.plan_skill(sh, env.skill_cfg)[0] is None


def test_counterfactual_leaves_state_and_rng_unchanged():
  env, _ = env_at_stance(rho=0.60, speed=0.2)
  env.step(act(nav=(0.0, 0.0, 0.0)))                                # settle a bit, still inside the gate
  place(env, ag_scenes.head_on_stance(env.scene, 0.60), 0.2)

  def fingerprint(e):
    s = e.backend.state
    return (copy.deepcopy(s.__dict__.copy()), e.rng.bit_generator.state, e.backend.rng.bit_generator.state,
            e.goal.rng.bit_generator.state, e.label_rng.bit_generator.state, e._step, e._done,
            copy.deepcopy(e.goal.state()), e._phis, e._est.copy())

  def same(a, b):
    assert a[1:5] == b[1:5] and a[5:7] == b[5:7] and a[7][0] == b[7][0] and a[8] == b[8]
    np.testing.assert_array_equal(a[9], b[9])
    for k in a[0]:
      np.testing.assert_array_equal(np.asarray(a[0][k], dtype=object) if k == 'arm_queue' else a[0][k],
                                    np.asarray(b[0][k], dtype=object) if k == 'arm_queue' else b[0][k])

  twin = copy.deepcopy(env)
  before = fingerprint(env)
  cf = env.counterfactual()
  assert cf['valid'] and cf['success']
  same(before, fingerprint(env))
  assert env.backend.state.v[0] == 0.2                              # the live base did not settle
  # the continuation is identical to an environment that never computed the label
  a, b = env.step(act(nav=(0.1, 0, 0))), twin.step(act(nav=(0.1, 0, 0)))
  for k in ('depth_head', 'depth_wrist', 'proprio', 'goal', 'reward'):
    np.testing.assert_array_equal(a[k], b[k])
  # outside the gate no label exists
  far = make_ag_env()
  far.forced_seed = good_seed()
  far.step(act(reset=True))
  assert not far.counterfactual()['valid']


def test_labels_inside_gate_only_and_do_not_change_the_trajectory():
  def run(rate):
    env, _ = env_at_stance(rho=0.60, speed=0.0, labels={'rate': rate, 'terminal': False, 'seed': 0})
    outs = [env.step(act(nav=(0.0, 0.0, 0.0))) for _ in range(6)]
    return outs
  with_labels, without = run(1.0), run(0.0)
  assert all(o['label_feas_mask'] == 1 and o['label_feas'] == 1 for o in with_labels)
  assert all(o['label_feas_mask'] == 0 for o in without)
  for a, b in zip(with_labels, without):
    for k in ('depth_head', 'proprio', 'goal', 'reward'):
      np.testing.assert_array_equal(a[k], b[k])
  env = make_ag_env(labels={'rate': 1.0, 'terminal': False, 'seed': 0})
  env.forced_seed = good_seed()
  obs = env.step(act(reset=True))
  assert obs['label_feas_mask'] == 0                                # outside the gate


def test_version_tag_in_metadata():
  env = make_ag_env()
  assert env.metadata['version'] == 'legmm_ag_v1' == I.AG_VERSION
  assert env.metadata['surrogate']['kind'] == 'kin'


def test_nav_only_acts_on_nav_and_trigger_is_discrete():
  space = I.make_ag_act_space()
  assert set(space) == {'reset', 'nav', 'trigger'}
  assert space['trigger'].discrete and space['trigger'].shape == () and not space['nav'].discrete
  assert space['nav'].shape == (3,)


# ---------------------------------------------------------------- gate reads the estimate, not the truth

def _radial(env):
  """Unit vector (world, xy) from the shoulder to the true object centre."""
  d = env.backend.state.obj_center[:2] - env._shoulder()
  return d / np.linalg.norm(d)


def test_gate_follows_the_estimate_when_the_truth_moves():
  env, _ = env_at_stance(rho=0.55)
  est0 = env._est.copy()
  g0 = env._gate()
  assert g0['ok']
  # true object far away (est held fixed): the gate does not notice, the true-position gate does
  s = env.backend.state
  s.obj_center = s.obj_center.copy()
  s.obj_center[:2] += 0.5 * _radial(env)
  g1 = env._gate()
  np.testing.assert_array_equal(env._est, est0)
  assert g1 == g0 and g1['ok']
  assert not env._gate(true=True)['ok'] and env._gate(true=True)['dist'] > 1.0
  # an invalid result must not reveal the true distance: the reported distance is the estimated one
  s.obj_center[:2] += 0.5 * _radial(env)
  assert abs(env._gate()['dist'] - g0['dist']) < 1e-12


def test_gate_follows_the_estimate_when_the_estimate_moves():
  env, _ = env_at_stance(rho=0.55)
  true0 = env.backend.state.obj_center.copy()
  assert env._gate()['ok']
  u = _radial(env)
  set_est(env, offset=(*(0.3 * u), 0.0))                            # estimate says 0.85 m: outside
  g = env._gate()
  np.testing.assert_array_equal(env.backend.state.obj_center, true0)
  assert not g['ok'] and not g['dist_ok'] and abs(g['dist'] - 0.85) < 1e-6
  assert env._gate(true=True)['ok']
  x0 = env.backend.state.x
  obs = env.step(act(nav=(1.0, 0.0, 0.0), trigger=1))               # estimate-outside: invalid although truth is inside
  assert obs['log/invalid'] == 1 and not obs['is_last'] and env.backend.state.x != x0
  assert obs['log/trig_gate_mismatch'] == 1 and obs['log/trig_gate_true_ok'] == 1
  # and the other way round: estimate inside, truth outside -> the trigger is accepted (skill runs from truth)
  env2, _ = env_at_stance(rho=0.55)
  s2 = env2.backend.state
  s2.obj_center = s2.obj_center.copy()
  s2.obj_center[:2] += 0.4 * _radial(env2)
  obs2 = env2.step(act(trigger=1))
  assert obs2['log/invalid'] == 0 and obs2['is_last'] and obs2['is_terminal']
  assert obs2['log/trig_gate_mismatch'] == 1 and obs2['log/trig_gate_true_ok'] == 0
  assert obs2['log/success'] == 0                                  # the skill did not succeed from the true state


def test_post_settle_recheck_uses_the_estimate():
  # Estimate 0.69 m, truth 0.60 m, moving away at 0.28 m/s: settle slides the base; the re-check sees the
  # estimated distance exceed 0.70 although the true one stays inside.
  env, _ = env_at_stance(rho=0.60, speed=-0.28)
  set_est(env, offset=(*(0.09 * _radial(env)), 0.0))
  assert env._gate()['ok']
  obs = env.step(act(trigger=1))
  assert obs['log/slid_out'] == 1 and obs['log/invalid'] == 1 and not obs['is_last']
  assert env._gate(true=True)['dist'] <= 0.70          # the truth stayed inside; only the estimate left the band
  # reverse: truth would slide out, the estimate (0.60) stays inside -> the skill runs
  env2, _ = env_at_stance(rho=0.69, speed=-0.28)
  set_est(env2, offset=(*(-0.09 * _radial(env2)), 0.0))
  assert env2._gate()['ok']
  obs2 = env2.step(act(trigger=1))
  assert obs2['log/slid_out'] == 0 and obs2['is_last'] and obs2['log/settle_steps'] > 0
  assert env2._gate(true=True)['dist'] > 0.70


def test_gate_reuses_the_estimate_of_the_observation_and_draws_no_extra_noise():
  def run(extra_gate_calls):
    env, _ = env_at_stance(exact=False, rho=0.55)
    out = []
    for _ in range(12):
      for _ in range(extra_gate_calls):
        env._gate()
        env._gate(true=True)
      o = env.step(act(nav=(0.1, 0.0, 0.0)))
      out.append((o['goal'].copy(), env._est.copy(), env.goal.rng.bit_generator.state))
    return out
  a, b = run(0), run(5)
  for (g1, e1, r1), (g2, e2, r2) in zip(a, b):
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(e1, e2)
    assert r1 == r2
  # the estimate the gate reads is exactly the one behind the observation (latency / dropout included)
  env, _ = env_at_stance(exact=False, rho=0.55, goal_provider=dict(pos_sigma=0.03, latency_steps=[2, 2], dropout=0.5))
  for _ in range(15):
    o = env.step(act(nav=(0.0, 0.0, 0.0)))
    s = env.backend.state
    d = env._est - env.backend.base_pos()
    assert abs(o['goal'][3] - math.hypot(d[0], d[1])) < 1e-4 and abs(o['goal'][4] - d[2]) < 1e-4
    np.testing.assert_array_equal(env._est, env.goal.last)


def test_snapshot_restore_and_counterfactual_carry_the_estimate():
  env, _ = env_at_stance(rho=0.60)
  snap = env.snapshot()
  est0 = env._est.copy()
  np.testing.assert_array_equal(snap['est'], est0)
  env.step(act(nav=(0.2, 0.0, 0.1)))
  env.restore(snap)
  np.testing.assert_array_equal(env._est, est0)
  assert env._est is not snap['est']
  # a counterfactual from a snapshot uses the snapshot's estimate; the current one is not touched
  set_est(env, offset=(*(0.3 * _radial(env)), 0.0))                  # now outside the gate
  est1 = env._est.copy()
  assert not env.counterfactual()['valid']
  cf = env.counterfactual(snap)
  assert cf['valid'] and cf['success']
  np.testing.assert_array_equal(env._est, est1)
  # labels (inside the gate by the estimate) are side-effect free: same trajectory with and without
  def run(rate):
    e, _ = env_at_stance(rho=0.60, labels={'rate': rate, 'terminal': False, 'seed': 0})
    return [e.step(act(nav=(0.0, 0.0, 0.0))) for _ in range(6)], e
  (la, ea), (lb, eb) = run(1.0), run(0.0)
  assert all(o['label_feas_mask'] == 1 for o in la[1:])
  np.testing.assert_array_equal(ea._est, eb._est)
  assert ea.goal.rng.bit_generator.state == eb.goal.rng.bit_generator.state
  for a, b in zip(la, lb):
    np.testing.assert_array_equal(a['goal'], b['goal'])


def test_gate_mismatch_logs_are_consistent_and_nonzero_at_the_boundary():
  env, _ = env_at_stance(exact=False, rho=0.70)                     # on the boundary: noise decides
  mism, oks = 0, set()
  for _ in range(60):
    o = env.step(act(nav=(0.0, 0.0, 0.0)))
    assert o['log/gate_mismatch'] == float(o['log/gate_ok'] != o['log/gate_ok_true'])
    assert abs(o['log/shoulder_dist_true'] - env._gate(true=True)['dist']) < 1e-5
    assert abs(o['log/shoulder_dist'] - env._gate()['dist']) < 1e-5
    mism += int(o['log/gate_mismatch'])
    oks.add(float(o['log/gate_ok']))
  assert mism > 0 and oks == {0.0, 1.0}
  far = make_ag_env()
  far.forced_seed = good_seed()
  o = far.step(act(reset=True))
  assert o['log/gate_mismatch'] == 0 and o['log/gate_ok_true'] == 0 and o['log/shoulder_dist_true'] > 1.0


def _scripted_contact(env, pattern):
  """Make the base report a collision on decision step i (0-based after reset) iff pattern[i]."""
  def hit(s=None):
    return bool(pattern[min(env._step, len(pattern) - 1)])
  env.backend.base_collides = hit


def test_collision_is_charged_once_per_contact_not_per_step():
  env, _ = env_at_stance(rho=0.80)
  pattern = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]                          # a 4-step contact, a gap, a separate 2-step contact
  _scripted_contact(env, [0] * env._step + pattern)
  phi = env._phis[0]
  idle = -0.01 + shp.shaping_diff(phi, phi, env.gamma, False)       # no motion: only time + (gamma-1) * phi
  rows = [env.step(act()) for _ in pattern]
  assert [int(r['log/collision']) for r in rows] == pattern         # contact steps are still logged
  assert [int(r['log/collision_start']) for r in rows] == [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
  assert sum(r['log/collision_start'] for r in rows) == 2 and sum(r['log/collision'] for r in rows) == 6
  for r, start in zip(rows, [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]):
    assert not r['is_last'] and not r['is_terminal']                # collisions do not terminate
    assert abs(float(r['rew_task']) - (idle - 2.0 * start)) < 1e-5


def test_contact_state_resets_and_is_part_of_snapshot_restore():
  env, _ = env_at_stance(rho=0.80)
  _scripted_contact(env, [1] * 1000)
  assert env.step(act())['log/collision_start'] == 1
  assert env.step(act())['log/collision_start'] == 0
  snap = env.snapshot()
  assert snap['env']['contact'] is True
  env._contact = False
  env.restore(snap)
  assert env._contact is True                                       # restored in-contact: the next contact step is not a start
  assert env.step(act())['log/collision_start'] == 0
  env.step(act(reset=True))
  assert env._contact is False                                      # reset clears it
