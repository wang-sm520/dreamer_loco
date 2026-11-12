import math
import sys

import numpy as np
import pytest

from embodied.envs.legmm import ag_scenes, ag_shaping as shp
from embodied.run.qlam import QLambda, lambda_of
from ag_utils import act, good_seed, make_ag_env
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tools'))
import legmm_ag_calibrate as cal


def _episode(env, seed, policy_kind=None, steps=None):
  """Rewards and logs of one episode: scripted privileged policy or random navigation."""
  env.forced_seed = seed
  obs = env.step(act(reset=True))
  first = obs
  rows = []
  if policy_kind:
    pol = cal.StancePolicy(policy_kind)
    pol.reset(env)
  rng = np.random.default_rng(seed)
  while True:
    a = pol.act(env, obs) if policy_kind else act(nav=rng.uniform(-1, 1, 3))
    obs = env.step(a)
    rows.append(obs)
    if obs['is_last']:
      return first, rows


def _parts(row, gamma):
  """(distance shaping part, q shaping part, non-shaping part) of one step from the logged pieces."""
  non = -0.01 - 0.1 * row['log/invalid'] - 2.0 * row['log/collision_start']
  if row['is_terminal']:
    non += 20.0 if row['log/success'] else -5.0
  return float(row['rew_task'] - non), float(row['rew_qshape']), float(non)


def _discounted(xs, gamma):
  return sum(gamma ** t * x for t, x in enumerate(xs))


def test_shaping_telescopes_to_minus_phi0_on_termination():
  env = make_ag_env()
  g = env.gamma
  assert abs(g - 332 / 333) < 1e-12
  first, rows = _episode(env, good_seed(), 'privileged')
  assert rows[-1]['is_terminal'] and len(rows) > 5
  pd, pq, non = zip(*[_parts(r, g) for r in rows])
  assert abs(_discounted(pd, g) + float(first['log/phi_dist'])) < 1e-4
  assert abs(_discounted(pq, g) + float(first['log/phi_q'])) < 1e-4
  assert float(first['log/phi_dist']) < -1.0                          # a nontrivial potential at s0
  assert any(abs(x) > 1e-3 for x in pq)


def test_shaping_on_truncation_keeps_the_potential():
  env = make_ag_env(max_episode_steps=20)
  g = env.gamma
  first, rows = _episode(env, good_seed(), 'privileged')               # 20 steps are not enough to arrive
  last = rows[-1]
  assert last['is_last'] and not last['is_terminal']
  T = len(rows)
  pd, pq, _ = zip(*[_parts(r, g) for r in rows])
  assert abs(_discounted(pd, g) - (g ** T * float(last['log/phi_dist']) - float(first['log/phi_dist']))) < 1e-4
  assert abs(_discounted(pq, g) - (g ** T * float(last['log/phi_q']) - float(first['log/phi_q']))) < 1e-4
  assert abs(float(last['log/phi_dist'])) > 0.1                       # it is not zeroed at the cut


@pytest.mark.parametrize('lam', [0.0, 0.37, 1.0])
def test_identity_with_lambda_recomputed_at_training_time(lam):
  from dreamerv3.agent import compose_reward
  env = make_ag_env()
  g = env.gamma
  first, rows = _episode(env, good_seed(), 'privileged')
  data = {'rew_task': np.array([r['rew_task'] for r in rows], np.float32),
          'rew_qshape': np.array([r['rew_qshape'] for r in rows], np.float32),
          'qstar_lam': np.full(len(rows), lam, np.float32)}
  reward = np.asarray(compose_reward(data))
  non = [_parts(r, g)[2] for r in rows]
  expect = _discounted(non, g) - float(first['log/phi_dist']) - lam * float(first['log/phi_q'])
  assert abs(_discounted(reward, g) - expect) < 1e-3
  if lam == 1.0:                                                      # reference reward stored by the env
    np.testing.assert_allclose(reward, [r['reward'] for r in rows], atol=1e-5)
  if lam == 0.0:
    np.testing.assert_allclose(reward, data['rew_task'], atol=1e-7)


def test_q_star_is_smooth_bounded_and_geometric():
  env = make_ag_env()
  env.forced_seed = good_seed()
  env.step(act(reset=True))
  obj = env.scene.obj
  sh = env.scene.obj.center[:2] + 0.55 * np.asarray(env.scene.meta['normal'])
  yaw = math.atan2(-env.scene.meta['normal'][1], -env.scene.meta['normal'][0])
  assert shp.q_star(sh, yaw, obj) == pytest.approx(1.0)               # head-on at a narrow face
  q = [shp.q_star(env.scene.obj.center[:2] + r * np.array([math.cos(a), math.sin(a)]),
                  a + math.pi, obj)
       for a in np.linspace(0, 2 * math.pi, 721) for r in (0.35, 0.58, 0.9)]
  assert min(q) >= 0 and max(q) <= 1 + 1e-12
  ring = [shp.q_star(env.scene.obj.center[:2] + 0.58 * np.array([math.cos(a), math.sin(a)]), a + math.pi, obj)
          for a in np.linspace(0, 2 * math.pi, 7201)]
  assert max(abs(np.diff(ring))) < 0.02                               # no jumps
  # facing a wide face is bad for flat boxes
  wide = obj.R[:2, 0]
  assert shp.q_star(obj.center[:2] + 0.55 * wide, math.atan2(-wide[1], -wide[0]), obj) < 0.05
  # cylinders have no orientation
  assert shp.q_star(sh, yaw, obj) >= shp.q_star(sh, yaw + 1.0, obj)
  assert shp.dist_band(0.45) == 0.0 and shp.dist_band(0.70) == 0.0 and shp.dist_band(0.8) == pytest.approx(0.1)


def test_qlambda_schedule():
  q = QLambda('anneal', window=10, lo=0.2, hi=0.5)
  assert q.lam == 1.0
  for _ in range(9):
    q.update(1.0)
  assert q.lam == 1.0                                                  # window not full yet
  q.update(1.0)
  assert q.lam == 0.0                                                  # 100% success
  for _ in range(10):
    q.update(0.0)
  assert q.lam == 0.0                                                  # never goes back up
  q = QLambda('anneal', window=10, lo=0.2, hi=0.5)
  for i in range(10):
    q.update(1.0 if i < 3 else 0.0)                                    # 30% success
  assert q.lam == pytest.approx((0.5 - 0.3) / 0.3)
  lams = []
  q = QLambda('anneal', window=20, lo=0.2, hi=0.5)
  rng = np.random.default_rng(0)
  for p in np.linspace(0.0, 0.8, 200):
    q.update(float(rng.random() < p))
    lams.append(q.lam)
  assert all(b <= a + 1e-12 for a, b in zip(lams, lams[1:])) and lams[-1] < lams[30]
  assert lambda_of(0.2, 0.2, 0.5) == 1.0 and lambda_of(0.5, 0.2, 0.5) == 0.0
  fixed, none = QLambda('fixed'), QLambda('none')
  for _ in range(300):
    fixed.update(1.0)
    none.update(0.0)
  assert fixed.lam == 1.0 and none.lam == 0.0
  with pytest.raises(ValueError):
    QLambda('bogus')


def test_qlambda_counts_truncation_as_failure_and_reads_log_success():
  q = QLambda('anneal', window=2, lo=0.2, hi=0.5)
  q.on_step({'is_last': False, 'log/success': np.float32(1.0)})
  assert q.episodes == 0
  q.on_step({'is_last': True, 'log/success': np.float32(0.0)})         # a timeout reports success 0
  q.on_step({'is_last': True, 'log/success': np.float32(1.0)})
  assert q.success_rate == 0.5 and q.lam == 0.0
