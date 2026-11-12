"""Contract boundaries, legacy identity and nontrivial end-to-end consistency."""
import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from embodied.envs.legmm import ag_contract as ct, ag_skill as sk, ag_scenes, scenes as sc
from ag_utils import act, env_at_stance, make_ag_env, place, set_est, good_seed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tools'))
import legmm_ag_calibrate as cal
from drone_signature import load_config


def isolated(rho=.6, bearing=0., support=.55, gap=.2):
  env, _ = env_at_stance()
  b = env.backend
  b.state.x = b.state.y = b.state.yaw = 0.
  env.scene.obj.support_z = support
  angle = math.radians(bearing)
  b.state.obj_center = np.array([b.shoulder_b[0]+rho*math.cos(angle), rho*math.sin(angle), support+.07])
  # Static obstacle alongside the footprint, low enough that it is not an arm obstacle.
  b._static = np.array([[-.1, sc.FOOT_HALF[1]+gap, 0., .1, sc.FOOT_HALF[1]+gap+.1, .01]])
  return b


@pytest.mark.parametrize('rho,bearing,support,gap,expected,bad', [
    (.45,0.,.50,.03,True,None), (.80,45.,.55,.03,True,None),
    (.78,-45.,.65,.030001,True,None), (.80001,0.,.55,.03,False,'dist_ok'),
    (.44999,0.,.55,.03,False,'dist_ok'), (.6,46.,.55,.03,False,'bearing_ok'),
    (.6,-46.,.55,.03,False,'bearing_ok'), (.6,0.,.55,.029,False,'clearance_ok'),
    (.6,0.,.4999,.03,False,'height_ok'), (.6,0.,.6501,.03,False,'height_ok'),
])
def test_true_contract_boundaries(rho,bearing,support,gap,expected,bad):
  rep=ct.report(isolated(rho,bearing,support,gap))
  assert rep['ok'] is expected
  assert rep['dist']==pytest.approx(rho)
  if bad:
    assert rep[bad] is False
    assert all(rep[k] for k in ('dist_ok','bearing_ok','height_ok','clearance_ok') if k!=bad)


def test_reach_interpolation_and_endpoint_margin():
  assert [ct.reach_limit(h) for h in ct.REACH_HEIGHTS]==list(ct.REACH_RADII)
  assert ct.reach_limit(.625)==pytest.approx(.785)
  assert ct.radius_grid(.625)==pytest.approx((.45,.50,.55,.60,.65,.70,.75,.785))
  env=make_ag_env(contract='true_state')
  env.forced_seed=good_seed(); env.step(act(reset=True))
  fs=list(sk.stance_candidates(env.backend,env.scene,rhos=ct.radius_grid(.625),dist_max=ct.reach_limit(.625),margin=.01))
  assert min(f[0] for f in fs)==pytest.approx(.46)
  assert max(f[0] for f in fs)==pytest.approx(.775)
  assert max(abs(f[2]) for f in fs)==42


def test_default_legacy_observation_and_gate_unchanged():
  assert load_config(['legmm_ag']).env.legmm_ag.contract=='legacy'
  a=make_ag_env(); b=make_ag_env(contract='legacy'); c=make_ag_env(contract='true_state')
  assert a.contract==b.contract=='legacy'
  assert set(a.obs_space)==set(c.obs_space)
  for env in (a,b):
    env.forced_seed=good_seed(); env.rng=np.random.default_rng(42)
  for action in [act(reset=True),act(nav=(.3,.1,-.2)),act(trigger=1)]:
    x,y=a.step(action),b.step(action)
    for k in x:
      if 'wall' not in k:
        np.testing.assert_array_equal(x[k],y[k],err_msg=k)


def test_estimated_gate_uses_estimated_height_and_keeps_other_limits():
  env,_=env_at_stance(rho=.79,contract='true_state')
  set_est(env, offset=(0.,0.,.50+env.scene.obj.half[2]-env.backend.state.obj_center[2]))
  assert env._gate()['dist_ok']
  set_est(env, offset=(0.,0.,.65+env.scene.obj.half[2]-env.backend.state.obj_center[2]))
  assert not env._gate()['dist_ok']
  env.backend.state.v=np.array([.30001,0,0])
  assert not env._gate()['speed_ok']
  assert env.gate_cfg['bearing_deg']==45. and tuple(env.gate_cfg['height'])==(.50,.65) and env.gate_cfg['speed']==.3


def test_legacy_success_becomes_true_state_contract_failure():
  env,_=env_at_stance()
  b=env.backend
  # A real successful legacy grasp with a 2.9 cm nearby obstacle; footprint stays collision free.
  s=b.state; c,sn=math.cos(s.yaw),math.sin(s.yaw)
  # Place a small low obstacle behind the body, along a body axis.
  rear=np.array([s.x,s.y])-(sc.FOOT_HALF[0]+.029+.005)*np.array([c,sn])
  b._static=np.concatenate([b._static, [[rear[0]-.005,rear[1]-.005,0.,rear[0]+.005,rear[1]+.005,.005]]])
  # For rotated footprints this obstacle may be <2.9 cm away, but is strictly outside the real footprint.
  assert sc.footprint_free(b._static,s.x,s.y,s.yaw,0.)
  assert not ct.report(b)['clearance_ok']
  snap=b.snapshot()
  legacy=sk.run_skill(b,env.skill_cfg)
  assert legacy.success and legacy.executed
  b.restore(snap)
  true=sk.run_skill(b,dict(env.skill_cfg,contract='true_state'))
  assert not true.success and true.reason=='contract_violation' and not true.executed and true.steps==0
  assert not true.contract['clearance_ok']


def test_post_settle_true_state_violation_is_terminal_failure():
  env,_=env_at_stance(rho=.65,contract='true_state')
  phi=env._phis
  original=env._settle
  def settle(b):
    n,col=original(b)
    base=ag_scenes.head_on_stance(env.scene,ct.reach_limit(env.scene.obj.support_z)+.001)
    b.state.x,b.state.y,b.state.yaw=base
    return n,col
  env._settle=settle
  assert env._gate()['ok']
  obs=env.step(act(trigger=1))
  assert obs['is_terminal'] and obs['log/invalid']==0 and obs['log/success']==0
  assert env.last_info['skill'].reason=='contract_violation'
  assert not env.last_info['contract']['dist_ok']
  assert obs['rew_task']==pytest.approx(-.01-5-phi[0],abs=1e-5)
  assert obs['rew_qshape']==pytest.approx(-phi[1],abs=1e-5)
  assert obs['log/skill_steps']==0


def _check_scene(seed):
  env=make_ag_env(contract='true_state')
  env.forced_seed=int(seed); obs=env.step(act(reset=True))
  fs=sk.feasible_stances(env.backend,env.scene,env.skill_cfg)
  for f in fs:
    rep=ct.report(env.backend,f['base'],env.gate_cfg)
    assert rep['ok'], (seed,f,rep)
  pol=cal.StancePolicy('privileged'); pol.reset(env)
  while not obs['is_last']:
    obs=env.step(pol.act(env,obs))
  result=env.last_info.get('skill')
  if result:
    assert result.reason!='contract_violation', (seed,result.contract)
  return len(fs), int(result is not None)


def test_random_scene_search_and_privileged_execution_consistency():
  import multiprocessing
  rng=np.random.default_rng(20261110)
  seeds=rng.choice(np.arange(20000,21000),size=100,replace=False)
  with multiprocessing.get_context('spawn').Pool(16) as pool:
    results=pool.map(_check_scene,map(int,seeds),chunksize=1)
  assert sum(x[0] for x in results)>100 and sum(x[1] for x in results)>0


@pytest.mark.parametrize('lam',[0.,.37,1.])
def test_true_state_shaping_telescopes(lam):
  from test_ag_shaping import _episode, _parts, _discounted
  env=make_ag_env(contract='true_state')
  first,rows=_episode(env,good_seed(),'privileged')
  assert rows[-1]['is_terminal'] and len(rows)>5
  assert float(first['log/phi_dist']) < -1.
  pd,pq,non=zip(*[_parts(r,env.gamma) for r in rows])
  assert abs(_discounted(pd,env.gamma)+float(first['log/phi_dist']))<1e-4
  assert abs(_discounted(pq,env.gamma)+float(first['log/phi_q']))<1e-4
  reward=[float(r['rew_task'])+lam*float(r['rew_qshape']) for r in rows]
  expected=_discounted(non,env.gamma)-float(first['log/phi_dist'])-lam*float(first['log/phi_q'])
  assert abs(_discounted(reward,env.gamma)-expected)<1e-3
