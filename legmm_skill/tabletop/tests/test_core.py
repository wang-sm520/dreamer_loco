from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tabletop.spec import sample_candidate, gate_reason, STOW, contract_modules
from tabletop.outcome import EpisodeOutcome, classify_contacts, lowest_point
from tabletop.control import DelayedCommands, expand_chunk
from tabletop.data import Reservoir, actor_inputs
from tabletop.storage import StorageGuard, SparseSaveGate, FLOOR_BYTES


def test_self_collision_excludes_only_mechanical_neighbours():
    from tabletop.collision import excluded_pair
    dogs=['base_link','FL_calf','FR_calf']
    assert excluded_pair('FL_calf','FR_calf',dogs)
    assert excluded_pair('base_link','arm_base',dogs)
    assert excluded_pair('link6','left',dogs)
    assert excluded_pair('link3','link2',dogs)
    assert not excluded_pair('link1','link3',dogs)
    assert not excluded_pair('left','right',dogs)
    assert not excluded_pair('link2','base_link',dogs)


def test_seed_reproduction_and_velocity_independence():
    a = sample_candidate(12,7)
    assert a == sample_candidate(12,7)
    b = sample_candidate(12,7,True)
    assert replace(b,velocity=(0.,0.,0.)) == a
    assert a != sample_candidate(12,8)
    assert {sample_candidate(12,i).shape for i in range(100)} == {a.shape}
    assert 0 < math.hypot(*b.velocity[:2]) < .05


def test_distribution_not_local_noise():
    rows = [sample_candidate(i,residual=True) for i in range(1000)]
    assert {s.shape for s in rows} == {'box','cylinder'}
    assert 620 < sum(s.shape=='box' for s in rows) < 780
    assert min(s.radius for s in rows)<.46 and max(s.radius for s in rows)>.78
    assert min(s.bearing for s in rows)<-.75 and max(s.bearing for s in rows)>.75
    assert min(s.support for s in rows)<.51 and max(s.support for s in rows)>.64
    assert all(.6<=s.friction<=1. and .1<=s.mass<=.3 for s in rows)
    assert len(set(s.size for s in rows))==len(rows)
    radii2 = [sum(v*v for v in s.velocity[:2]) for s in rows]
    assert abs(np.mean(radii2)-.05**2/2)<.0001


def test_inertia_responds_to_actual_dimensions():
    s = next(sample_candidate(i) for i in range(20) if sample_candidate(i).shape=='box')
    assert np.allclose(replace(s,mass=2*s.mass).inertia,2*s.inertia)
    changed = replace(s,size=(s.size[0]*2,*s.size[1:]))
    assert changed.inertia[0] == s.inertia[0]
    assert changed.inertia[1] > s.inertia[1]


def test_gate_is_actual_contract_and_strict_settle():
    contract,_ = contract_modules()
    assert contract.reach_limit(.625)==pytest.approx(.785)
    s = next(s for i in range(1000) if (s:=sample_candidate(i)) and gate_reason(s) is None)
    assert gate_reason(replace(s,velocity=(.05,0,0))) == 'unsettled'
    assert gate_reason(replace(s,velocity=(0,0,.1))) == 'unsettled'
    assert gate_reason(replace(s,base_xy=(0.,0.))) in ('envelope','clearance')


def update(out,n,bottom=.56,v=0.,contacts=(True,True),**kw):
    for _ in range(n):
        out.update(bottom,v,contacts,**kw)


def test_hold_duration_and_lowest_point():
    o = EpisodeOutcome(.5)
    update(o,199)
    assert o.reason is None and not o.success
    update(o,1)
    assert o.success and o.steps==200
    assert lowest_point(.6,np.eye(3),'box',(.04,.12,.14))==pytest.approx(.53)
    R=np.array([[0,0,1],[0,1,0],[-1,0,0]])
    assert lowest_point(.6,R,'cylinder',(.06,.06,.14))==pytest.approx(.57)


def test_hold_must_be_continuous_and_failure_dominates():
    o=EpisodeOutcome(.5)
    update(o,199);update(o,1,bottom=.54);update(o,199)
    assert not o.success
    update(o,1,forbidden=('arm:table',))
    assert o.reason=='collision:arm:table' and not o.success


@pytest.mark.parametrize('kind',['timeout','drop','fall'])
def test_negative_outcomes(kind):
    o=EpisodeOutcome(.5)
    if kind=='timeout':
        update(o,1500,bottom=.5)
        assert o.reason=='timeout'
    elif kind=='drop':
        update(o,1);update(o,1,bottom=.509)
        assert o.reason=='drop'
    else:
        update(o,1,fallen=True)
        assert o.reason=='base_fall'
    assert not o.success


def test_zero_contact_is_not_drop_without_downward_motion():
    o=EpisodeOutcome(.5)
    update(o,100);update(o,100,contacts=(False,False))
    assert o.success
    o=EpisodeOutcome(.5)
    update(o,1);update(o,1,v=-.02,contacts=(False,False))
    assert o.reason=='drop'


def test_timeout_not_extended_by_batch_waypoints():
    a,b=EpisodeOutcome(.5),EpisodeOutcome(.5)
    for _ in range(375):
        update(a,4,bottom=.5)
    for _ in range(100):
        update(b,30,bottom=.5)
    assert a.reason==b.reason=='timeout' and a.steps==b.steps==1500


def test_contact_semantics_direction_threshold_and_forbidden():
    legal=[('foot','floor',10),('table','object',2),('object','left',2)]
    fingers,bad=classify_contacts(legal)
    assert fingers==(True,False) and not bad
    assert classify_contacts([(b,a,f) for a,b,f in legal])==(fingers,bad)
    assert not classify_contacts([('arm','table',1)])[1]
    assert classify_contacts([('arm','table',1.01)])[1]==('arm:table',)
    assert classify_contacts([('object','dog',2)])[1]==('dog:object',)


def test_delay_survives_decisions_and_checkpoint():
    queue=DelayedCommands(np.zeros(7))
    assert all(not queue.step(np.ones(7)).any() for _ in range(10))
    assert queue.step(np.ones(7)).tolist()==[1.]*7
    restored=DelayedCommands(np.zeros(7));restored.load_state_dict(queue.state_dict())
    for v in range(30):
        assert np.array_equal(queue.step(np.full(7,v)),restored.step(np.full(7,v)))


def test_chunk_keeps_native_scale_and_limits():
    q=np.array((*STOW,.072)); chunk=np.tile(q,(4,1));chunk[:,0]+=.08
    points=expand_chunk(q,chunk,np.zeros((4,7)))
    assert np.allclose(points[-1],chunk[-1])
    assert np.max(abs(np.diff(np.vstack((q,points)),axis=0)[:,:6])) <= .01000001
    assert np.allclose(expand_chunk(q,chunk),points)


def test_actor_whitelist_and_failure_reservoir():
    images=np.zeros((2,2,128,128)); proprio=np.zeros(7); label=np.zeros((8,7))
    buf=Reservoir(20,3)
    for i in range(10):
        buf.add(images,proprio,label,dict(shape='box',outcome='collision',round=0,seed=i))
    assert len(buf.items)==10 and buf.composition[('box','collision',0)]==10
    for i in range(30):
        buf.add(images,proprio,label,dict(shape='cylinder',outcome='success',round=1,seed=i))
    assert len(buf.items)==20 and buf.seen==40
    assert set(actor_inputs(images[None],proprio[None]))=={'images','proprio'}
    with pytest.raises(TypeError):
        actor_inputs(images[None],proprio[None],target_position=np.zeros(3))
    assert buf.tensor_bytes==20*(images.size*2+proprio.size*4+label.size*4)


def test_sparse_framework_defaults_cannot_write(tmp_path):
    for stage,count,expected in [('dagger',6,[3,6]),('ppo',1000,[500,1000])]:
        writes=[];gate=SparseSaveGate(stage,writes.append)
        for i in range(count+1):
            gate.save(i,{'step':i})
        assert gate.writes==expected
        assert [w['step'] for w in writes]==expected


def test_atomic_peak_includes_old_checkpoint_and_preserves_it(tmp_path):
    code=tmp_path/'code'; code.mkdir()
    root=tmp_path/'data'; root.mkdir()
    (root/'latest.pt').write_bytes(b'previous')
    g=StorageGuard(root,code,free=lambda _:SimpleNamespace(free=FLOOR_BYTES+100))
    with pytest.raises(OSError,match='storage stop'):
        g.atomic_bytes('latest.pt',b'new')
    assert (root/'latest.pt').read_bytes()==b'previous'
    assert not (root/'latest.pt.pending').exists()
    g.free=lambda _:SimpleNamespace(free=FLOOR_BYTES+8192)
    g.atomic_bytes('latest.pt',b'new')
    assert (root/'latest.pt').read_bytes()==b'new'
    with pytest.raises(ValueError):
        g.atomic_bytes('../old_experiment.pt',b'x')


@pytest.mark.parametrize('delta,allowed',[(-1,False),(0,True),(1,True)])
def test_ten_gb_boundary(tmp_path,delta,allowed):
    assert FLOOR_BYTES == 10_000_000_000
    g=StorageGuard(tmp_path,tmp_path,free=lambda _:SimpleNamespace(free=FLOOR_BYTES+delta))
    if allowed:
        g.check()
    else:
        with pytest.raises(OSError,match='storage stop'): g.check()


def test_kit_zero_exit_without_physics_is_failure():
    from tabletop.launch import completed_probe
    assert not completed_probe(0,None,32)
    assert not completed_probe(0,dict(physical_steps=0,num_envs=32,episodes=[{}]*32),32)
    valid=dict(physical_steps=1800,num_envs=32,episodes=[{}]*32)
    assert completed_probe(0,valid,32)
    assert not completed_probe(0,valid,32,error_exists=True)
    assert not completed_probe(1,valid,32)


def test_contact_configuration_matches_installed_api():
    import ast
    source=Path('/home/bdi/wang-sm/IsaacLab/source/isaaclab/isaaclab/sensors')
    fields=set()
    for filename in ('sensor_base_cfg.py','contact_sensor/contact_sensor_cfg.py'):
        tree=ast.parse((source/filename).read_text())
        fields.update(n.target.id for n in ast.walk(tree) if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name))
    tree=ast.parse((Path(__file__).parents[1]/'isaac_scene.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ContactSensorCfg']
    assert calls
    assert all({k.arg for k in c.keywords}<=fields for c in calls)


def test_reused_sampling_cannot_be_relabelled_residual():
    from tabletop.spec import InitialState,validate_reused_sampling
    static=[sample_candidate(i,4) for i in range(3)]
    restored=[InitialState(**json.loads(json.dumps(s.json()))) for s in static]
    validate_reused_sampling(restored,0,3,False)
    with pytest.raises(ValueError,match='distribution'):
        validate_reused_sampling(restored,0,3,True)
    residual=[sample_candidate(i,4,True) for i in range(3)]
    validate_reused_sampling(residual,0,3,True)
    with pytest.raises(ValueError,match='seeds'):
        validate_reused_sampling(residual,1,3,True)


def test_gate_diagnostics_retain_failure_components():
    from tabletop.spec import gate_diagnostics
    s=next(s for i in range(1000) if (s:=sample_candidate(i)) and gate_reason(s) is None)
    report=gate_diagnostics(s,velocity=(.05,0.,0.))
    assert report['contract']['envelope_ok']
    assert report['clearance_ok'] and report['supported']
    assert not report['settled'] and report['reason']=='unsettled'
    far=gate_diagnostics(s,shoulder_xy=(100.,100.))
    assert not far['contract']['dist_ok'] and far['reason']=='envelope'
    assert far['contract']['dist']>100


def test_gate_diagnostics_serializes_live_float32():
    from tabletop.spec import gate_diagnostics
    s=sample_candidate(6)
    report=gate_diagnostics(s,np.array([.1,.2],dtype=np.float32),
                            np.array(s.base_xy,dtype=np.float32),np.float32(s.base_yaw),
                            np.array([.01,.02,.03],dtype=np.float32))
    restored=json.loads(json.dumps(report))
    assert restored['velocity']==pytest.approx([.01,.02,.03])
    assert restored['shoulder_xy']==pytest.approx([.1,.2])


def test_geometry_uses_measured_object_and_joint_pose(tmp_path):
    from tabletop.assets import build_robot
    from tabletop.geometry import GeometryPlanner
    s=sample_candidate(6)
    planner=GeometryPlanner(build_robot(tmp_path/'model'))
    base=[*s.base_xy,.382,math.cos(s.base_yaw/2),0,0,math.sin(s.base_yaw/2)]
    pose=np.r_[s.center+[.003,-.004,-.001],[1.,0.,0.,0.]]
    planner.configure(s,base,pose,{'joint1':.02,'FL_thigh_joint':.93})
    assert planner.object_center==pytest.approx(pose[:3])
    assert planner.d.qpos[planner.m.joint('FL_thigh_joint').qposadr[0]]==pytest.approx(.93)
    assert planner.initial_arm[0]==pytest.approx(.02)
    planner.collision(planner.initial_arm,np.array([0.,0.,.08]))
    assert planner.d.qpos[planner.objqa:planner.objqa+3]==pytest.approx(pose[:3]+[0.,0.,.08])


def test_teacher_timing_independent_of_collinear_geometry_subdivision():
    from tabletop.control import teacher_commands
    q=np.array([*STOW,.072]);end=q.copy();end[0]+=.03
    def plan(points):
        return SimpleNamespace(joints=[p[:6] for p in points],phases=['pre']*len(points),width=.04)
    coarse=teacher_commands(plan([end]),q)
    dense=teacher_commands(plan([q+(end-q)*f for f in np.linspace(.1,1,10)]),q)
    assert len(coarse)==len(dense)==63  # 3 motion ticks + configured 60 hold ticks
    assert np.allclose(coarse,dense)
    assert np.max(abs(np.diff(np.vstack((q,dense)),axis=0)[:,:6]))<=.01000001


def test_teacher_closure_dwell_precedes_lift():
    from tabletop.control import teacher_commands
    q=np.array([*STOW,.072]);pre=q.copy();pre[0]+=.02;lift=pre.copy();lift[0]+=.05
    p=SimpleNamespace(joints=[pre[:6],lift[:6]],phases=['grasp','lift'],width=.04)
    commands=teacher_commands(p,q)
    moving=np.flatnonzero(commands[:,0]>pre[0]+1e-8)
    assert len(moving)>0
    assert np.allclose(commands[moving,6],0.)
    closed_at_pre=np.flatnonzero((abs(commands[:,6])<1e-8)&(abs(commands[:,0]-pre[0])<1e-8))
    assert len(closed_at_pre)>=15


def test_calibrated_pose_transfers_measured_stance_with_world_yaw():
    from tabletop.prepare_sampling import calibrated_pose
    from scipy.spatial.transform import Rotation
    s=sample_candidate(4)
    q=Rotation.from_euler('xyz',[.01,-.02,s.base_yaw]).as_quat()
    measured=np.r_[np.array(s.base_xy)+[.005,-.003],.382,q[3],q[:3]]
    assert calibrated_pose(s,s,measured)==pytest.approx(measured)
    changed=replace(s,base_yaw=s.base_yaw+math.pi/2,base_xy=(2.,3.))
    transferred=calibrated_pose(changed,s,measured)
    assert transferred[:3]==pytest.approx([2.003,3.005,.382])
    turn=Rotation.from_euler('z',math.pi/2)
    expected=(turn*Rotation.from_quat(q)).as_matrix()
    assert Rotation.from_quat(np.r_[transferred[4:7],transferred[3]]).as_matrix()==pytest.approx(expected)


def test_mesh_shoulder_keeps_verified_inertials_and_source_meshes(tmp_path):
    import xml.etree.ElementTree as ET
    from tabletop.assets import build_robot
    from tabletop.spec import W8
    original=ET.parse(W8/'stand_probe/model/a2_arm.xml').getroot()
    built=ET.parse(build_robot(tmp_path/'robot')).getroot()
    old={b.get('name'):b.find('inertial') for b in original.iter('body')}
    for body in built.iter('body'):
        if old[body.get('name')] is not None:
            assert body.find('inertial').attrib==old[body.get('name')].attrib
        if body.get('name') in ('arm_base','link2'):
            colliders=[g for g in body.findall('geom') if g.get('class')=='arm_collision']
            assert len(colliders)==2
            assert all(g.get('mesh') and g.get('mass')=='0' for g in colliders)
