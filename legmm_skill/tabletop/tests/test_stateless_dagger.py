"""Geometry, filtering and dual-pool recovery regressions."""
import gzip
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tabletop.assets import build_robot
from tabletop.closed_loop import ClosedLoopTeacher, MeasuredState
from tabletop.dagger_recovery import reservoir_state, restore_latest, save_latest
from tabletop.data import Reservoir, SplitReservoir
from tabletop.geometry import GeometryPlanner, side_heading
from tabletop.learning import DaggerSession
from tabletop.spec import InitialState, OUTPUT, PHYSICS_DT, STEP_LENGTH, UPPER
from tabletop.state_fingerprints import digest
from tabletop.stateless_teacher import EpisodeLabels, StatelessDaggerTeacher
from tabletop.student import Student


@pytest.fixture
def physical_geometry(tmp_path):
    source = OUTPUT / 'teacher_causal_train_after_01/result.json'
    data = json.loads(source.read_text())
    state = InitialState(**data['states'][0])
    joints = dict(zip(data['joint_names'], data['initial_joint_positions'][0]))
    native = np.array([*(joints[f'joint{i}'] for i in range(1, 7)), joints['endleft']-joints['endright']])
    measured = MeasuredState(np.array(data['initial_base_pose'][0]), np.array(data['initial_object_pose'][0]),
                             joints, np.array(data['initial_base_velocity'][0])[[0,1,5]], native,
                             np.repeat(native[None], 11, axis=0), (False, False), 0.)
    planner = GeometryPlanner(build_robot(tmp_path/'robot'), cache_models=True)
    return source, planner, state, measured


def test_offline_geometry_replay(physical_geometry):
    source, planner, state, measured = physical_geometry
    old = ClosedLoopTeacher(planner, state)
    new = StatelessDaggerTeacher(planner, state)
    _, initial = old.label(measured)
    assert initial['reason'] is None
    old.phase = 'pre'  # Prior aligning decisions, as in the archived student trace.
    live, _, _ = new._configure(measured)
    heading, width, _ = side_heading(live, new.cfg)
    c, s = np.cos(heading), np.sin(heading)
    axis = np.array([c, s, 0.])
    rotation = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    depth, _ = planner.grasp_depth(axis)
    goal = measured.object_pose[:3] - depth*axis
    distance = planner.pregrasp_distance(axis, depth)
    grasp_off_axis = float(np.clip(.5 * (UPPER[6] - width) - .001, .002, .014))
    grasp_lateral = min(.005, .8 * grasp_off_axis) if width > .060 else .005
    samples = []
    for name, standoff, lateral, contact, elevation in [
        ('grasp', distance-.006, grasp_lateral, (False,False), 0.),
        ('close', .002, .001, (True,False), 0.),
        ('lift', 0., .000, (True,True), 0.),
        ('hold', 0., .000, (True,True), new.cfg.lift+.003),
    ]:
        target = goal - standoff*axis + lateral*rotation[:,1] + [0,0,elevation]
        arm = planner.ik(target, rotation, planner.initial_arm)
        assert arm is not None
        opening = width if all(contact) else .072
        native = np.r_[arm,opening]
        joints = dict(measured.joints, endleft=opening/2, endright=-opening/2)
        joints.update({f'joint{i}': value for i,value in enumerate(arm,1)})
        object_pose = measured.object_pose.copy()
        object_pose[2] += elevation
        observation = replace(measured, joints=joints, last_command=native,
                              delay_queue=np.repeat(native[None],11,axis=0), finger_contact=contact,
                              object_pose=object_pose, elapsed_s=1.)
        prior, legacy = old.label(observation)
        actual, metadata = new.label(observation)
        assert legacy['phase'] == 'pre'
        assert metadata['phase'] == name and metadata['reason'] is None
        assert (np.abs(np.diff(np.vstack((native,actual)),axis=0)) <= STEP_LENGTH+1e-8).all()
        assert np.array_equal(actual, StatelessDaggerTeacher(planner, state).label(observation)[0])
        planner.fk(arm)
        actual_tcp = planner.fk(actual[0,:6])[0]
        legacy_tcp = planner.fk(prior[0,:6])[0]
        forward = float((actual_tcp-target)@axis)
        legacy_forward = float((legacy_tcp-target)@axis)
        if name == 'grasp':
            assert forward > .001 and legacy_forward < 0
        if name in ('close','lift','hold'):
            assert actual[-1,6] < native[6]
        if name == 'lift':
            assert actual_tcp[2] > target[2]
        samples.append(dict(expected=name,phase=metadata['phase'],reason=metadata['reason'],
                            old_phase=legacy['phase'],old_reason=legacy['reason'],
                            axial_step_from_fixture_target_m=forward,old_axial_step_from_fixture_target_m=legacy_forward,
                            first_tcp_z_m=float(actual_tcp[2]), first_old_tcp_z_m=float(legacy_tcp[2]),
                            desired_tcp_z_m=float(metadata['target_tcp'][2])))
    if path := os.environ.get('W8_OFFLINE_EVIDENCE'):
        Path(path).write_text(json.dumps(dict(settled_state_source=str(source),seed=state.seed,
            synthetic_waypoints_at_measured_stance=True, samples=samples),indent=2)+'\n')


@pytest.mark.parametrize('angle,contact,reason', [
    (2.,(False,False),None),
    (16.,(False,False),'tilted_object_hold'),
    (16.,(True,True),None),
])
def test_object_tilt_filter(physical_geometry,angle,contact,reason):
    from scipy.spatial.transform import Rotation
    _,planner,state,measured=physical_geometry
    measured.elapsed_s=1.
    q=measured.object_pose[[4,5,6,3]]
    q=(Rotation.from_quat(q)*Rotation.from_euler('x',angle,degrees=True)).as_quat()
    measured.object_pose[3:]=q[[3,0,1,2]]
    measured.finger_contact=contact
    commands,meta=StatelessDaggerTeacher(planner,state).label(measured)
    if reason:
        assert meta['reason']==reason
        assert np.array_equal(commands,np.repeat(measured.last_command[None],8,axis=0))
    else:
        assert meta['reason'] not in ('tilted_object_hold','terminal_safety_hold')


def test_safety_and_delay_queue(physical_geometry):
    _,planner,state,measured=physical_geometry
    teacher=StatelessDaggerTeacher(planner,state)
    invalid=replace(measured,base_pose=measured.base_pose+np.array([2.,2.,0,0,0,0,0]))
    hold=np.repeat(measured.last_command[None],8,axis=0)
    for observation, reason in [(invalid,'initial_gate:envelope'),
                                (replace(measured, elapsed_s=15.),'terminal_safety_hold'),
                                (replace(measured, forbidden_contact=True),'terminal_safety_hold')]:
        label,metadata=teacher.label(observation)
        assert metadata['reason']==reason
        assert np.array_equal(label,hold)
    invalid_queue=replace(measured,delay_queue=measured.delay_queue+1)
    with pytest.raises(ValueError, match='live delay queue'):
        teacher.label(invalid_queue)


def test_scoped_adapter_teacher_reason_and_student_labels(physical_geometry):
    from tabletop import closed_loop
    _,planner,state,measured=physical_geometry
    session=DaggerSession.__new__(DaggerSession)
    session.states=[state]
    session.teacher_drives=session.stateless_dagger=True
    session.episode_labels=EpisodeLabels()
    from collections import Counter
    session.stateless_phase_counts,session.valid_phase_counts,session.filtered_reasons=Counter(),Counter(),Counter()
    with session.episode_labels.install():
        teacher=closed_loop.ClosedLoopTeacher(planner,state)
        _,meta=teacher.label(replace(measured,forbidden_contact=True))
        labels=np.zeros((1,8,7))
        assert meta['reason']=='terminal_safety_hold'
        assert session._select_supervision(labels,[0],0)[1]==[]
        assert session.filtered_reasons=={'terminal_safety_hold':1}
        session.teacher_drives=False
        teacher.label(replace(measured,elapsed_s=16*PHYSICS_DT))
        recovered,active=session._select_supervision(labels,[0],16)
        assert active==[0] and np.any(recovered) and not np.any(labels)
        assert sum(session.stateless_phase_counts.values())==1
    assert closed_loop.ClosedLoopTeacher is ClosedLoopTeacher
    session.stateless_dagger=False
    assert session._select_supervision(labels,[0],16)==(labels,[0])


def add(pool,count,value,round_index):
    for i in range(count):
        pool.add(np.zeros((2,2,128,128)), np.full(14,i), np.full((8,7),value),
                 dict(shape='box', outcome='success', round=round_index, seed=i))


def test_balanced_sampling_and_streaming_restore(tmp_path):
    student=Student('cpu',123,command_history=True,relative_delta=True)
    visual=SimpleNamespace(manifest=lambda:{'test':True})
    def fresh(cache):
        return SimpleNamespace(student=student,visual=visual,reservoir=cache,seen=64,budget=64,
            round=1,teacher_drives=False,episode_serial=2,physical_env_decisions=64,collected_samples=64)
    original=fresh(SplitReservoir(5,capacity=40))
    demo,dagger=original.reservoir.demo, original.reservoir.dagger
    def counts():
        return np.unique(original.reservoir.sample(64)['label'][:,0,0],return_counts=True)[1].tolist()
    add(demo,61,1,0)
    assert counts()==[64]
    add(dagger,5,2,1)
    assert counts()==[59,5]
    add(dagger,56,2,1)
    assert counts()==[32,32] and original.reservoir.retained==80
    guard=SimpleNamespace(check=lambda *args:None)
    record=save_latest(original,{},2,384,2,tmp_path,guard,compress=True)
    target=fresh(SplitReservoir(789,capacity=40))
    _,cursor=restore_latest(target,record['path'],expected_sha256=record['sha256'])
    assert cursor['next_round']==2 and cursor['next_seed']==384
    assert digest(reservoir_state(target.reservoir))==digest(reservoir_state(original.reservoir))
    for _ in range(3):
        assert digest(target.reservoir.sample(64))==digest(original.reservoir.sample(64))
    with gzip.open(record['path'],'rb') as stream:
        restored=torch.load(stream,map_location='cpu',weights_only=False)
    assert set(restored['reservoir'])=={'demo','dagger'}
    with pytest.raises(ValueError,match='reservoir mode'):
        restore_latest(fresh(Reservoir(40,789,14)),record['path'])


def test_stateless_fast_gate_requires_valid_phases_direction_and_success(tmp_path):
    from tabletop.train_dagger import write_round1_fast_gate
    cohort=tmp_path/'round_01_episode_0006'
    cohort.mkdir()
    phases=dict(clear=1,pre=1,grasp=1,close=1,lift=1)

    def gate(cosine, success, valid):
        result=dict(episodes=[dict(reason='success' if success else 'timeout',success=success)])
        (cohort/'result.json').write_text(json.dumps(result))
        stats=dict(arm_step4_eligible=5,arm_step4_amp_ratio_median=.8,arm_step4_cos_median=cosine)
        (cohort/'learning_decisions.jsonl').write_text(json.dumps(stats)+'\n')
        collection=dict(stateless_phase_counts=phases,valid_phase_counts=valid,
                        filtered_reasons={'tilted_object_hold':1})
        (cohort/'collection.json').write_text(json.dumps(collection))
        return write_round1_fast_gate(tmp_path,tmp_path,stateless_dagger=True)

    assert gate(.51,True,phases)['passed']
    assert not gate(.50,True,phases)['passed']
    assert not gate(.51,False,phases)['passed']
    missing=gate(.51,True,dict(phases,lift=0))
    assert not missing['passed'] and missing['missing_valid_phases']==['lift']
    assert write_round1_fast_gate(tmp_path,tmp_path)['passed']


@pytest.mark.parametrize('success, stop', [
    (44,'round_03_improvement_gate_stopped'),
    (45,None),
    (217,None),
    (218,'round_03_student_gate_passed'),
])
def test_stateless_validation_round_three_milestone(success,stop):
    from tabletop.train_dagger import stateless_round_three_stop
    assert stateless_round_three_stop(success)==stop


@pytest.mark.parametrize('compute_mib,graphics_mib,expected', [
    (0,7,'passive_vulkan_other_gpus'),
    (7,0,'wrong_gpu_allocations'),
    (0,33,'wrong_gpu_allocations'),
    (0,32,'passive_vulkan_other_gpus'),
    (7,7,'wrong_gpu_allocations'),
])
def test_nvml_cross_gpu_separates_compute_and_vulkan(monkeypatch,compute_mib,graphics_mib,expected):
    import sys
    from tabletop.gpu_guard import gpu_snapshot

    mib=2**20
    owner, child, foreign=os.getpgrp(),os.getpid(),2**30
    def row(pid, amount):
        return SimpleNamespace(pid=pid,usedGpuMemory=amount*mib)

    compute={0:[row(child,compute_mib)],1:[],2:[row(owner,11),row(foreign,4)]}
    graphics={0:[row(child,graphics_mib),row(foreign,50)],1:[],2:[row(owner,13),row(child,5)]}
    nv=SimpleNamespace(nvmlInit=lambda:None,nvmlShutdown=lambda:None,
        nvmlDeviceGetHandleByIndex=lambda index:index,nvmlDeviceGetUUID=lambda index:f'GPU-{index}',
        nvmlDeviceGetCount=lambda:3,nvmlDeviceGetMemoryInfo=lambda handle:SimpleNamespace(used=28*mib,free=100*mib),
        nvmlDeviceGetComputeRunningProcesses=compute.__getitem__,
        nvmlDeviceGetGraphicsRunningProcesses=graphics.__getitem__)
    monkeypatch.setitem(sys.modules,'pynvml',nv)
    monkeypatch.setattr(os,'getpgid',lambda pid:owner if pid==child else pid)

    result=gpu_snapshot(2,owner,baseline_overhead=2)
    assert result[expected]==[dict(index=0,uuid='GPU-0',pid=child,
                                  memory_mib=max(compute_mib,graphics_mib),
                                  compute_mib=compute_mib,graphics_mib=graphics_mib)]
    other='wrong_gpu_allocations' if expected=='passive_vulkan_other_gpus' else 'passive_vulkan_other_gpus'
    assert result[other]==[]
    assert result['known_own_mib']==18 and result['owned_upper_bound_mib']==22
    assert result['known_external_mib']==4 and result['external_upper_bound_mib']==6
    assert result['processes']==[dict(uuid='GPU-2',pid=pid,memory_mib=amount)
                                 for pid,amount in [(owner,13),(foreign,4),(child,5)]]


@pytest.mark.parametrize('opening_offset,phase,direction',[
    (.002,'close',-1),
    (-.005,'pre',1),
])
def test_close_hysteresis_uses_measured_opening(physical_geometry,opening_offset,phase,direction):
    _,planner,state,measured=physical_geometry
    teacher=StatelessDaggerTeacher(planner,state)
    live,_,_=teacher._configure(measured)
    heading,width,_=side_heading(live,teacher.cfg)
    c,s=np.cos(heading),np.sin(heading)
    axis=np.array([c,s,0.])
    rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    depth,_=planner.grasp_depth(axis)
    goal=measured.object_pose[:3]-depth*axis
    arm=planner.ik(goal-.013*axis,rotation,planner.initial_arm)
    assert arm is not None
    tcp,_=planner.fk(arm)
    assert .010 < (goal-tcp)@axis < .018
    opening=width+opening_offset if direction<0 else min(state.size[:2])-.006
    native=np.r_[arm,opening]
    joints=dict(measured.joints,endleft=opening/2,endright=-opening/2)
    joints.update({f'joint{i}':value for i,value in enumerate(arm,1)})
    observation=replace(measured,joints=joints,last_command=native,
                        delay_queue=np.repeat(native[None],11,axis=0),elapsed_s=1.)
    labels,metadata=teacher.label(observation)
    assert metadata['phase']==phase and metadata['reason'] is None
    assert direction*(labels[-1,6]-opening)>.001


@pytest.mark.parametrize('width,opening,off_axis,expected',[
    (.059,.046,0.,'close'),
    (.068,.072,.008,'pre'),
    (.068,.072,.001,'grasp'),
])
def test_twisted_box_thickness_and_narrow_pad_clearance(
        physical_geometry,monkeypatch,width,opening,off_axis,expected):
    from tabletop import stateless_teacher
    _,_,state,measured=physical_geometry
    state=replace(state,shape='box',size=(.035,.127,.124))
    goal=np.array([.50,0.,.61])
    initial=np.zeros(6)
    initial[:3]=goal+[-.013,off_axis,0.]
    planner=SimpleNamespace(initial_arm=initial,fk=lambda q:(q[:3],np.eye(3)),
        grasp_depth=lambda axis:(0.,{}),pregrasp_distance=lambda axis,depth:.04,
        ik=lambda tcp,rotation,previous:np.r_[tcp,previous[3:]],
        collision=lambda *args:None)
    teacher=StatelessDaggerTeacher(planner,state)
    monkeypatch.setattr(teacher,'_configure',lambda measured:(state,np.eye(3),np.zeros(3)))
    monkeypatch.setattr(stateless_teacher,'side_heading',lambda state,cfg:(0.,width,0.))
    native=np.r_[initial,opening]
    obj=measured.object_pose.copy()
    obj[:3]=goal
    observation=replace(measured,object_pose=obj,elapsed_s=1.,
        joints=dict(measured.joints,endleft=opening/2,endright=-opening/2),
        last_command=native,delay_queue=np.repeat(native[None],11,axis=0))
    labels,meta=teacher.label(observation)
    assert meta['phase']==expected and meta['reason'] is None
    if expected=='close':
        assert labels[-1,6]<opening and meta['target_tcp']==goal.tolist()
    elif expected=='pre':
        assert meta['target_tcp'][0]<initial[0] and labels[-1,0]<initial[0]
    else:
        assert meta['target_tcp']==goal.tolist() and labels[-1,0]>initial[0]
