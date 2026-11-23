import copy
import json
import math
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tabletop.closed_loop import ClosedLoopTeacher, MeasuredState
from tabletop.geometry import GeometryPlanner
from tabletop.assets import build_robot
from tabletop.spec import STOW, InitialState, OUTPUT, STEP_LENGTH


@pytest.fixture
def measured_teacher(tmp_path):
    # Real training-seed settled stance; no validation/test states are queried.
    data=json.loads((OUTPUT/'teacher_causal_train_after_01/result.json').read_text())
    s=InitialState(**data['states'][0])
    joints=dict(zip(data['joint_names'],np.asarray(data['initial_joint_positions'][0],dtype=np.float32)))
    initial=np.r_[[joints[f'joint{i}'] for i in range(1,7)],joints['endleft']-joints['endright']]
    measured=MeasuredState(np.array(data['initial_base_pose'][0]),np.array(data['initial_object_pose'][0]),
        joints,np.array(data['initial_base_velocity'][0])[[0,1,5]],initial,np.repeat(initial[None],11,axis=0),(False,False),0.)
    return ClosedLoopTeacher(GeometryPlanner(build_robot(tmp_path/'model')),s),measured


def test_object_translation_changes_label_from_actual_pose(measured_teacher):
    t,m=measured_teacher;t.phase='grasp'
    a,meta=t.label(m)
    json.dumps(meta)  # Real simulator joints are numpy float32 scalars.
    shifted=copy.deepcopy(m);shifted.object_pose[:2]+=[.003,-.004]
    t.phase='grasp';b,other=t.label(shifted)
    assert meta['reason'] is None and other['reason'] is None
    assert np.linalg.norm(a-b)>1e-5
    assert other['target_tcp']!=pytest.approx(meta['target_tcp'])
    assert np.max(abs(np.diff(np.vstack((m.last_command,b)),axis=0))/STEP_LENGTH)<=1.000001


def test_live_yaw_and_joint_deviation_change_supervision(measured_teacher):
    t,m=measured_teacher;t.phase='pre';a,meta=t.label(m)
    shifted=copy.deepcopy(m)
    q=Rotation.from_quat(np.r_[m.object_pose[4:7],m.object_pose[3]])
    q=(Rotation.from_euler('z',.02)*q).as_quat()
    shifted.object_pose[3:]=np.r_[q[3],q[:3]]
    shifted.joints['joint1']+=.03
    t.phase='pre';b,other=t.label(shifted)
    assert other['live_object_yaw']-meta['live_object_yaw']==pytest.approx(.02)
    assert not np.allclose(a,b)


def test_close_needs_measured_bilateral_contact(measured_teacher):
    t,m=measured_teacher;t.phase='close'
    t.label(m);assert t.phase=='close'
    m.finger_contact=(True,False);t.label(m);assert t.phase=='close'
    m.finger_contact=(True,True);t.label(m)
    assert t.phase=='lift' and t.grasp_relative is not None


def test_safety_hold_does_not_reset_delay_queue(measured_teacher):
    t,m=measured_teacher
    m.delay_queue[:,0]+=np.linspace(0,.005,11);before=m.delay_queue.copy()
    m.forbidden_contact=True
    labels,meta=t.label(m)
    assert meta['reason']=='terminal_safety_hold'
    assert labels==pytest.approx(np.repeat(m.last_command[None],8,axis=0))
    assert np.array_equal(before,m.delay_queue)


def test_physical_pad_normals_and_slide_axes_match_tcp(measured_teacher):
    t,m=measured_teacher;p=t.planner;p.configure(t.state,m.base_pose,m.object_pose,m.joints)
    _,tcp=p.fk(p.initial_arm)
    for name,joint in [('left_finger_pad','endleft'),('right_finger_pad','endright')]:
        gid=p.m.geom(name).id
        # Thin pad dimension is local y, not the long z axis.
        assert p.m.geom_size[gid]==pytest.approx([.00525,.002,.02])
        pad_R=p.d.geom_xmat[gid].reshape(3,3)
        slide=p.d.xaxis[p.m.joint(joint).id]
        assert abs(np.dot(pad_R[:,1],tcp[:,1]))>1-1e-8
        assert abs(np.dot(slide,tcp[:,1]))>1-1e-8


def test_cached_geometry_matches_fresh_after_object_and_joint_deviation(measured_teacher):
    t,m=measured_teacher;p=t.planner
    p.cache_models=True
    p.configure(t.state,m.base_pose,m.object_pose,m.joints)
    model=p.m
    changed=m.object_pose.copy();changed[:2]+=[.004,-.002]
    joints=dict(m.joints);joints['joint1']+=.02
    p.configure(t.state,m.base_pose,changed,joints)
    assert p.m is model
    cached=p.fk(p.initial_arm);cached_collision=p.collision(p.initial_arm)
    fresh=GeometryPlanner(p.robot_path);fresh.configure(t.state,m.base_pose,changed,joints)
    exact=fresh.fk(fresh.initial_arm)
    assert cached[0]==pytest.approx(exact[0])
    assert cached[1]==pytest.approx(exact[1])
    assert cached_collision==fresh.collision(fresh.initial_arm)


def test_pending_commands_change_lookahead_without_changing_actual_state(measured_teacher):
    t,m=measured_teacher;t.phase='pre'
    a,meta=t.label(m)
    pending=copy.deepcopy(m);pending.last_command=a[0].copy()
    pending.delay_queue[-1]=pending.last_command
    t.phase='pre';b,other=t.label(pending)
    assert other['first_target_tcp']!=pytest.approx(meta['first_target_tcp'])
    assert not np.allclose(a,b)
    assert pending.joints==m.joints


def test_translation_and_orientation_follow_same_geometric_path_fraction():
    from tabletop.closed_loop import local_pose_step
    p=np.array([0.,0.,0.]);goal=np.array([.2,0.,.1]);R=np.eye(3)
    end=Rotation.from_rotvec([0.,0.,.8]).as_matrix()
    position,rotation=local_pose_step(p,R,goal,end)
    translation_fraction=np.linalg.norm(position)/np.linalg.norm(goal)
    rotation_fraction=np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())/.8
    assert translation_fraction==pytest.approx(rotation_fraction)
    assert 0<translation_fraction<1
    assert np.linalg.norm(position)<=.010000001


def test_initial_actual_safety_gate_cannot_be_bypassed_by_closed_loop(measured_teacher):
    t,m=measured_teacher
    m.base_pose[:2]+=np.array([2.,2.])
    labels,meta=t.label(m)
    assert meta['reason']=='initial_gate:envelope'
    assert meta['phase']=='rejected'
    assert labels==pytest.approx(np.repeat(m.last_command[None],8,axis=0))
    m.base_pose[:2]-=np.array([2.,2.])
    _,meta=t.label(m)
    assert meta['reason']=='initial_gate:envelope'  # no in-macro retry


def test_collision_query_carries_actual_grasp_pose_and_restores_unheld_pose(measured_teacher):
    t,m=measured_teacher;p=t.planner
    p.configure(t.state,m.base_pose,m.object_pose,m.joints)
    turn=Rotation.from_euler('xyz',[.03,-.02,.1]).as_quat()
    quaternion=np.r_[turn[3],turn[:3]];delta=np.array([.004,-.003,.08])
    p.collision(p.initial_arm,delta,quaternion)
    assert p.d.qpos[p.objqa:p.objqa+3]==pytest.approx(m.object_pose[:3]+delta)
    assert p.d.qpos[p.objqa+3:p.objqa+7]==pytest.approx(quaternion)
    p.collision(p.initial_arm)
    assert p.d.qpos[p.objqa:p.objqa+7]==pytest.approx(m.object_pose)


def test_grasp_depth_uses_actual_length_and_pad_not_fixed_offset(measured_teacher):
    from dataclasses import replace
    t,m=measured_teacher;p=t.planner
    # Independent lengths span the authorised box range; all use the same live
    # robot pose. At a narrow-face approach the near edge must meet pad rear.
    for length in (.10,.12,.14):
        state=replace(t.state,shape='box',size=(.04,length,.13),object_yaw=0.)
        pose=m.object_pose.copy();pose[3:]=[1.,0.,0.,0.]
        p.configure(state,m.base_pose,pose,m.joints)
        depth,geometry=p.grasp_depth(np.array([0.,1.,0.]))
        assert depth==pytest.approx(length/2-.02,abs=2e-6)
        assert depth-length/2==pytest.approx(geometry['pad_rear_tcp_x'])
        assert depth-length/2-(-.06)==pytest.approx(.04,abs=2e-6)
    # A changed actual quaternion changes projection even with stale nominal yaw.
    turn=Rotation.from_euler('z',.2).as_quat();pose[3:]=np.r_[turn[3],turn[:3]]
    p.configure(state,m.base_pose,pose,m.joints)
    depth,_=p.grasp_depth(np.array([0.,1.,0.]))
    assert depth==pytest.approx(.07*math.cos(.2)+.02*math.sin(.2)-.02,abs=2e-6)


def test_pregrasp_clears_near_face_and_preserves_straight_insert(measured_teacher):
    from dataclasses import replace
    t,m=measured_teacher;p=t.planner;axis=np.array([0.,1.,0.])
    for shape,size in [('box',(.04,.10,.13)),('box',(.04,.14,.13)),('cylinder',(.06,.06,.13))]:
        state=replace(t.state,shape=shape,size=size,object_yaw=0.)
        pose=m.object_pose.copy();pose[3:]=[1.,0.,0.,0.]
        p.configure(state,m.base_pose,pose,m.joints)
        depth,_=p.grasp_depth(axis);distance=p.pregrasp_distance(axis,depth)
        extent=size[1]/2
        pre_near_face=depth+distance-extent
        tolerance=2*p.cfg.ik_pos_tol+.036*math.sin(2*p.cfg.ik_rot_tol)
        assert pre_near_face==pytest.approx(.02+tolerance,abs=2e-6)
        assert 0<distance<.08
        assert p.pregrasp_distance(axis,depth)==pytest.approx(distance)
