"""Privileged teacher queried on measured state, with no time-indexed path labels.

Phase is controller memory; transitions require measured progress/contact. Queue
state is supplied by the real actuator interface, never reset by a teacher query.
Only returned native actions may enter supervision; truth remains training-side.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .spec import LOWER, UPPER, STEP_LENGTH, PlannerConfig,gate_reason
from .geometry import side_heading


def local_pose_step(position, rotation, goal, goal_rotation):
    """One common path fraction for translation and rotation (original SLERP path).

    Independent caps change the geometric path and can add large joint travel.
    Native joint speed/limits are enforced separately on every output target.
    """
    delta=goal-position
    rotvec=Rotation.from_matrix(goal_rotation@rotation.T).as_rotvec()
    fraction=1/max(1.,np.linalg.norm(delta)/.01,np.linalg.norm(rotvec)/.05)
    return position+fraction*delta,Rotation.from_rotvec(fraction*rotvec).as_matrix()@rotation


@dataclass
class MeasuredState:
    base_pose: np.ndarray
    object_pose: np.ndarray
    joints: dict
    velocity: np.ndarray
    last_command: np.ndarray
    delay_queue: np.ndarray
    finger_contact: tuple
    elapsed_s: float
    forbidden_contact: bool = False

    def validate(self):
        arrays=[self.base_pose,self.object_pose,self.velocity,self.last_command,self.delay_queue]
        if any(not np.isfinite(x).all() for x in arrays):
            raise FloatingPointError('Nonfinite measured teacher state')
        if np.shape(self.delay_queue)!=(11,7):
            raise ValueError('Teacher requires the live 100 ms actuator queue')
        if any(np.shape(x)!=(7,) for x in (self.base_pose,self.object_pose,self.last_command)):
            raise ValueError('Invalid measured pose/command')


class ClosedLoopTeacher:
    def __init__(self, planner, state, cfg=PlannerConfig()):
        self.planner,self.state,self.cfg=planner,state,cfg
        self.phase='clear'
        self.grasp_relative=None
        self.grasp_rotation_relative=None
        self.gate_checked=False
        self.initial_gate_reason=None

    def label(self, measured, horizon=8):
        measured.validate()
        p=self.planner;cfg=self.cfg;m=measured
        # Use actual full quaternion for geometry; live normal projection supplies
        # yaw, never the nominal sampling yaw. Tilt cannot be repaired by top grasp.
        Robj=Rotation.from_quat(np.r_[m.object_pose[4:7],m.object_pose[3]]).as_matrix()
        yaw=math.atan2(Robj[1,0],Robj[0,0])
        state=replace(self.state,object_xy=tuple(m.object_pose[:2]),object_yaw=yaw)
        p.configure(state,m.base_pose,m.object_pose,m.joints)
        q=p.initial_arm.copy();tcp,Rtcp=p.fk(q)
        # The previous target is already in the 100 ms actuator pipeline.
        # Progress from that pending reference while testing phase completion
        # against actual TCP/contact. Restarting each target at the delayed
        # measured TCP repeatedly spends the same actuator latency.
        if not np.any(np.max(abs(m.delay_queue-m.last_command),axis=1)<1e-5):
            raise ValueError('Last command is absent from the live delay queue')
        pending_tcp,pending_R=p.fk(m.last_command[:6])
        base_R=Rotation.from_quat(np.r_[m.base_pose[4:7],m.base_pose[3]]).as_matrix()
        shoulder=p.d.xpos[p.m.body('link2').id].copy()
        delta=m.object_pose[:2]-shoulder[:2]
        base_yaw=math.atan2(base_R[1,0],base_R[0,0])
        live=replace(state,base_yaw=base_yaw,bearing=math.atan2(delta[1],delta[0])-base_yaw)
        heading,width,twist=side_heading(live,cfg)
        if not self.gate_checked:
            self.initial_gate_reason=gate_reason(live,shoulder[:2],m.base_pose[:2],base_yaw,m.velocity,m.object_pose[:3])
            self.gate_checked=True
        if self.initial_gate_reason:
            self.phase='rejected'
            return np.repeat(m.last_command[None],horizon,axis=0),dict(phase=self.phase,
                reason='initial_gate:'+self.initial_gate_reason,live_object_yaw=yaw,
                target_tcp=tcp.tolist(),first_target_tcp=tcp.tolist(),width=float(width),twist=float(twist),
                opening=float(m.joints['endleft']-m.joints['endright']),pending_queue_last=m.delay_queue[-1].tolist())
        axis=np.array([math.cos(heading),math.sin(heading),0.])
        rotation=np.array([[axis[0],-axis[1],0],[axis[1],axis[0],0],[0,0,1.]])
        depth,depth_geometry=p.grasp_depth(axis)
        goal=m.object_pose[:3]-depth*axis
        pre_distance=p.pregrasp_distance(axis,depth)
        pre=goal-pre_distance*axis
        opening=m.joints['endleft']-m.joints['endright']
        reason=None
        if m.forbidden_contact or m.elapsed_s>=15.:
            reason='terminal_safety_hold'
        elif Robj[2,2]<math.cos(cfg.ik_rot_tol) and self.phase not in ('lift','hold'):
            reason='tilted_object_hold'
        elif width>=UPPER[6]:
            reason='opening_hold'
        target=None
        # At most one transition per real observation, no hypothetical future
        # transition based on a nominal timestamp or an unexecuted label.
        orient_error=np.linalg.norm(Rotation.from_matrix(rotation@Rtcp.T).as_rotvec())
        reached=lambda x:np.linalg.norm(tcp-x)<=2*cfg.ik_pos_tol
        if self.phase=='clear':
            target=tcp.copy();target[2]=max(tcp[2],goal[2]);rotation=Rtcp
            if reached(target):self.phase='pre'
        elif self.phase=='pre':
            target=pre
            if reached(pre) and orient_error<=2*cfg.ik_rot_tol:self.phase='grasp'
        elif self.phase=='grasp':
            target=goal
            if reached(goal) and orient_error<=2*cfg.ik_rot_tol:self.phase='close'
        elif self.phase=='close':
            target=goal
            if all(m.finger_contact):
                self.grasp_relative=Rtcp.T@(m.object_pose[:3]-tcp)
                self.grasp_rotation_relative=Rtcp.T@Robj
                self.phase='lift'
        elif self.phase in ('lift','hold'):
            if not all(m.finger_contact):reason='lost_grip_hold'
            if self.grasp_relative is None:raise RuntimeError('Lift requires measured bilateral grasp')
            target=tcp.copy()
            target[2]=max(tcp[2],state.support+state.size[2]/2+cfg.lift-(m.object_pose[:3]-tcp)[2])
            rotation=Rtcp
            if reached(target):self.phase='hold'
        if target is None:target=tcp.copy()
        # Bound Cartesian progress before IK and check every proposed native
        # command. This recomputes from actual robot/object state on each query.
        closing=self.phase in ('close','lift','hold')
        previous=np.asarray(m.last_command).copy();labels=[];first_target=None
        for _ in range(horizon):
            # A chunk contains eight successive control targets. Integrate the
            # local path at EACH target, not once per four-target decision.
            # The phase and object goals still come from the real observation.
            predicted_tcp,predicted_R=p.fk(previous[:6])
            next_tcp,next_R=local_pose_step(predicted_tcp,predicted_R,target,rotation)
            if first_target is None:first_target=next_tcp.copy()
            solved=None if reason else p.ik(next_tcp,next_R,previous[:6])
            if solved is None and reason is None:reason='ik_hold'
            desired=np.r_[previous[:6] if solved is None else solved,0. if closing else .072]
            delta=desired-previous
            delta=delta/max(1.,float(np.max(abs(delta)/STEP_LENGTH)))
            candidate=np.clip(previous+delta,LOWER,UPPER)
            object_delta=None;object_quaternion=None
            if self.phase in ('lift','hold') and self.grasp_relative is not None:
                carried_tcp,carried_R=p.fk(candidate[:6])
                actual_relative=Rtcp.T@(m.object_pose[:3]-tcp)
                actual_relative_rotation=Rtcp.T@Robj
                object_delta=carried_tcp+carried_R@actual_relative-m.object_pose[:3]
                carried_q=Rotation.from_matrix(carried_R@actual_relative_rotation).as_quat()
                object_quaternion=np.r_[carried_q[3],carried_q[:3]]
            collision=None if reason else p.collision(candidate[:6],object_delta,object_quaternion)
            if collision:
                reason='collision_hold:'+collision
            if reason:
                # A safe hold preserves the pending commands; it does not jump
                # to a stale reference or reset simulator/controller state.
                candidate=previous.copy()
            labels.append(candidate);previous=candidate
        return np.array(labels),dict(phase=self.phase,reason=reason,live_object_yaw=yaw,
            pregrasp_distance=float(pre_distance),grasp_depth_geometry=depth_geometry,target_tcp=target.tolist(),width=float(width),twist=float(twist),opening=float(opening),
            first_target_tcp=first_target.tolist(),
            pending_queue_last=m.delay_queue[-1].tolist())
