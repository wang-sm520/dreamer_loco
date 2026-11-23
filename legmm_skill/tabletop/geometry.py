"""Six-joint IK, side orientation and sampled collision checks at actual base pose.

Accepted means geometrically feasible under rigid grasp transport, NOT rollout
success. Isaac probes must quantify model disagreement before training.
"""
from dataclasses import dataclass, field
import math
from collections import OrderedDict

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .assets import geometry_xml
from .spec import STOW, LOWER, UPPER, PlannerConfig, gate_reason
from .collision import excluded_pair


@dataclass
class Plan:
    accepted: bool
    reason: str
    joints: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    tcp: list = field(default_factory=list)
    width: float | None = None
    twist: float | None = None
    actual_base_pose: list = field(default_factory=list)


def side_heading(state, cfg=PlannerConfig()):
    heading = state.base_yaw + state.bearing
    if state.shape == 'cylinder':
        return heading, state.size[0], 0.
    # Closing axis is box thickness x; approach along box width y (mod pi).
    desired = state.object_yaw + math.pi/2
    error = (desired-heading+math.pi/2) % math.pi-math.pi/2
    twist = float(np.clip(error,-math.pi/12,math.pi/12))
    beta = error - twist
    width = state.size[0]*abs(math.cos(beta)) + state.size[1]*abs(math.sin(beta))
    return heading+twist, width, twist


class GeometryPlanner:
    def __init__(self, robot_path, cfg=PlannerConfig(), cache_models=False):
        self.cfg = cfg
        self.cache_models=cache_models
        self.model_cache=OrderedDict()
        self.robot_path = robot_path
        self.models = {s: mujoco.MjModel.from_xml_string(geometry_xml(robot_path,s)) for s in ('box','cylinder')}
        self.data = {s: mujoco.MjData(m) for s,m in self.models.items()}

    def configure(self, state, base_pose, object_pose=None, joint_state=None):
        # Compile each candidate's true dimensions. Mutating geom_size alone
        # leaves MuJoCo broad-phase bounds stale and can miss real collisions.
        key=(state.shape,tuple(state.size),tuple(state.table_size),state.mass)
        cached=self.model_cache.pop(key,None) if self.cache_models else None
        if cached is None:
            m = mujoco.MjModel.from_xml_string(geometry_xml(self.robot_path,state.shape,state))
            d = mujoco.MjData(m)
        else:m,d=cached
        if self.cache_models:
            self.model_cache[key]=(m,d)
            while len(self.model_cache)>32:self.model_cache.popitem(last=False)
        self.m,self.d,self.state = m,d,state
        self.qa = np.array([m.joint(f'joint{i}').qposadr[0] for i in range(1,7)])
        self.va = np.array([m.joint(f'joint{i}').dofadr[0] for i in range(1,7)])
        self.site = m.site('tcp').id
        self.objqa = m.joint('object_free').qposadr[0]
        self.arm_bodies = {m.body(n).id for n in ('arm_base', *(f'link{i}' for i in range(1,7)), 'left','right','payload')}
        self.finger_bodies = {m.body(n).id for n in ('left','right')}
        d.qpos[:] = m.qpos0
        d.qpos[:7] = base_pose
        for leg in ('FL','FR','RL','RR'):
            for j,value in zip(('hip','thigh','calf'),(-.1 if leg in ('FL','RL') else .1,.9,-1.8)):
                d.qpos[m.joint(f'{leg}_{j}_joint').qposadr[0]] = value
        d.qpos[self.qa] = STOW
        d.qpos[m.joint('endleft').qposadr[0]],d.qpos[m.joint('endright').qposadr[0]] = .036,-.036
        d.qpos[self.objqa:self.objqa+3] = state.center
        d.qpos[self.objqa+3:self.objqa+7] = (math.cos(state.object_yaw/2),0,0,math.sin(state.object_yaw/2))
        if object_pose is not None:
            d.qpos[self.objqa:self.objqa+7] = object_pose
        if joint_state is not None:
            for name,value in joint_state.items():
                d.qpos[m.joint(name).qposadr[0]]=value
        self.object_center=d.qpos[self.objqa:self.objqa+3].copy()
        self.object_quaternion=d.qpos[self.objqa+3:self.objqa+7].copy()
        self.initial_arm=d.qpos[self.qa].copy()
        mujoco.mj_forward(m,d)

    def fk(self,q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m,self.d)
        mujoco.mj_comPos(self.m,self.d)
        return self.d.site_xpos[self.site].copy(),self.d.site_xmat[self.site].reshape(3,3).copy()

    def grasp_depth(self, axis):
        """Place the pad's rear edge at the measured near face of a box.

        Read the compiled pad and palm geometry in TCP coordinates. The old
        fixed 20 mm offset ignored box length and left only 20--25 mm palm
        clearance before loaded 4/0 tilt. Full pad engagement leaves the full
        physical pad-to-palm gap, without moving to the object's centre.
        """
        if self.state.shape != 'box':
            return 0., {}
        tcp,R = self.fk(self.initial_arm)
        gid=self.m.geom('left_finger_pad').id
        pad_center=R.T@(self.d.geom_xpos[gid]-tcp)
        pad_half=abs(R.T@self.d.geom_xmat[gid].reshape(3,3))@self.m.geom_size[gid]
        object_R=Rotation.from_quat(self.object_quaternion[[1,2,3,0]]).as_matrix()
        extent=float(abs(np.asarray(axis)@object_R)@(np.asarray(self.state.size)/2))
        # Near object face coincides with the back edge of the long finger pad.
        near=float(pad_center[0]-pad_half[0])
        depth=extent+near
        return depth,dict(projected_half_length=extent,pad_rear_tcp_x=near,
            pad_front_tcp_x=float(pad_center[0]+pad_half[0]),grasp_depth=depth)

    def pregrasp_distance(self, axis, depth):
        """Align just outside the object before a straight, collision-tested insert.

        The old fixed 80 mm stroke repeated needless straight travel after the
        long alignment phase. Here the near face clears the actual pad front,
        with the existing measured arrival tolerances included geometrically.
        No control speed or task timeout changes.
        """
        tcp,R=self.fk(self.initial_arm)
        gid=self.m.geom('left_finger_pad').id
        center=R.T@(self.d.geom_xpos[gid]-tcp)
        half=abs(R.T@self.d.geom_xmat[gid].reshape(3,3))@self.m.geom_size[gid]
        object_R=Rotation.from_quat(self.object_quaternion[[1,2,3,0]]).as_matrix()
        projected=np.asarray(axis)@object_R
        if self.state.shape=='box':
            extent=float(abs(projected)@(np.asarray(self.state.size)/2))
        else:
            extent=float(self.state.size[0]/2*np.linalg.norm(projected[:2])+
                         self.state.size[2]/2*abs(projected[2]))
        front=float(center[0]+half[0])
        tolerance=2*self.cfg.ik_pos_tol+UPPER[6]/2*math.sin(2*self.cfg.ik_rot_tol)
        return extent+front+tolerance-depth

    def ik(self, target, rotation, seed):
        q = np.clip(seed,LOWER[:6],UPPER[:6])
        for _ in range(self.cfg.ik_iterations):
            p,R = self.fk(q)
            er = Rotation.from_matrix(rotation @ R.T).as_rotvec()
            ep = target-p
            if np.linalg.norm(ep) <= self.cfg.ik_pos_tol and np.linalg.norm(er) <= self.cfg.ik_rot_tol:
                return q
            jp,jr = np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv))
            mujoco.mj_jacSite(self.m,self.d,jp,jr,self.site)
            J = np.vstack((jp[:,self.va],jr[:,self.va]))
            delta = J.T @ np.linalg.solve(J@J.T+1e-4*np.eye(6),np.r_[ep,er])
            q = np.clip(q+np.clip(delta,-.15,.15),LOWER[:6],UPPER[:6])
        return None

    def collision(self,q,object_delta=None,object_quaternion=None):
        self.d.qpos[self.qa] = q
        self.d.qpos[self.objqa:self.objqa+3] = self.object_center + (np.zeros(3) if object_delta is None else object_delta)
        self.d.qpos[self.objqa+3:self.objqa+7] = self.object_quaternion if object_quaternion is None else object_quaternion
        mujoco.mj_forward(self.m,self.d)
        m,d = self.m,self.d
        object_body,table_body = m.body('object').id,m.body('table').id
        for contact in d.contact[:d.ncon]:
            if contact.dist >= -1e-5:
                continue
            a,b = (int(m.geom_bodyid[g]) for g in (contact.geom1,contact.geom2))
            pair = {a,b}
            # Body0=floor. Internal dog collisions and adjacent arm geometry
            # are not environment collisions. Arm-vs-dog is checked explicitly.
            if object_body in pair:
                other = b if a == object_body else a
                if other == table_body or other in self.finger_bodies:
                    continue
                return 'object_robot' if other else 'object_floor'
            if table_body in pair:
                return 'robot_table'
            if a in self.arm_bodies or b in self.arm_bodies:
                if a in self.arm_bodies and b in self.arm_bodies:
                    if excluded_pair(m.body(a).name,m.body(b).name,()):
                        continue
                    return 'arm_self'
                arm = a if a in self.arm_bodies else b
                dog = b if a in self.arm_bodies else a
                # Bolted arm_base necessarily overlaps the mounting body.
                if arm == m.body('arm_base').id and dog == m.body('base_link').id:
                    continue
                return 'arm_floor' if dog == 0 else 'arm_dog'
        return None

    def plan(self,state,base_pose,actual_velocity=None,object_pose=None,joint_state=None):
        self.configure(state,base_pose,object_pose,joint_state)
        w,x,y,z = base_pose[3:7]
        yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        shoulder = self.d.xpos[self.m.body('link2').id,:2].copy()
        why = gate_reason(state,shoulder,base_pose[:2],yaw,actual_velocity,self.object_center)
        if why:
            return Plan(False,why)
        # Wrist compensation is relative to the live shoulder-to-object ray.
        delta = self.object_center[:2]-shoulder
        from dataclasses import replace
        live = replace(state,base_yaw=yaw,bearing=math.atan2(delta[1],delta[0])-yaw)
        heading,width,twist = side_heading(live,self.cfg)
        if width >= UPPER[6]:
            return Plan(False,'opening',width=width,twist=twist)
        c,s = math.cos(heading),math.sin(heading)
        rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        axis = np.array([c,s,0])
        depth,_ = self.grasp_depth(axis)
        goal = self.object_center-depth*axis
        pre = goal-self.pregrasp_distance(axis,depth)*axis
        lift = goal+np.array([0,0,self.cfg.lift])
        current = self.initial_arm.copy()
        initial,initial_R = self.fk(current)
        # Raise at STOW before aligning over the table, then approach horizontally.
        clearance = initial.copy()
        clearance[2] = max(initial[2],goal[2])
        plan = Plan(True,'geometry_only',width=width,twist=twist,actual_base_pose=list(base_pose))
        targets = [('clear',clearance,initial_R),('pre',pre,rotation),('grasp',goal,rotation),('lift',lift,rotation)]
        for phase,target,R in targets:
            start_p,start_R = self.fk(current)
            # Cartesian subdivision plus joint collision samples guards arcs.
            n = max(1,int(math.ceil(np.linalg.norm(target-start_p)/.01)))
            for i in range(1,n+1):
                fraction = i/n
                interp_R = Rotation.from_rotvec(fraction*Rotation.from_matrix(R@start_R.T).as_rotvec()).as_matrix() @ start_R
                point = start_p+(target-start_p)*fraction
                solved = self.ik(point,interp_R,current)
                if solved is None:
                    return Plan(False,'ik:'+phase,width=width,twist=twist,actual_base_pose=list(base_pose))
                count = max(1,int(math.ceil(np.max(abs(solved-current))/self.cfg.path_joint_step)))
                for j in range(1,count+1):
                    q = current+(solved-current)*j/count
                    p,_ = self.fk(q)
                    reason = self.collision(q,p-goal if phase=='lift' else None)
                    if reason:
                        return Plan(False,reason+':'+phase,width=width,twist=twist,actual_base_pose=list(base_pose))
                    plan.joints.append(q.tolist()); plan.phases.append(phase); plan.tcp.append(p.tolist())
                current = solved
        return plan


def sample_reachable(seed, planner, base_pose_fn, record, residual=False, max_attempts=2000):
    """Audit every rejection. Never call an execution-success oracle."""
    from .spec import sample_candidate
    for attempt in range(max_attempts):
        state = sample_candidate(seed,attempt,residual)
        plan = planner.plan(state,base_pose_fn(state))
        record(state,plan)
        if plan.accepted:
            return state,plan
    raise RuntimeError(f'No geometrically reachable state for seed={seed}; preserve rejection log')
