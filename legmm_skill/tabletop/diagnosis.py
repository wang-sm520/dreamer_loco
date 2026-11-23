"""Read-only, sparse physical trace for bounded teacher causal diagnosis."""
import json
import numpy as np
from scipy.spatial.transform import Rotation

class TeacherTrace:
    def __init__(self, out, robot, target, origins, states, arm_ids, finger_ids):
        self.file=(out/'teacher_trace.jsonl').open('x')
        self.robot,self.target,self.origins=robot,target,origins
        self.states,self.arm_ids,self.finger_ids=states,arm_ids,finger_ids
        self.link=robot.find_bodies('link6')[0][0]
        self.relative_initial={}
    def record(self,frame,command,scene,outcomes,schedules):
        d=self.robot.data
        pos=(d.body_link_pos_w[:,self.link]-self.origins).cpu().numpy()
        quat=d.body_link_quat_w[:,self.link].cpu().numpy()
        op=(self.target.data.root_pos_w-self.origins).cpu().numpy()
        oq=self.target.data.root_quat_w.cpu().numpy()
        joints=d.joint_pos.cpu().numpy();cmd=command.cpu().numpy()
        for i,state in enumerate(self.states):
            R=Rotation.from_quat(quat[i,[1,2,3,0]]).as_matrix()
            # Exact W2 TCP site: link6 translation + (0,0,.005), rotation Ry(-pi/2).
            T=R@Rotation.from_quat([0,-.707108,0,.707105]).as_matrix()
            tcp=pos[i]+R@np.array([0,0,.005])
            rel=T.T@(op[i]-tcp)
            relR=T.T@Rotation.from_quat(oq[i,[1,2,3,0]]).as_matrix()
            schedule=schedules[i];index=min(frame//4,len(schedule)-1)
            label=schedule[index] if schedule else dict(phase='rejected')
            if label['phase']=='close' and i not in self.relative_initial:self.relative_initial[i]=rel.copy()
            forces={};contact_points={}
            for b in ('left','right','link6'):
                sensor=scene['contact_'+b]
                forces[b]=sensor.data.force_matrix_w[i,0,:3].cpu().tolist()
                points=sensor.data.contact_pos_w[i,0,:3].cpu().numpy()-self.origins[i].cpu().numpy()
                objR=Rotation.from_quat(oq[i,[1,2,3,0]]).as_matrix()
                contact_points[b]=[dict(world=v.tolist(),object=(objR.T@(v-op[i])).tolist()) if np.isfinite(v).all() else None for v in points]
            row=dict(seed=state.seed,shape=state.shape,time_s=(frame+1)*.01,frame=frame,
                schedule=label,command=cmd[i].tolist(),actual_arm=joints[i,self.arm_ids].tolist(),
                actual_fingers=joints[i,self.finger_ids].tolist(),opening=float(joints[i,self.finger_ids[0]]-joints[i,self.finger_ids[1]]),
                tcp_position=tcp.tolist(),tcp_rotation=T.tolist(),object_pose=[*op[i].tolist(),*oq[i].tolist()],
                object_in_tcp=rel.tolist(),object_rotation_in_tcp=relR.tolist(),
                slip_from_close=(rel-self.relative_initial[i]).tolist() if i in self.relative_initial else None,
                contact_points_world_object=contact_points,forces_world_table_object_floor=forces,outcome=outcomes[i].reason,held_steps=outcomes[i].held_steps)
            self.file.write(json.dumps(row)+'\n')
        self.file.flush()
