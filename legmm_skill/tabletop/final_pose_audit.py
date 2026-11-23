"""Read-only actual hand angles for final physical evidence, never actor input."""
import json,math
import numpy as np
from scipy.spatial.transform import Rotation


def pose_angles(tcp_rotation,object_rotation,shoulder,object_position):
    approach=np.asarray(tcp_rotation)[:,0];closing=np.asarray(tcp_rotation)[:,1]
    object_rotation=np.asarray(object_rotation);delta=np.asarray(object_position)-np.asarray(shoulder)
    planar=np.linalg.norm(approach[:2]);compensation=None
    if planar>1e-6:
        difference=math.atan2(approach[1],approach[0])-math.atan2(delta[1],delta[0])
        compensation=math.degrees((difference+math.pi)%(2*math.pi)-math.pi)
    angle=lambda value:math.degrees(math.acos(float(np.clip(abs(value),0,1))))
    return dict(wrist_heading_compensation_deg=compensation,
        horizontal_approach_tilt_deg=math.degrees(math.asin(float(np.clip(abs(approach[2]),0,1)))),
        closing_axis_error_deg=angle(np.dot(closing,object_rotation[:,0])),
        approach_normal_error_deg=angle(np.dot(approach,object_rotation[:,1])))


class ActualPoseRecorder:
    def __init__(self,out,seed):self.out=out;self.seed=seed;self.rows=[]
    def observe(self,scene,index):
        robot=scene['robot'];target=scene['target'];link=robot.find_bodies('link6')[0][0];shoulder=robot.find_bodies('link2')[0][0]
        quaternions=robot.data.body_link_quat_w[:,link].cpu().numpy()
        objects=target.data.root_pos_w.cpu().numpy();object_quats=target.data.root_quat_w.cpu().numpy()
        shoulders=robot.data.body_link_pos_w[:,shoulder].cpu().numpy()
        relative=Rotation.from_quat([0,-.707108,0,.707105]).as_matrix()
        hands=Rotation.from_quat(quaternions[:,[1,2,3,0]]).as_matrix()@relative
        object_rotations=Rotation.from_quat(object_quats[:,[1,2,3,0]]).as_matrix()
        for i in range(len(objects)):
            self.rows.append(dict(seed=self.seed+i,time_s=index*.16,**pose_angles(hands[i],object_rotations[i],shoulders[i],objects[i])))
    def finish(self):
        with (self.out/'actual_pose_audit.jsonl').open('x') as file:
            for row in self.rows:file.write(json.dumps(row)+'\n')
        self.rows=[]


def summarize_successful_poses(driver_out,episodes):
    outcomes={row['seed']:row for row in episodes if row['success']};closest={}
    for path in sorted(driver_out.glob('seed_*/actual_pose_audit.jsonl')):
        for line in path.open():
            row=json.loads(line);outcome=outcomes.get(row['seed'])
            if outcome is None or row['time_s']>outcome['steps']*.01+1e-9:continue
            closest[row['seed']]=row
    rows=[]
    for seed,outcome in outcomes.items():
        row=closest.get(seed)
        if row is not None:rows.append(dict(row,seconds_before_success=outcome['steps']*.01-row['time_s']))
    values=[abs(row['wrist_heading_compensation_deg']) for row in rows if row['wrist_heading_compensation_deg'] is not None]
    return dict(successful_episodes=len(outcomes),sampled_successful_episodes=len(rows),
        max_heading_compensation_abs_deg=max(values) if values else None,
        heading_compensation_over15_count=sum(value>15 for value in values),
        max_horizontal_tilt_deg=max((row['horizontal_approach_tilt_deg'] for row in rows),default=None),
        max_sample_gap_s=max((row['seconds_before_success'] for row in rows),default=None),
        read_only_diagnostic_not_a_new_success_rule=True,episodes=rows)
