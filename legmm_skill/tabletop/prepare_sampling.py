"""Prepare deterministic geometric cohorts from a measured free-standing template.

These are candidates: Isaac still rechecks each real settled pose. No execution
outcome enters rejection sampling. All final validation seeds must be counted.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .spec import OUTPUT,InitialState,sample_candidate
from .assets import build_robot
from .geometry import GeometryPlanner
from .storage import StorageGuard


def calibrated_pose(state, reference, measured_pose):
    angle=state.base_yaw-reference.base_yaw
    turn=Rotation.from_euler('z',angle)
    displacement=np.asarray(measured_pose[:3])-np.r_[reference.base_xy,0.]
    position=np.r_[state.base_xy,0.]+turn.apply(displacement)
    w,x,y,z=measured_pose[3:7]
    quat=(turn*Rotation.from_quat([x,y,z,w])).as_quat()
    return np.r_[position,quat[3],quat[:3]]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--template',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=10000);p.add_argument('--count',type=int,default=256)
    p.add_argument('--residual-velocity',action='store_true');a=p.parse_args()
    if not a.out.resolve().is_relative_to(OUTPUT):raise ValueError('Output outside W8 task')
    a.out.mkdir(exist_ok=False);guard=StorageGuard(OUTPUT,Path(__file__).parent);guard.check()
    raw=json.loads(a.template.read_text());reference=InitialState(**raw['states'][0])
    pose=raw['initial_base_pose'][0];joints=dict(zip(raw['joint_names'],raw['initial_joint_positions'][0]))
    planner=GeometryPlanner(build_robot(OUTPUT/'models'));audit=[]
    (a.out/'status.json').write_text(json.dumps(dict(completed=False,template=str(a.template),geometry_only=True)))
    for seed in range(a.seed,a.seed+a.count):
        for attempt in range(2000):
            state=sample_candidate(seed,attempt,a.residual_velocity)
            plan=planner.plan(state,calibrated_pose(state,reference,pose),state.velocity,joint_state=joints)
            audit.append(dict(state=state.json(),accepted=plan.accepted,reason=plan.reason))
            if plan.accepted:break
        else:
            (a.out/'sampling_failed.json').write_text(json.dumps(audit));raise RuntimeError(f'No reachable candidate for {seed}')
        if (seed-a.seed+1)%32==0:
            batch=audit;audit=[];guard.check(2000000)
            (a.out/f'sampling_{seed-31}.json').write_text(json.dumps(batch,indent=1))
            print(json.dumps(dict(seed=seed,attempt=attempt)),flush=True)
    if audit:raise ValueError('Count must be a multiple of 32')
    (a.out/'status.json').write_text(json.dumps(dict(completed=True,template=str(a.template),geometry_only=True,
        seed=a.seed,count=a.count,residual_velocity=a.residual_velocity)))


if __name__=='__main__':main()
