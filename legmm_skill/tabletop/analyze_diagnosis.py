"""Offline trace analysis; no test access and no rollout-dependent resampling."""
import argparse,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from .spec import InitialState,CONTROL_DT
from .assets import build_robot
from .geometry import GeometryPlanner
from .spec import OUTPUT

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.run/'result.json').read_text());traces=defaultdict(list)
    for line in (a.run/'teacher_trace.jsonl').open():
        r=json.loads(line);traces[r['seed']].append(r)
    planner=GeometryPlanner(build_robot(OUTPUT/'models'));rows=[]
    for i,(s,e) in enumerate(zip(result['states'],result['episodes'])):
        state=InitialState(**s);planner.configure(state,result['initial_base_pose'][i],result['initial_object_pose'][i],dict(zip(result['joint_names'],result['initial_joint_positions'][i])))
        trace=traces[s['seed']];stages={}
        for phase in dict.fromkeys(r['schedule']['phase'] for r in trace):
            seq=[r for r in trace if r['schedule']['phase']==phase];last=seq[-1]
            target,R=planner.fk(last['command'][:6]);forces=np.array([r['forces_world_table_object_floor']['left'][1] for r in seq]);right=np.array([r['forces_world_table_object_floor']['right'][1] for r in seq])
            palm=np.array([r['forces_world_table_object_floor']['link6'][1] for r in seq])
            stages[phase]=dict(start_s=seq[0]['time_s'],end_s=last['time_s'],command_tcp=target.tolist(),command_tcp_rotation=R.tolist(),actual_tcp_rotation=last["tcp_rotation"],actual_tcp=last['tcp_position'],tcp_error_m=float(np.linalg.norm(target-last['tcp_position'])),actual_fingers=last['actual_fingers'],opening=last['opening'],command_opening=last['command'][6],object_in_tcp=last['object_in_tcp'],object_rotation_in_tcp=last['object_rotation_in_tcp'],finger_object_peak_N=[float(np.linalg.norm(forces,axis=1).max()),float(np.linalg.norm(right,axis=1).max())],palm_object_peak_N=float(np.linalg.norm(palm,axis=1).max()),last_slip_from_close=last['slip_from_close'])
        lift=stages.get('lift',{});close=stages.get('close',{})
        category=e['reason']
        if category=='timeout':
            category=('rejected' if not result['live_geometry'][i]['accepted'] else 'not_reached' if 'close' not in stages else 'hold_insufficient' if e['peak_lift']>=.05 else 'not_gripped' if e['peak_lift']<.005 else 'not_lifted_5cm')
        rows.append(dict(seed=s['seed'],shape=s['shape'],reason=e['reason'],timeout_category=category,state=s,episode=e,command_duration_s=result['trajectory_control_points'][i]*CONTROL_DT,available_action_s=15.,stages=stages))
    a.out.write_text(json.dumps(dict(source=str(a.run),counts=dict(Counter(r['shape']+':'+r['timeout_category'] for r in rows)),episodes=rows),indent=2))
    print(json.dumps(dict(counts=dict(Counter(r['shape']+':'+r['timeout_category'] for r in rows)))))
if __name__=='__main__':main()
