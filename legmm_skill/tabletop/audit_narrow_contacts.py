"""Actual bilateral contact points and box axes, without changing episode verdicts."""
import argparse,json,math
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    records=[]
    for run in a.runs:
        result=json.loads((run/'result.json').read_text());states={x['seed']:x for x in result['states']}
        selected={k:dict(seed=k,shape=s['shape'],size=s['size'],first_bilateral=None,first_lift=None,first_success=None) for k,s in states.items()}
        last_rows={}
        for line in (run/'teacher_trace.jsonl').open():
            row=json.loads(line);s=selected[row['seed']]
            if s['shape']!='box':continue
            last_rows[row['seed']]=row
            forces=row['forces_world_table_object_floor'];bilateral=all(np.linalg.norm(forces[b][1])>1 for b in ['left','right'])
            events=[]
            if bilateral and s['first_bilateral'] is None:events.append('first_bilateral')
            if row['schedule']['phase']=='lift' and s['first_lift'] is None:events.append('first_lift')
            if row['outcome']=='success' and s['first_success'] is None:events.append('first_success')
            if not events:continue
            rot=np.asarray(row['object_rotation_in_tcp']);half=np.asarray(s['size'])/2
            angle=math.degrees(math.acos(float(np.clip(abs(rot[1,0]),0,1))))
            approach=math.degrees(math.acos(float(np.clip(abs(rot[0,1]),0,1))))
            points={}
            for b in ['left','right']:
                point=row.get('contact_points_world_object',{}).get(b,[None,None])[1]
                if point is not None:
                    local=np.asarray(point['object']);dist=abs(abs(local)-half)
                    points[b]=dict(object=local.tolist(),distance_to_abs_faces=dist.tolist(),nearest_face_axis='xyz'[int(np.argmin(dist))])
            evidence=dict(time_s=row['time_s'],phase=row['schedule']['phase'],bilateral=bilateral,
                closing_axis_error_deg=angle,approach_normal_error_deg=approach,
                object_in_tcp=row['object_in_tcp'],opening=row['opening'],contact_points=points)
            for event in events:s[event]=evidence
        for i,e in enumerate(result['episodes']):selected[e['seed']].update(success=e['success'],reason=e['reason'],result_path=str(run/'result.json'))
        for seed,record in selected.items():
            if record['shape']=='box' and record['success'] and record['first_success'] is None:
                row=last_rows[seed];rot=np.asarray(row['object_rotation_in_tcp'])
                record['success_frame_not_sampled']=True
                record['last_before_success']=dict(time_s=row['time_s'],held_steps=row['held_steps'],
                    closing_axis_error_deg=math.degrees(math.acos(float(np.clip(abs(rot[1,0]),0,1)))),
                    approach_normal_error_deg=math.degrees(math.acos(float(np.clip(abs(rot[0,1]),0,1)))),
                    contact_points=row.get('contact_points_world_object'))
        records.extend(selected.values())
    stats={}
    for event in ['first_bilateral','first_lift','first_success']:
        values=[x[event]['closing_axis_error_deg'] for x in records if x[event]]
        stats[event]=dict(count=len(values),median_degrees=float(np.median(values)) if values else None,
            max_degrees=max(values) if values else None,over15_degrees=sum(v>15 for v in values))
    a.out.write_text(json.dumps(dict(summary=stats,successful_boxes=sum(x['shape']=='box' and x['success'] for x in records),success_frames_not_sampled=sum(x.get('success_frame_not_sampled',False) for x in records),episodes=records,verdicts_unchanged=True),indent=2)+'\n');print(json.dumps(stats))


if __name__=='__main__':main()
