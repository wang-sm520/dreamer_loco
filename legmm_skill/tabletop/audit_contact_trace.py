"""Summarise true 4/0 training contact traces without selecting validation states."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    a=argparse.ArgumentParser();a.add_argument('run',type=Path);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    result=json.loads((args.run/'result.json').read_text());states={s['seed']:s for s in result['states']}
    summary={s:dict(state=v,transitions=[],first_palm_contact=None) for s,v in states.items()}
    for line in (args.run/'teacher_trace.jsonl').open():
        row=json.loads(line);s=summary[row['seed']];phase=row['schedule']['phase']
        rel=np.asarray(row['object_in_tcp']);rot=np.asarray(row['object_rotation_in_tcp'])
        half=np.asarray(s['state']['size'])/2
        extent=abs(rot)@half
        evidence=dict(time_s=row['time_s'],phase=phase,object_in_tcp=rel.tolist(),object_rotation_in_tcp=rot.tolist(),
            near_object_face_tcp_x=float(rel[0]-extent[0]),palm_front_tcp_x=-.06,
            axial_palm_clearance=float(rel[0]-extent[0]+.06),opening=row['opening'],
            slip_from_close=row['slip_from_close'],forces=row['forces_world_table_object_floor'],
            contact_points=row.get('contact_points_world_object'),actual_arm=row['actual_arm'],command=row['command'])
        if not s['transitions'] or s['transitions'][-1]['phase']!=phase:s['transitions'].append(evidence)
        if s['first_palm_contact'] is None and np.linalg.norm(row['forces_world_table_object_floor']['link6'][1])>1:s['first_palm_contact']=evidence
    args.out.write_text(json.dumps(dict(source=str(args.run),episodes=result['episodes'],diagnostics=summary),indent=2)+'\n')


if __name__=='__main__':main()
