"""CPU-only approximate replay to separate teacher-label cost from GPU contention.

Only arm/object/pending commands are exact log values; base and leg state are
nominal approximations. This is a timing diagnostic, never physical evidence.
"""
import argparse,cProfile,json,math,pstats,time
from collections import Counter
from pathlib import Path
import numpy as np
from .closed_loop import ClosedLoopTeacher,MeasuredState
from .learning_geometry import LearningGeometryPlanner
from .spec import InitialState,OUTPUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False)
    rows=[json.loads(line) for line in (a.episode/'closed_loop_labels.jsonl').open()]
    sampling=json.loads((a.episode/'sampling.json').read_text())
    states={r['state']['seed']:InitialState(**r['state']) for r in sampling if r['accepted']}
    latest=max(r['frame'] for r in rows);planner=LearningGeometryPlanner(OUTPUT/'models/a2_tabletop_mesh_shoulder_v2.xml',len(states))
    result=dict(formal_training=False,Isaac_App_started=False,environment_interactions=0,
        exact_fields=['arm_joint_position','object_pose','pending_last_command'],approximated_fields=['nominal_base_pose','nominal_leg_joints','repeat_pending_delay_queue','no_finger_contact'],frames=[])
    for frame in [0,latest]:
        selected=[r for r in rows if r['frame']==frame];counts=Counter();prof=cProfile.Profile();prof.enable();start=time.monotonic()
        for r in selected:
            state=states[r['seed']];base=np.r_[state.base_xy,.3763,math.cos(state.base_yaw/2),0,0,math.sin(state.base_yaw/2)]
            joints=dict(zip([f'joint{i}' for i in range(1,7)],r['measured_arm']));opening=r['metadata']['opening'];joints.update(endleft=opening/2,endright=-opening/2)
            for leg in ('FL','FR','RL','RR'):
                joints.update({f'{leg}_{joint}_joint':value for joint,value in zip(('hip','thigh','calf'),(-.1 if leg in ('FL','RL') else .1,.9,-1.8))})
            previous=np.asarray(r['metadata']['pending_queue_last']);teacher=ClosedLoopTeacher(planner,state)
            teacher.phase=r['metadata']['phase'];teacher.gate_checked=True
            if teacher.phase in ('lift','hold','rejected'):
                counts['skipped_unreconstructable_phase']+=1;continue
            label,metadata=teacher.label(MeasuredState(base,np.asarray(r['measured_object_pose']),joints,np.zeros(3),previous,np.repeat(previous[None],11,axis=0),(False,False),frame*.01,False))
            counts[str(metadata['reason'])]+=1
        elapsed=time.monotonic()-start;prof.disable();prof.dump_stats(a.out/f'frame_{frame}.prof')
        with (a.out/f'frame_{frame}_profile.txt').open('w') as file:pstats.Stats(prof,stream=file).sort_stats('cumtime').print_stats(25)
        result['frames'].append(dict(frame=frame,labels=len(selected),elapsed_s=elapsed,reasons=dict(counts)))
        (a.out/'status.json').write_text(json.dumps(result,indent=2)+'\n')
    result['completed']=True;(a.out/'status.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
