"""CPU geometry diagnostic, NEVER an Isaac success evaluation."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

from .assets import build_robot
from .geometry import GeometryPlanner
from .spec import sample_candidate,OUTPUT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base-height',type=float,required=True,help='Explicit diagnostic pose, NOT measured live shoulder height')
    a=p.parse_args()
    a.out.mkdir(exist_ok=False)
    status=dict(config=vars(a)|{'out':str(a.out)},seed=0,steps=0,checkpoint=None,CUDA_VISIBLE_DEVICES='',PID=os.getpid(),completed=False)
    (a.out/'status.json').write_text(json.dumps(status,indent=2))
    planner=GeometryPlanner(build_robot(OUTPUT/'models'))
    counts=Counter();accepted=[];start=time.monotonic()
    with (a.out/'candidates.jsonl').open('x') as f:
        for i in range(a.count):
            s=sample_candidate(i)
            pose=[*s.base_xy,a.base_height,math.cos(s.base_yaw/2),0,0,math.sin(s.base_yaw/2)]
            plan=planner.plan(s,pose)
            counts[plan.reason]+=1
            record={'state':s.json(),'plan':asdict(plan)}
            f.write(json.dumps(record)+'\n');f.flush()
            if plan.accepted: accepted.append(i)
    status.update(completed=True,steps=a.count,elapsed_s=time.monotonic()-start)
    (a.out/'summary.json').write_text(json.dumps(dict(counts=counts,accepted_seeds=accepted,not_execution_success=True),indent=2))
    (a.out/'status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(dict(counts=counts,accepted_seeds=accepted)),flush=True)

if __name__=='__main__': main()
