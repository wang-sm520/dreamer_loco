"""CPU-only deterministic train geometry producer; never sees rollout outcomes."""
import argparse,datetime,hashlib,json,math,os,time
from pathlib import Path
from .spec import OUTPUT
from .geometry import GeometryPlanner,sample_reachable
from .assets import build_robot
from .storage import StorageGuard


def nominal(state):return [*state.base_xy,.3763,math.cos(state.base_yaw/2),0,0,math.sin(state.base_yaw/2)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--publish',type=Path,required=True)
    p.add_argument('--start',type=int,default=1024);p.add_argument('--count',type=int,default=2048);p.add_argument('--batch',type=int,default=64);a=p.parse_args()
    if not (0<=a.start<a.start+a.count<=10000) or a.count%a.batch:raise ValueError('Whole train-only geometric cohorts required')
    if not a.out.resolve().is_relative_to(OUTPUT) or not a.publish.resolve().is_relative_to(OUTPUT):raise ValueError('W8 task paths only')
    a.out.mkdir(exist_ok=False);guard=StorageGuard(OUTPUT,Path(__file__).parent);guard.check()
    planner=GeometryPlanner(build_robot(OUTPUT/'models'),cache_models=False)
    # Verify the no-cache producer against eight actual earlier sampling traces.
    reference=json.loads((a.publish/'sampling_256.json').read_text());actual=[]
    for seed in range(256,264):
        local=[]
        state,plan=sample_reachable(seed,planner,nominal,lambda s,p:local.append(dict(state=s.json(),accepted=p.accepted,reason=p.reason)),True)
        expected=[row for row in reference if row['state']['seed']==seed]
        if json.loads(json.dumps(local))!=expected:raise RuntimeError('CPU prefetch differs from original sampling at '+str(seed))
        actual.append(dict(seed=seed,attempt=state.attempt,matched=True))
    status=dict(PID=os.getpid(),phase='CPU_train_geometry_prefetch',completed=False,formal_training=False,Isaac_App_started=False,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),start=a.start,count=a.count,batch=a.batch,
        reference_eight_seed_pair=actual,selection_uses_execution_outcomes=False,records=[])
    def save():(a.out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();rows=[];started=time.monotonic()
    from .lifecycle_cleanup import memory_snapshot,trim_unused
    for seed in range(a.start,a.start+a.count):
        available=memory_snapshot()['available_ram_bytes']
        if available<20*1024**3:
            status.update(phase='stopped_for_host_margin',reason='CPU prefetch yields before formal warm-stop margin',available_ram_bytes=available);save();return
        state,plan=sample_reachable(seed,planner,nominal,lambda s,p:rows.append(dict(state=s.json(),accepted=p.accepted,reason=p.reason)),True)
        if (seed-a.start+1)%a.batch:continue
        first=seed-a.batch+1;target=a.out/f'sampling_{first}.json';pending=a.out/f'sampling_{first}.json.pending'
        data=json.dumps(rows,indent=1).encode();guard.check(len(data))
        with pending.open('xb') as file:file.write(data);file.flush();os.fsync(file.fileno())
        os.replace(pending,target)
        link=a.publish/target.name
        # Add only future input names. Existing records and their manifest are
        # immutable; no output or previous sample is overwritten.
        if link.exists() or link.is_symlink():raise FileExistsError(link)
        link.symlink_to(target)
        status['records'].append(dict(seed_range=[first,seed],path=str(target),published_link=str(link),sha256=hashlib.sha256(data).hexdigest(),candidates=len(rows)))
        rows=[];status.update(elapsed_s=time.monotonic()-started,last_seed=seed,memory=trim_unused());save()
        print(json.dumps(dict(last_seed=seed,elapsed_s=status['elapsed_s'])),flush=True)
    status.update(completed=True,phase='CPU_geometry_prefetch_complete');save()


if __name__=='__main__':main()
