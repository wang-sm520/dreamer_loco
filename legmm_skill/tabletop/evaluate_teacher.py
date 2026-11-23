"""Detached sequential fixed-seed evaluation; counts all candidates and failures."""
import argparse,json,os,subprocess,time,math
from collections import Counter,defaultdict
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard
from .gates import fixed_validation_gate
from .physics_config import verify_task_physics_audit


def interval(k,n):
    if not n:return None
    z=1.959963984540054;den=1+z*z/n
    center=(k/n+z*z/(2*n))/den
    half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def main():
    p=argparse.ArgumentParser();p.add_argument('--sampling-dir',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,choices=[0,2],default=0)
    p.add_argument('--residual-velocity',action='store_true');p.add_argument('--trace-teacher',action='store_true')
    p.add_argument('--closed-loop-teacher',action='store_true');a=p.parse_args()
    if Path(a.name).name!=a.name or not a.sampling_dir.resolve().is_relative_to(OUTPUT):raise ValueError('Invalid task path')
    out=OUTPUT/a.name;out.mkdir(exist_ok=False);guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');guard.check()
    state=dict(PID=os.getpid(),completed=False,phase='fixed_teacher_validation',formal_training=False,
        current_supervisor_pid=None,current_isaac_pid=None,completed_episodes=0,checkpoint=None,
        CUDA_VISIBLE_DEVICES=str(a.gpu),seed_range=[10000,10255],residual_velocity=a.residual_velocity,
        teacher_kind='measured_state_feedback' if a.closed_loop_teacher else 'open_loop_checked_path')
    def save():
        (out/'status.json').write_text(json.dumps(state,indent=2))
    episodes=[];groups=defaultdict(lambda:[0,0]);started=time.monotonic();save()
    try:
        for seed in range(10000,10256,32):
            sample=a.sampling_dir/f'sampling_{seed}.json'
            while not sample.exists():
                guard.check();prep=a.sampling_dir/'status.json'
                if prep.exists() and json.loads(prep.read_text()).get('completed'):raise RuntimeError('Completed sampling missing cohort')
                time.sleep(2)
            name=f'{a.name}_seed_{seed}'
            command=['/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.launch',
                '--name',name,'--gpu',str(a.gpu),'--seed',str(seed),'--mode','teacher','--sampling',str(sample)]
            if a.trace_teacher:command.append('--trace-teacher')
            if a.closed_loop_teacher:command.append('--closed-loop-teacher')
            if a.residual_velocity:command.append('--residual-velocity')
            with (out/f'launcher_{seed}.log').open('x') as log:
                job=subprocess.Popen(command,cwd=WORKTREE/'legmm_skill',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['current_supervisor_pid']=job.pid;save()
                while job.poll() is None:
                    child_status=OUTPUT/name/'status.json'
                    if child_status.exists():state['current_isaac_pid']=json.loads(child_status.read_text()).get('PID')
                    save();time.sleep(2)
            child=json.loads((OUTPUT/name/'status.json').read_text())
            if job.returncode or not child['completed']:raise RuntimeError(f'Physical cohort failed: {name}')
            result=json.loads((OUTPUT/name/'result.json').read_text())
            verify_task_physics_audit(json.loads((OUTPUT/name/'physics_audit.json').read_text()))
            if result.get('diagnostic_only') or result['mode']!='teacher':raise RuntimeError('Diagnostic cannot enter validation')
            if result.get('safety_stop'):raise RuntimeError('Base stability safety stop; retain cohort evidence')
            for s,e,g in zip(result['states'],result['episodes'],result['gate_reports']):
                row=dict(e,state=s,gate=g,result_path=str(OUTPUT/name/'result.json'));episodes.append(row)
                keys=[s['shape'],'height:'+('low' if s['table_size'][2]<.55 else 'mid' if s['table_size'][2]<.60 else 'high'),
                      'distance:'+('near' if g['contract']['dist']<.6 else 'mid' if g['contract']['dist']<.7 else 'far')]
                for key in keys:groups[key][0]+=int(e['success']);groups[key][1]+=1
            state['completed_episodes']=len(episodes);state['success_count']=sum(e['success'] for e in episodes);save()
            print(json.dumps(state),flush=True)
        if sorted(e['seed'] for e in episodes)!=list(range(10000,10256)):raise RuntimeError('Validation seeds mismatch')
        k=sum(e['success'] for e in episodes);n=len(episodes)
        decision=fixed_validation_gate(episodes)
        summary=dict(episodes=episodes,success_count=k,num_episodes=n,success_rate=k/n,wilson95=interval(k,n),
            teacher_kind=state['teacher_kind'],teacher_threshold=decision['threshold'],teacher_gate_passed=decision['passed'],failures=dict(Counter(e['reason'] for e in episodes)),
            strata={key:dict(success=k,total=n,wilson95=interval(k,n)) for key,(k,n) in groups.items()},elapsed_s=time.monotonic()-started)
        guard.check(2000000);(out/'summary.json').write_text(json.dumps(summary,indent=2))
        state.update(completed=True,teacher_gate_passed=summary['teacher_gate_passed'],current_supervisor_pid=None,current_isaac_pid=None);save()
    except BaseException as error:
        state.update(reason=repr(error),completed=False);save();raise


if __name__=='__main__':main()
