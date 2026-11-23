"""Verify corrected native action-chunk feedback on static and residual states."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record-root',type=Path,required=True)
    parser.add_argument('--start-index',type=int,default=4);parser.add_argument('--wait-status',default='height_bridge_status.json')
    a=parser.parse_args();r=a.record_root
    status_name='closed_verification_status.json' if a.start_index==4 else f'closed_verification_status_{a.start_index:02d}.json'
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    while True:
        guard.check()
        path=r/a.wait_status
        if path.exists() and json.loads(path.read_text())['completed']:break
        time.sleep(5)
    results=[]
    for name,residual,sampling in [(f'teacher_closed_loop_resume90_{a.start_index:02d}',False,'teacher_host_probe_05'),
                                   (f'teacher_closed_loop_residual_resume90_{a.start_index+1:02d}',True,'residual_resume90_sampling_01')]:
        cmd=['/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.launch','--name',name,
            '--mode','teacher','--gpu','0','--seed','0','--sampling',str(OUTPUT/sampling/'sampling.json'),
            '--closed-loop-teacher','--trace-teacher']
        if residual:cmd.append('--residual-velocity')
        with (r/(name+'_launcher.log')).open('x') as f:
            child=subprocess.Popen(cmd,cwd=WORKTREE/'legmm_skill',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                status=dict(PID=os.getpid(),phase=name,formal_training=False,completed=False,
                    current_supervisor_pid=child.pid,output=str(OUTPUT/name),cohorts=results)
                for fn in [status_name,'progress.json','attention.json']:(r/fn).write_text(json.dumps(status,indent=2)+'\n')
                time.sleep(5)
        s=json.loads((OUTPUT/name/'status.json').read_text())
        if child.returncode or not s['completed']:raise RuntimeError(name)
        result=json.loads((OUTPUT/name/'result.json').read_text())
        results.append(dict(result_path=str(OUTPUT/name/'result.json'),success=sum(e['success'] for e in result['episodes']),total=32))
        if result['safety_stop']:raise RuntimeError('Base safety stop')
    (r/status_name).write_text(json.dumps(dict(completed=True,formal_training=False,cohorts=results),indent=2))


if __name__=='__main__':main()
