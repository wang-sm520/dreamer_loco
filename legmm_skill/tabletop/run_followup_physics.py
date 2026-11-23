"""Continue the authorised reset-residual and closed-loop teacher probes."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .spec import OUTPUT,WORKTREE,sample_candidate
from .storage import StorageGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--record-root',type=Path,required=True);a=p.parse_args();root=a.record_root
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    while True:
        guard.check()
        status=json.loads((root/'stance_audit_status.json').read_text())
        if status['completed']:break
        if status.get('reason'):raise RuntimeError('Standing audit blocked: '+str(status['reason']))
        time.sleep(5)
    sampling=OUTPUT/'residual_resume90_sampling_01';sampling.mkdir(exist_ok=False)
    original=json.loads((OUTPUT/'teacher_host_probe_05/sampling.json').read_text())
    for row in original:
        s=row['state'];row['state']=sample_candidate(s['seed'],s['attempt'],True).json()
    path=sampling/'sampling.json';path.write_text(json.dumps(original,indent=1))
    (sampling/'status.json').write_text(json.dumps(dict(completed=True,phase='deterministic_reset_residual_sampling',PID=os.getpid())))
    for name,extra,sample in [
        ('teacher_residual_resume90_01',['--residual-velocity'],path),
        ('teacher_closed_loop_resume90_01',['--closed-loop-teacher'],OUTPUT/'teacher_host_probe_05/sampling.json')]:
        command=['/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.launch',
            '--name',name,'--mode','teacher','--gpu','0','--seed','0','--trace-teacher','--sampling',str(sample)]+extra
        with (root/(name+'_launcher.log')).open('x') as log:
            proc=subprocess.Popen(command,cwd=WORKTREE/'legmm_skill',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                state=dict(PID=os.getpid(),current_supervisor_pid=proc.pid,phase=name,formal_training=False,
                    checkpoint=None,completed=False,output=str(OUTPUT/name))
                for f in ['progress.json','attention.json','followup_status.json']:(root/f).write_text(json.dumps(state,indent=2)+'\n')
                time.sleep(5)
        status=json.loads((OUTPUT/name/'status.json').read_text())
        if proc.returncode or not status['completed']:raise RuntimeError('Physical probe failed: '+name)
    (root/'followup_status.json').write_text(json.dumps(dict(PID=os.getpid(),completed=True,phase='followup_physics_complete',formal_training=False)))


if __name__=='__main__':main()
