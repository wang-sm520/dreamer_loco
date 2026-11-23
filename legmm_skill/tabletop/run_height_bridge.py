"""Bridge W2's yaw-zero/250g payload convention with the random-stance probe."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record-root',type=Path,required=True);a=parser.parse_args();r=a.record_root
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    while True:
        guard.check()
        p=r/'followup_status.json'
        if p.exists() and json.loads(p.read_text())['completed']:break
        time.sleep(5)
    results=[]
    for yaw_zero,payload in [(True,None),(False,.25),(True,.25)]:
        name=f'height_bridge_resume90_yaw0_{int(yaw_zero)}_payload_{"025" if payload else "00001"}_01'
        cmd=['/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.launch','--name',name,
            '--mode','stand','--gpu','0','--seed','0','--sampling',str(OUTPUT/'teacher_host_probe_05/sampling.json'),
            '--audit-w2-model','--audit-no-gravity-compensation']
        if yaw_zero:cmd.append('--audit-yaw-zero')
        if payload is not None:cmd+=['--audit-payload',str(payload)]
        with (r/(name+'_launcher.log')).open('x') as f:
            child=subprocess.Popen(cmd,cwd=WORKTREE/'legmm_skill',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                status=dict(PID=os.getpid(),phase='height_bridge_physical_controls',formal_training=False,
                    completed=False,current_supervisor_pid=child.pid,output=str(OUTPUT/name),cohorts=results)
                for fn in ['height_bridge_status.json','progress.json','attention.json']:(r/fn).write_text(json.dumps(status,indent=2)+'\n')
                time.sleep(5)
        s=json.loads((OUTPUT/name/'status.json').read_text())
        if child.returncode or not s['completed']:raise RuntimeError(name)
        results.append(dict(yaw_zero=yaw_zero,payload=payload,result_path=str(OUTPUT/name/'result.json')))
    (r/'height_bridge_status.json').write_text(json.dumps(dict(completed=True,formal_training=False,cohorts=results),indent=2))


if __name__=='__main__':main()
