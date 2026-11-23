"""Two final one-variable controls matching historical W2 reset conventions."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--record-root',type=Path,required=True)
    p.add_argument('--material-only',action='store_true');a=p.parse_args();r=a.record_root
    status_name='w2_material_bridge_status.json' if a.material_only else 'w2_reset_bridge_status.json'
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    while True:
        guard.check();path=r/('closed_verification_status_08.json' if a.material_only else 'closed_verification_status.json')
        if path.exists() and json.loads(path.read_text())['completed']:break
        time.sleep(5)
    results=[]
    for noise in ([True] if a.material_only else [False,True]):
        name='w2_material_bridge_resume90_01' if a.material_only else f'w2_reset_bridge_resume90_selfoff_noise_{int(noise)}_01'
        cmd=['/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.launch','--name',name,
            '--mode','stand','--gpu','0','--seed','0','--sampling',str(OUTPUT/'teacher_host_probe_05/sampling.json'),
            '--audit-w2-model','--audit-no-gravity-compensation','--audit-yaw-zero','--audit-payload','.25',
            '--audit-w2-self-collision-off']
        if noise:cmd.append('--audit-reset-noise')
        if a.material_only:cmd.append('--audit-w2-robot-material')
        with (r/(name+'_launcher.log')).open('x') as f:
            child=subprocess.Popen(cmd,cwd=WORKTREE/'legmm_skill',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                s=dict(PID=os.getpid(),phase='w2_historical_reset_physical_controls',formal_training=False,
                    completed=False,current_supervisor_pid=child.pid,output=str(OUTPUT/name),cohorts=results)
                for fn in [status_name,'progress.json','attention.json']:(r/fn).write_text(json.dumps(s,indent=2)+'\n')
                time.sleep(5)
        s=json.loads((OUTPUT/name/'status.json').read_text())
        if child.returncode or not s['completed']:raise RuntimeError(name)
        result=json.loads((OUTPUT/name/'result.json').read_text())
        if result['safety_stop']:raise RuntimeError('Base safety stop')
        results.append(dict(reset_noise=noise,result_path=str(OUTPUT/name/'result.json')))
    (r/status_name).write_text(json.dumps(dict(completed=True,formal_training=False,cohorts=results),indent=2))


if __name__=='__main__':main()
