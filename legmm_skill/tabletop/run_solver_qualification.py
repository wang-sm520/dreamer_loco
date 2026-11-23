"""Causal solver control and task smokes; never another fixed-256 evaluation."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record-root',type=Path,required=True);a=parser.parse_args();r=a.record_root
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');results=[]
    jobs=[('w2_scene_solver_unclamped_resume90_01','stand','teacher_host_probe_05',
        ['--audit-w2-model','--audit-no-gravity-compensation','--audit-yaw-zero','--audit-payload','.25',
         '--audit-w2-self-collision-off','--audit-reset-noise','--audit-w2-robot-material','--audit-scene-solver-unclamped']),
        ('teacher_true_scene_4_0_resume90_01','teacher','teacher_host_probe_05',['--trace-teacher']),
        ('teacher_closed_true_scene_4_0_residual_resume90_01','teacher','residual_resume90_sampling_01',
         ['--trace-teacher','--closed-loop-teacher','--residual-velocity'])]
    for name,mode,sampling,extra in jobs:
        command=['/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.launch','--name',name,
            '--mode',mode,'--gpu','0','--seed','0','--sampling',str(OUTPUT/sampling/'sampling.json')]+extra
        with (r/(name+'_launcher.log')).open('x') as f:
            child=subprocess.Popen(command,cwd=WORKTREE/'legmm_skill',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                guard.check()
                s=dict(PID=os.getpid(),phase=name,formal_training=False,completed=False,
                    current_supervisor_pid=child.pid,output=str(OUTPUT/name),cohorts=results)
                for fn in ['solver_qualification_status.json','progress.json','attention.json']:(r/fn).write_text(json.dumps(s,indent=2)+'\n')
                time.sleep(5)
        status=json.loads((OUTPUT/name/'status.json').read_text())
        if child.returncode or not status['completed']:raise RuntimeError(name)
        d=json.loads((OUTPUT/name/'result.json').read_text());audit=json.loads((OUTPUT/name/'physics_audit.json').read_text())
        if mode=='teacher':
            limits=audit['scene_solver_iteration_limits']
            assert limits['physxScene:maxPositionIterationCount']==4 and limits['physxScene:maxVelocityIterationCount']==0
            assert all(x['rigid_body_solver']['Object']==dict(position=4,velocity=0) for x in audit['environments'])
        if d['safety_stop']:raise RuntimeError('Base safety stop')
        results.append(dict(result_path=str(OUTPUT/name/'result.json'),physics_audit_path=str(OUTPUT/name/'physics_audit.json'),
            success=sum(e['success'] for e in d['episodes']),total=32))
    (r/'solver_qualification_status.json').write_text(json.dumps(dict(completed=True,formal_training=False,cohorts=results),indent=2))


if __name__=='__main__':main()
