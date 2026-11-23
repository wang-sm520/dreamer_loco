"""Sequential 2x2 standing controls; all variants are diagnostic, never training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .spec import OUTPUT, WORKTREE
from .storage import StorageGuard


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record-root',type=Path,required=True)
    args=parser.parse_args();root=args.record_root
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    state=dict(PID=os.getpid(),phase='standing_height_control_variables',formal_training=False,
        completed=False,checkpoint=None,cohorts=[])
    def save():
        guard.check(100000)
        for name in ('stance_audit_status.json','progress.json','attention.json'):
            (root/name).write_text(json.dumps(state,indent=2)+'\n')
    for w2,ff in [(False,True),(False,False),(True,False),(True,True)]:
        name=f'stance_resume90_w2_{int(w2)}_ff_{int(ff)}_01'
        cmd=['/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.launch',
            '--name',name,'--gpu','0','--seed','0','--mode','stand','--sampling',
            str(OUTPUT/'teacher_host_probe_05/sampling.json')]
        if w2:cmd.append('--audit-w2-model')
        if not ff:cmd.append('--audit-no-gravity-compensation')
        with (root/(name+'_launcher.log')).open('x') as log:
            process=subprocess.Popen(cmd,cwd=WORKTREE/'legmm_skill',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(current_supervisor_pid=process.pid,current_output=str(OUTPUT/name));save()
            while process.poll() is None:save();time.sleep(5)
        child=json.loads((OUTPUT/name/'status.json').read_text())
        if process.returncode or not child['completed']:
            state.update(reason='physical_cohort_failed',child_status=child);save();raise RuntimeError(name)
        result=json.loads((OUTPUT/name/'result.json').read_text())
        state['cohorts'].append(dict(w2_model=w2,gravity_ff=ff,path=str(OUTPUT/name/'result.json'),
            initial_heights=[p[2] for p in result['initial_base']],final_heights=[p[2] for p in result['final_base']],
            safety_stop=result['safety_stop'],contacts=result['forbidden_contact_peaks']))
    state.update(completed=True,current_supervisor_pid=None,current_output=None);save()


if __name__=='__main__':main()
