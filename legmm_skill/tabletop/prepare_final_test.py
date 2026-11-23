"""Precommit independent geometric states after models are selected, before tests."""
import argparse,datetime,hashlib,json,os,subprocess
from collections import Counter
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .train_ppo import student_qualification,teacher_qualification


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('out','student-status','student-checkpoint','ppo-status','qualification'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();student_qualification(a.student_status,a.student_checkpoint);q=teacher_qualification(a.qualification)
    status=json.loads(a.ppo_status.read_text())
    if not status.get('completed') or status.get('phase')!='ppo_budget_complete':raise ValueError('Select models before independent test access')
    if a.out.exists() or not a.out.resolve().is_relative_to(OUTPUT):raise ValueError('Fresh W8 output subdirectory required')
    template=Path(q['episodes'][0]['result_path'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['nice','-n','10','/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.prepare_sampling',
        '--template',str(template),'--out',str(a.out),'--seed','20000','--count','256'],
        cwd=WORKTREE/'legmm_skill',env=env,check=True)
    files={};attempts=Counter();families=Counter()
    for seed in range(20000,20256,32):
        path=a.out/f'sampling_{seed}.json';files[path.name]=sha(path)
        for row in json.loads(path.read_text()):
            attempts['accepted' if row['accepted'] else row['reason']]+=1
            families[(row['state']['shape'],bool(row['accepted']))]+=1
    manifest=dict(completed=True,geometry_only=True,precommitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seed_range=[20000,20255],files=files,template=str(template),template_sha256=sha(template),
        models=dict(student_sha256=sha(a.student_checkpoint),residual_sha256=sha(status['selected_actor'])),
        candidate_reasons=dict(attempts),family_acceptance=[dict(shape=k[0],accepted=k[1],count=v) for k,v in families.items()],
        execution_outcomes_used_for_sampling=False,residual_velocity=False)
    with (a.out/'fixed_manifest.json').open('x') as file:json.dump(manifest,file,indent=2)


if __name__=='__main__':main()
