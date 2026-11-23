"""Witness closed student cohorts after a full round, before its first RAM abort."""
import hashlib,json
from pathlib import Path
from .spec import OUTPUT
from .resource_closed_prefix import closed_prefix_plan,sample_ledger


def first_abort_prefix_plan(payload,previous,updates_per_round=2000):
    c=payload['collection'];s=payload['status'];n=s['num_envs'];r=c['round']
    current=Path(s['current_episode']);root=current.parent;boundary=r*32000
    # Only the observed same-root, full-save -> closed cohorts -> RAM stop case.
    if Path(previous['path']).parent!=root or previous.get('validation_pending') is not None:
        raise ValueError('First-abort full-boundary root/validation differs')
    final=int(current.name.rsplit('_',1)[1])
    candidates=sorted(root.glob(f'round_{r:02d}_episode_*'))
    serials=[int(p.name.rsplit('_',1)[1]) for p in candidates if p.name.rsplit('_',1)[1].isdigit()]
    if not serials or serials!=list(range(serials[0],final+1)) or serials[0]>=final:
        raise ValueError('First-abort closed-prefix episode coverage differs')
    first=serials[0];anchor=root/f'round_{r-1:02d}_episode_{first-1:04d}'
    names=['result.json','collection.json','post_stage_memory_release.json','learning_decisions.jsonl']
    if s.get('formal_training'):names.append('physics_audit.json')
    if anchor.is_symlink() or not anchor.resolve().is_relative_to(OUTPUT):raise ValueError('Owned round-boundary anchor required')
    if any((anchor/k).is_symlink() or not (anchor/k).is_file() for k in names):raise ValueError('Missing round-boundary witness')
    result=json.loads((anchor/'result.json').read_text());record=json.loads((anchor/'collection.json').read_text())
    rows=[json.loads(line) for line in (anchor/'learning_decisions.jsonl').read_text().splitlines()]
    if record.get('interrupted') or result.get('requested_safe_stop') or result.get('safety_stop') is not None:
        raise ValueError('Round-boundary anchor was interrupted or unsafe')
    if record['round']!=r-1 or record['env_decisions']!=boundary or record['physical_env_decisions']!=boundary:
        raise ValueError('Round-boundary counters differ from full save')
    if result['num_envs']!=n or len(result['episodes'])!=n or sorted(x['seed'] for x in result['episodes'])!=list(range(result['seed'],result['seed']+n)):
        raise ValueError('Round-boundary initial-state identities differ')
    teacher=r==1
    if result.get('driver')!=('teacher' if teacher else 'student'):raise ValueError('Round-boundary driver differs')
    steps=result['physical_steps']
    if not 300<steps<=1800 or any(result[k]!=steps for k in ('engine_controlled_steps','native_delay_queue_advances')) or result['actual_standing_policy_calls']!=(steps+1)//2:
        raise ValueError('Round-boundary physical/control witness differs')
    if [x['frame'] for x in rows]!=list(range(0,steps-300,16)) or not rows:
        raise ValueError('Round-boundary frame ledger differs')
    if [x['env_decisions'] for x in rows]!=list(range(boundary-(len(rows)-1)*n,boundary+1,n)) or rows[0]['env_decisions']-n<(r-1)*32000:
        raise ValueError('Round-boundary decision ledger differs')
    if any(x['physical_env_decisions']!=x['env_decisions'] or x['actual_envs']!=n or x['teacher_drives']!=teacher for x in rows):
        raise ValueError('Round-boundary driver/physical ledger differs')
    samples=record['collected_samples']
    if sum(x['count'] for x in record.get('composition',[]))!=samples:raise ValueError('Round-boundary sample composition differs')
    sample_ledger(rows,rows[0]['collected_samples'],samples,n)
    if s.get('formal_training'):
        from .physics_config import verify_task_physics_audit
        verify_task_physics_audit(json.loads((anchor/'physics_audit.json').read_text()))
    seed=result['seed']+n
    if seed+n>10000:seed=256
    plan=closed_prefix_plan(payload,dict(episode=first,next_seed=seed),boundary,samples,updates_per_round)
    plan['evidence_sha256'].update({str(anchor/k):hashlib.sha256((anchor/k).read_bytes()).hexdigest() for k in names})
    plan['full_boundary_anchor']=str(anchor)
    return plan
