
"""Verify every completed student cohort between two resource interruptions."""
import hashlib,json
from pathlib import Path
from .spec import OUTPUT
from .closed_collection_recovery import closed_collection_plan


def sample_ledger(rows,before,after,n):
    values=[row.get('collected_samples') for row in rows]
    if any(not isinstance(v,int) for v in values):raise ValueError('Missing closed-prefix sample ledger')
    if any(not 0<=value-previous<=n for previous,value in zip([before]+values[:-1],values)) or values[-1]!=after:
        raise ValueError('Closed-prefix sample ledger gap')


def closed_prefix_plan(payload,prior_cursor,seen,samples,updates_per_round=2000):
    c=payload['collection'];s=payload['status'];n=s['num_envs'];r=c['round']
    current=Path(s['current_episode']);root=current.parent
    serial=prior_cursor['episode'];seed=prior_cursor['next_seed']
    final_serial=int(current.name.rsplit('_',1)[1]);start_seen=seen;evidence={};cohorts=[]
    if current.name!=f'round_{r:02d}_episode_{final_serial:04d}' or not serial<=final_serial<=serial+500:
        raise ValueError('Closed-prefix episode cursor differs')
    for index in range(serial,final_serial):
        episode=root/f'round_{r:02d}_episode_{index:04d}'
        required=['result.json','collection.json','post_stage_memory_release.json','learning_decisions.jsonl']
        if s.get('formal_training'):required.append('physics_audit.json')
        if episode.is_symlink() or not episode.resolve().is_relative_to(OUTPUT):raise ValueError('Owned closed-prefix episode required')
        if any((episode/name).is_symlink() or not (episode/name).is_file() for name in required):raise ValueError('Missing closed-prefix witness')
        result=json.loads((episode/'result.json').read_text());record=json.loads((episode/'collection.json').read_text())
        rows=[json.loads(line) for line in (episode/'learning_decisions.jsonl').read_text().splitlines()]
        if result.get('driver')!='student' or result['seed']!=seed:raise ValueError('Closed-prefix student seed differs')
        if any(bool(row['success'])!=(row['reason']=='success') for row in result['episodes']):raise ValueError('Closed-prefix terminal outcome differs')
        next_seed=seed+n
        if next_seed+n>10000:next_seed=256
        virtual=dict(payload,collection=dict(c,seen=record['env_decisions'],physical_env_decisions=record['physical_env_decisions'],collected_samples=record['collected_samples'],episode_serial=index+1),
            status=dict(s,current_episode=str(episode)),cursor=dict(payload['cursor'],next_seed=next_seed,episode=index+1))
        plan=closed_collection_plan(virtual,updates_per_round,n)
        expected=list(range(seen+n,record['env_decisions']+1,n))
        if [row['env_decisions'] for row in rows]!=expected or any(row.get('physical_env_decisions')!=row['env_decisions'] for row in rows):
            raise ValueError('Closed-prefix decision ledger gap')
        sample_ledger(rows,samples,record['collected_samples'],n)
        if sum(row['count'] for row in record.get('composition',[]))!=record['collected_samples']:
            raise ValueError('Closed-prefix sample composition differs')
        evidence.update({str(episode/name):hashlib.sha256((episode/name).read_bytes()).hexdigest() for name in required})
        cohorts.append(dict(episode=index,seed=seed,start_seen=seen,env_decisions=record['env_decisions'],collected_samples=record['collected_samples'],cohort_env_decisions=plan['cohort_env_decisions']))
        seen=record['env_decisions'];samples=record['collected_samples'];seed=next_seed
    return dict(seen=seen,samples=samples,seed=seed,episode=final_serial,cohorts=cohorts,consumed=seen-start_seen,evidence_sha256=evidence)
