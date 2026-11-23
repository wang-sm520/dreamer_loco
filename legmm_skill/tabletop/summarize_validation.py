"""Add source/physics qualification and rejection/collision strata to a fixed run."""
import argparse,json,hashlib,math
from collections import Counter
from pathlib import Path
from .physics_config import verify_task_physics_audit
from .spec import PHYSICS_DT,CONTROL_DT,COMMAND_DELAY


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--baseline',type=Path);a=p.parse_args()
    summary=json.loads((a.run/'summary.json').read_text());cache={};details=[];batches=[]
    frozen={Path(x['path']).name:x['sha256'] for x in json.loads(a.freeze.read_text())}
    for path in sorted({row['result_path'] for row in summary['episodes']}):
        parent=Path(path).parent;result=json.loads(Path(path).read_text());audit=json.loads((parent/'physics_audit.json').read_text())
        verify_task_physics_audit(audit)
        inv={Path(x['path']).name:x['sha256'] for x in json.loads((parent/'source_inventory.json').read_text())}
        # Additional independent learning files may exist; every frozen source
        # used by this experiment must retain exactly its original bytes.
        if any(inv.get(k)!=v for k,v in frozen.items()):raise RuntimeError('Frozen source changed during validation')
        gates={}
        for line in (parent/'closed_loop_labels.jsonl').open():
            row=json.loads(line)
            if row['frame']!=0:break
            why=row['metadata'].get('reason') or ''
            if why.startswith('initial_gate:'):gates[row['seed']]=why
        for i,e in enumerate(result['episodes']):
            details.append(dict(seed=e['seed'],success=e['success'],reason=e['reason'],initial_gate_rejection=gates.get(e['seed']),
                forbidden_contacts=e['contacts'],forbidden_contact_peaks=result['forbidden_contact_peaks'][i],
                shape=e['shape'],height=result['states'][i]['table_size'][2],held_steps=e['held_steps'],peak_lift=e['peak_lift'],
                initial_geometry=result['live_geometry'][i],result_path=path))
        batches.append(dict(result_path=path,source_frozen=True,physics_qualified=True,
            sampling_source=json.loads((parent/'sampling_source.json').read_text()),
            physics_audit_sha256=hashlib.sha256((parent/'physics_audit.json').read_bytes()).hexdigest(),
            physical_steps=result['physical_steps'],safety_stop=result['safety_stop'],
            status=json.loads((parent/'status.json').read_text())))
    k=sum(x['success'] for x in details);assert len(details)==256
    report=dict(success=k,total=256,rate=k/256,required_success=231,passed=k>=231,
        hand_calculation=f'{k}/256={100*k/256:.6f}%; ceil(0.90*256)=231',
        physics=dict(global_min_max_position=[4,4],global_min_max_velocity=[0,0],all_applicable_actors=[4,0],
            free_base=True,self_collision=True,physics_dt=PHYSICS_DT,control_dt=CONTROL_DT,command_delay_physics_steps=COMMAND_DELAY),
        counts=dict(initial_gate_rejections=sum(bool(x['initial_gate_rejection']) for x in details),
            forbidden_contact_episodes=sum(bool(x['forbidden_contacts']) for x in details),
            failure_reasons=dict(Counter(x['reason'] for x in details if not x['success']))),
        wilson95=summary['wilson95'],strata=summary['strata'],batches=batches,episodes=details,
        dagger_started=False,ppo_started=False)
    if a.baseline:
        old=json.loads(a.baseline.read_text());old_by_seed={x['seed']:x for x in old['episodes']}
        if sorted(old_by_seed)!=sorted(x['seed'] for x in details):raise ValueError('Paired validation seed mismatch')
        changes=[dict(seed=x['seed'],before=old_by_seed[x['seed']]['reason'],after=x['reason'],
            before_success=old_by_seed[x['seed']]['success'],after_success=x['success']) for x in details]
        report['paired_comparison']=dict(baseline=str(a.baseline),before_success=old['success'],after_success=k,
            improved=[x for x in changes if not x['before_success'] and x['after_success']],
            regressed=[x for x in changes if x['before_success'] and not x['after_success']],all_seeds=changes)
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['success','total','rate','passed','counts']}))


if __name__=='__main__':main()
