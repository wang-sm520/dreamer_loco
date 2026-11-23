"""Read-only actual interrupted-checkpoint review; never resumes a scene/model."""
import hashlib,json,resource,time,datetime
from pathlib import Path

def main():
    r=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_resource_abort_continue_20261115');o=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_1_random_reachable_20261114/dagger_true4_visual64_retry1_resume_02');source=o/'latest.pt'
    available=int(next(t for t in Path('/proc/meminfo').read_text().splitlines() if t.startswith('MemAvailable:')).split()[1])*1024
    if available<20*1024**3:raise MemoryError('Bounded CPU source audit needs20GiB available before3.3GB load')
    import torch
    torch.set_num_threads(2)
    from .closed_collection_recovery import closed_collection_plan
    from .dagger_recovery import partial_update_plan
    from .state_fingerprints import expected_fingerprints,expected_next_batch
    h=hashlib.sha256();started=time.monotonic()
    with source.open('rb') as file:
        for part in iter(lambda:file.read(8*1024**2),b''):h.update(part)
        assert h.hexdigest()=='d0927058632044f05b65d1e19dc9b5f356062c27cfcc28d864aa8a4c483cca2e'
        file.seek(0);payload=torch.load(file,map_location='cpu',weights_only=False)
    rejects={}
    for name,guard in [('closed_cohort',closed_collection_plan),('partial_update',partial_update_plan)]:
        try:guard(payload)
        except ValueError as error:rejects[name]=str(error)
        else:raise AssertionError('Unsupported resource-interrupted stage was accepted by '+name)
    from .resource_abort_recovery import resource_abort_plan
    plan=resource_abort_plan(payload)
    c=payload['collection'];b=payload['reservoir'];status=payload['status']
    assert status['formal_training'] and not status['diagnostic_only'] and status['num_envs']==64
    assert c['seen']==67776 and payload['student']['updates']==4000 and c['collected_samples']==b['seen']==59940
    assert len(b['items'])==b['capacity']==10000 and sum(b['composition'].values())==b['seen']
    assert payload['cursor']['next_round']==2 and payload['cursor']['next_seed']==1088 and payload['cursor']['episode']==13
    record=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_path=str(source),source_sha256=h.hexdigest(),bytes=source.stat().st_size,resource_abort_plan=plan,format=payload['format'],complete_round_boundary=payload['complete_round_boundary'],formal=True,num_envs=64,model_updates=payload['student']['updates'],cursor=payload['cursor'],collection=c,buffer_frames=len(b['items']),buffer_seen=b['seen'],composition=[dict(key=list(k),count=v) for k,v in b['composition'].items()],source_fingerprints=expected_fingerprints(payload),expected_next_batch64_sha256=expected_next_batch(payload),existing_guards_reject=rejects,previous_full_source_record=status.get('latest_save'),current_round_update_count=status['current_round_update_count'],current_round_loss_sum=status['current_round_loss_sum'],global_rng_keys=list(payload['rng']),CUDA_rng_devices=len(payload['rng']['cuda']),CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_s=time.monotonic()-started,source_unmodified=True,GPU_restore_verified=False,physical_episode_resumed=False,formal_updates_added=0,formal_interactions_added=0)
    (r/'v11_actual_interrupted_source_CPU_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('composition','previous_full_source_record')}))
if __name__=='__main__':main()
