"""Bounded actual-source audit; no training, simulation or CUDA allocation."""
import argparse,copy,gc,gzip,json,os,resource,shutil,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--record-root',type=Path,required=True);a=p.parse_args()
    R=a.record_root
    from .spec import OUTPUT,WORKTREE
    from .storage import StorageGuard
    from .validation_recovery import sha,validation_ledger,recovery_plan
    import torch
    torch.set_num_threads(2)
    from .state_fingerprints import expected_fingerprints,expected_next_batch,session_fingerprints,digest,capture_rng
    from .dagger_recovery import restore_latest
    from .student import Student
    from .data import Reservoir
    from types import SimpleNamespace
    started=time.monotonic();available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    if available<20*1024**3:raise MemoryError('Existing CPU source audit margin requires20GiB')
    attempt=OUTPUT/'dagger_true4_visual64_retry1_resume_04';source=OUTPUT/'dagger_true4_visual64_retry1_resume_03/latest.pt'
    expected='439055bd61695cdbb60ad9e84a56959cb81cca468b1359b9e06171b8d14c91a5'
    assert sha(source)==expected
    with gzip.open(source,'rb') as f:payload=torch.load(f,map_location='cpu',weights_only=False)
    original=expected_fingerprints(payload);next_batch=expected_next_batch(payload)
    old=OUTPUT/'full_boundary_transport_20261116_033015/source.pt.gz'
    with gzip.open(old,'rb') as f:previous=torch.load(f,map_location='cpu',weights_only=False)
    old_fingerprints=expected_fingerprints(previous)
    unchanged={k:original[k]==old_fingerprints[k] for k in ['model','Adam','normalization','reservoir','collection']}
    assert all(unchanged.values()),unchanged
    assert payload['visual']==previous['visual'] and payload['student']['model_config']==previous['student']['model_config']
    del previous;gc.collect()
    model_path=Path(json.loads((attempt/'training_status.json').read_text())['checkpoint'])
    model=torch.load(model_path,map_location='cpu',weights_only=False)
    assert digest(model['model'])==original['model'] and digest(model['optimizer'])==original['Adam'] and model['updates']==6000
    del model;gc.collect()
    qualification=OUTPUT/'teacher_true4_pregrasp_fixed256_retry1_01/summary.json'
    ledger=validation_ledger(attempt/'validation_round_03',qualification)
    evidence=dict(ledger['evidence_sha256'])
    for path in [attempt/'status.json',attempt/'training_status.json',attempt/'source_inventory.json',model_path,old]:evidence[str(path)]=sha(path)
    names=['student.py','learning.py','learning_episode.py','visual.py','geometry.py','closed_loop.py','control.py','spec.py','outcome.py','collision.py','isaac_scene.py','physics_config.py','assets.py','physics_audit.py','lifecycle_cleanup.py','learning_geometry.py','audit_all_rigid.py']
    inventory={Path(x['path']).name:x['sha256'] for x in json.loads((attempt/'source_inventory.json').read_text())}
    code={name:sha(WORKTREE/'legmm_skill/tabletop'/name) for name in names}
    assert all(code[name]==inventory[name] for name in names)
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    archive_root=OUTPUT/'validation_recovery_source_20261116_103215';archive=archive_root/'source.pt.gz'
    guard.check(source.stat().st_size+4096)
    archive_root.mkdir(exist_ok=False)
    with source.open('rb') as src,archive.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())
    assert sha(archive)==sha(source)==expected
    manifest=dict(format='w8_validation_resource_resume_v1',source=str(archive),source_sha256=expected,attempt_root=str(attempt),validation_root=str(attempt/'validation_round_03'),qualification=str(qualification),ledger=ledger,evidence_sha256=evidence,unchanged_code_sha256=code,source_fingerprints=original)
    manifest_path=R/'validation_recovery_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    plan=recovery_plan(payload,manifest_path,expected)
    def fixture():
        return SimpleNamespace(student=Student('cpu',256),reservoir=Reservoir(payload['reservoir']['capacity'],payload['reservoir']['seed']),visual=SimpleNamespace(manifest=lambda:payload['visual']))
    session=fixture()
    try:restore_latest(session,archive,expected_sha256=expected,expected_formal=True)
    except ValueError as error:assert 'Partial recovery' in str(error)
    else:raise AssertionError('Default must refuse validation partial state')
    state,cursor=restore_latest(session,archive,expected_sha256=expected,expected_formal=True,validation_manifest=manifest_path)
    actual=session_fingerprints(session)
    checks={k:actual[k]==original[k] for k in unchanged};assert all(checks.values()),checks
    expected_cpu_rng=dict(payload['rng'],cuda=[]);assert actual['global_rng']==digest(expected_cpu_rng)
    assert digest(session.reservoir.sample(64))==next_batch
    assert cursor['next_round']==3 and cursor['validation_pending']==3 and cursor['next_seed']==1408 and cursor['episode']==18 and cursor['partial_phase'] is None
    assert session.seen==96000 and session.student.updates==6000 and state['total_env_decision_budget']==192000
    # Real source negative controls; rejection must happen before state restoration.
    rejected=[]
    mutations={'wrong_stage':lambda x:x['cursor'].update(partial_phase='dagger_update'),'extra_consumption':lambda x:x['collection'].update(seen=96064),'wrong_updates':lambda x:x['student'].update(updates=6001),'missing_validation':lambda x:x['status'].update(validation_pending=None),'missing_buffer':lambda x:x['reservoir'].update(capacity=10001)}
    for name,mutate in mutations.items():
        x=dict(payload,collection=dict(payload['collection']),cursor=dict(payload['cursor']),student=dict(payload['student']),status=dict(payload['status']),reservoir=dict(payload['reservoir']))
        mutate(x)
        try:recovery_plan(x,manifest_path,expected)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Accepted invalid source: '+name)
    report=dict(completed=True,phase='actual_partial_validation_CPU_verified',source=str(archive),source_sha256=expected,source_bytes=archive.stat().st_size,source_fingerprints=original,expected_next_batch64_sha256=next_batch,old_full_source=str(old),unchanged_learner_vs_previous=unchanged,global_rng_changed_since_full_source=old_fingerprints['global_rng']!=original['global_rng'],current_global_rng_preserved=True,current_cuda_rng_preserved_in_source=True,CPU_restore_checks=checks,CPU_global_rng_exact=True,next_sample64_exact=True,negative_controls_rejected=rejected,cursor=cursor,plan=plan,completed_episodes=ledger['completed_episodes'],interrupted_episodes=32,interrupted_evaluation_decisions=ledger['interrupted'][0]['env_decisions'],remaining_fixed_seeds=ledger['interrupted'][0]['seeds']+ledger['missing_seeds'],formal_env_decisions_added=0,optimizer_updates_added=0,GPU_restore_verified=False,CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_s=time.monotonic()-started,storage_after=guard.check())
    (R/'source_cpu_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (archive_root/'status.json').write_text(json.dumps(dict(completed=True,phase='immutable_partial_validation_source',seed=256,steps=96000,optimizer_updates=6000,checkpoint=str(archive),CUDA_VISIBLE_DEVICES='',source_sha256=expected),indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['completed','CPU_restore_checks','negative_controls_rejected','completed_episodes','interrupted_episodes','interrupted_evaluation_decisions','CPU_peak_RSS_bytes','elapsed_s','storage_after']}),flush=True)
if __name__=='__main__':main()
