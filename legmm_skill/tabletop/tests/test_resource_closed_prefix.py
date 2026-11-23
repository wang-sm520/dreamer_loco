
"""Synthetic exact-state recovery after completed cohorts, then another abort."""
import copy,json,shutil
from pathlib import Path
import pytest
import torch
from tabletop.tests.test_resource_abort_recovery import fixture,REASON,RESTORE
from tabletop.tests.test_closed_collection_recovery import add_samples
from tabletop.dagger_recovery import save_latest
from tabletop.resource_abort_recovery import resource_abort_plan
from tabletop.storage import StorageGuard
from tabletop.state_fingerprints import expected_fingerprints,session_fingerprints,expected_next_batch,digest


def prefix_fixture(tmp_path):
    first=tmp_path/'first';first.mkdir();session,prior_saved,_=fixture(first)
    prior=torch.load(first/'latest.pt',weights_only=False);plan=resource_abort_plan(prior,2)
    archive=tmp_path/'prior.pt';shutil.copyfile(first/'latest.pt',archive)
    second=tmp_path/'second';second.mkdir();sample_count=session.collected_samples
    for serial in range(13,17):
        seed=1088+(serial-13)*64;complete=serial<16;steps=1800 if complete else 332;count=94 if complete else 2
        before=session.seen;E=second/f'round_02_episode_{serial:04d}';E.mkdir()
        add_samples(session,seed,serial,2);session.collected_samples+=3;session.seen+=count*64;session.physical_env_decisions=session.seen;session.episode_serial=serial+1
        result=dict(seed=seed,num_envs=64,driver='student',requested_safe_stop=not complete,safety_stop=None,physical_steps=steps,engine_controlled_steps=steps,native_delay_queue_advances=steps,actual_standing_policy_calls=(steps+1)//2,episodes=[dict(seed=i,reason='timeout' if complete else None,success=False) for i in range(seed,seed+64)])
        (E/'result.json').write_text(json.dumps(result));(E/'post_stage_memory_release.json').write_text('{}')
        (E/'collection.json').write_text(json.dumps(dict(env_decisions=session.seen,physical_env_decisions=session.seen,collected_samples=session.collected_samples,round=2,interrupted=not complete,composition=[dict(count=session.collected_samples)])))
        rows=[dict(frame=i*16,actual_envs=64,teacher_drives=False,env_decisions=before+(i+1)*64,physical_env_decisions=before+(i+1)*64,collected_samples=sample_count+min(3,(i+1)*2)) for i in range(count)]
        (E/'learning_decisions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));sample_count=session.collected_samples
    status=dict(prior['status'],PID=99999998,current_episode=str(E),latest_save={},source_recovery=dict(path=str(archive),sha256=prior_saved['sha256'],env_decisions=67776,optimizer_updates=4,cursor=dict(prior['cursor'],resource_abort_plan=plan)))
    saved=save_latest(session,status,2,1344,17,first,StorageGuard(tmp_path,tmp_path/'code'),complete_round_boundary=False)
    (second/'status.json').write_text(json.dumps(dict(PID=99999998,num_envs=64,completed=False,exit_code=0,reason=REASON,safe_stop_requested=dict(available_ram_bytes=11*1024**3),minimum_available_ram_bytes=9*1024**3)))
    (second/'training_status.json').write_text(json.dumps(dict(PID=99999998,latest_save=saved)))
    return session,torch.load(first/'latest.pt',weights_only=False),saved,first,second


def test_complete_cohorts_are_counted_once_and_restore_exact_state_rng_batch(tmp_path):
    original,payload,saved,rolling,second=prefix_fixture(tmp_path)
    plan=resource_abort_plan(payload,2)
    assert plan['predecessor_consumed_decisions']==67776 and plan['closed_cohort_consumed_decisions']==18048
    assert [x['episode'] for x in plan['closed_cohorts_between_aborts']]==[13,14,15]
    assert plan['consumed_prefix_decisions']==128 and plan['remaining_collection_decisions']==10048
    assert plan['next_seed']==1344 and plan['next_episode']==17
    assert len([p for p in plan['evidence_sha256'] if '/second/round_02_episode_0013/' in p])==4
    other=tmp_path/'other';other.mkdir();restored,_,_=fixture(other,91)
    _,cursor=RESTORE(restored,rolling/'latest.pt',allow_resource_abort=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    assert restored.seen==85952 and restored.student.updates==4 and cursor['next_seed']==1344
    assert session_fingerprints(restored)==expected_fingerprints(payload)
    assert digest(restored.reservoir.sample(64))==expected_next_batch(payload)


@pytest.mark.parametrize('change',['missing','interrupted','unresolved','safety','seed','driver','round','engine','middle_decision','physical_decision','samples','composition','current_start','current_seed','current_samples'])
def test_prefix_corruption_is_rejected_before_restoring(tmp_path,change):
    _,payload,_,_,second=prefix_fixture(tmp_path);E=second/'round_02_episode_0014';file=E/'result.json'
    if change=='missing':(E/'post_stage_memory_release.json').unlink()
    elif change in ['interrupted','unresolved','safety','seed','driver','engine']:
        x=json.loads(file.read_text())
        if change=='interrupted':x['requested_safe_stop']=True
        elif change=='unresolved':x['episodes'][0]['reason']=None
        elif change=='safety':x['safety_stop']='standing_height'
        elif change=='seed':x['seed']-=64
        elif change=='driver':x['driver']='teacher'
        elif change=='engine':x['engine_controlled_steps']-=1
        file.write_text(json.dumps(x))
    elif change in ['round','composition']:
        file=E/'collection.json';x=json.loads(file.read_text())
        if change=='round':x['round']-=1
        else:x['composition'][0]['count']-=1
        file.write_text(json.dumps(x))
    else:
        if change.startswith('current_'):E=second/'round_02_episode_0016'
        if change=='current_seed':
            file=E/'result.json';x=json.loads(file.read_text());x['seed']=1088;x['episodes']=[dict(seed=i,reason=None,success=False) for i in range(1088,1152)];file.write_text(json.dumps(x))
        else:
            file=E/'learning_decisions.jsonl';rows=[json.loads(s) for s in file.read_text().splitlines()]
            if change=='middle_decision':rows[40]['env_decisions']+=64
            elif change=='physical_decision':rows[40]['physical_env_decisions']-=64
            elif change in ['samples','current_samples']:rows[0]['collected_samples']-=100
            elif change=='current_start':rows[0]['env_decisions']-=64
            file.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError):resource_abort_plan(payload,2)
