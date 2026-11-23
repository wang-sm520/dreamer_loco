"""First RAM abort after complete cohorts: retain ledger, reject missing evidence."""
import copy,json
from types import SimpleNamespace
import pytest
import torch
from tabletop.data import Reservoir
from tabletop.dagger_recovery import save_latest,restore_latest
from tabletop.resource_abort_recovery import resource_abort_plan
from tabletop.state_fingerprints import expected_fingerprints,session_fingerprints,expected_next_batch,digest
from tabletop.storage import StorageGuard
from tabletop.tests.test_closed_collection_recovery import TinyStudent,add_samples
from tabletop.tests.test_resource_abort_recovery import REASON


def fixture(root):
    torch.set_num_threads(2)
    session=SimpleNamespace(student=TinyStudent(7),reservoir=Reservoir(4,7),visual=SimpleNamespace(manifest=lambda:{}),seen=32000,budget=32000,round=0,teacher_drives=True,episode_serial=6,physical_env_decisions=32000,collected_samples=3)
    add_samples(session,576,5,0)
    for _ in range(2):session.student.update(session.reservoir)
    base=dict(PID=99999999,formal_training=False,diagnostic_only=True,num_envs=64,current_round_update_count=2,current_round_loss_sum=.2)
    def witness(r,serial,seed,steps,before,samples_before,complete=False,aborted=False):
        e=root/f'round_{r:02d}_episode_{serial:04d}';e.mkdir()
        count=len(range(0,steps-300,16));seen=before+count*64
        result=dict(seed=seed,num_envs=64,driver='teacher' if r==0 else 'student',requested_safe_stop=aborted,safety_stop=None,physical_steps=steps,engine_controlled_steps=steps,native_delay_queue_advances=steps,actual_standing_policy_calls=(steps+1)//2,episodes=[dict(seed=i,reason='timeout' if complete else None,success=False) for i in range(seed,seed+64)])
        (e/'result.json').write_text(json.dumps(result));(e/'post_stage_memory_release.json').write_text('{}')
        (e/'collection.json').write_text(json.dumps(dict(env_decisions=seen,physical_env_decisions=seen,collected_samples=session.collected_samples,round=r,interrupted=aborted,composition=[dict(count=session.collected_samples)])))
        rows=[dict(frame=i*16,actual_envs=64,teacher_drives=r==0,env_decisions=before+(i+1)*64,physical_env_decisions=before+(i+1)*64,collected_samples=samples_before+min(session.collected_samples-samples_before,(i+1)*3)) for i in range(count)]
        (e/'learning_decisions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        return e,seen
    witness(0,5,576,316,31936,0)
    guard=StorageGuard(root,root/'code')
    previous=save_latest(session,base,1,640,6,root,guard)
    session.round=1;session.budget=64000;session.teacher_drives=False
    for serial in (6,7,8):
        samples=session.collected_samples;before=session.seen;add_samples(session,640+(serial-6)*64,serial,1);session.collected_samples+=3
        e,seen=witness(1,serial,640+(serial-6)*64,332 if serial==8 else 1800,before,samples,complete=serial<8,aborted=serial==8)
        session.seen=session.physical_env_decisions=seen;session.episode_serial=serial+1
    status=dict(base,current_episode=str(e),last_phase='dagger_student_collection',reason=REASON,current_round_update_count=0,current_round_loss_sum=0.,latest_save=previous)
    saved=save_latest(session,status,1,832,9,root,guard,complete_round_boundary=False)
    (root/'status.json').write_text(json.dumps(dict(PID=99999999,num_envs=64,completed=False,exit_code=0,reason=REASON,safe_stop_requested=dict(available_ram_bytes=11*1024**3),minimum_available_ram_bytes=9*1024**3)))
    return session,torch.load(root/'latest.pt',weights_only=False),saved


def test_first_abort_retains_closed_prefix_exact_learner_and_next_batch(tmp_path):
    session,payload,saved=fixture(tmp_path);plan=resource_abort_plan(payload,2)
    assert plan['closed_cohort_consumed_decisions']==12032 and plan['consumed_prefix_decisions']==128
    assert [x['episode'] for x in plan['closed_cohorts_after_full_round']]==[6,7]
    assert plan['remaining_collection_decisions']==19840 and plan['next_episode']==9 and plan['next_seed']==832
    assert not plan['aborted_initial_states_replayed'] and not plan['physical_episode_resumed']
    expected=expected_fingerprints(payload)
    session.student.model.weight.data.add_(1);session.reservoir.rng.seed(999)
    _,cursor=restore_latest(session,tmp_path/'latest.pt',allow_resource_abort=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    assert session_fingerprints(session)==expected and session.seen==44160 and session.student.updates==2
    assert digest(session.reservoir.sample(64))==expected_next_batch(payload)
    assert cursor['resource_abort_plan']==plan


@pytest.mark.parametrize('change',['missing_anchor','anchor_count','anchor_seed','anchor_driver','anchor_samples','anchor_stop','anchor_steps','missing_cohort','cohort_seed','cohort_unresolved','cohort_ledger','current_samples','source_sha','pending_validation'])
def test_first_abort_rejects_broken_boundary_prefix_and_provenance(tmp_path,change):
    _,p,_=fixture(tmp_path);anchor=tmp_path/'round_00_episode_0005';cohort=tmp_path/'round_01_episode_0006'
    if change=='missing_anchor':(anchor/'collection.json').unlink()
    elif change=='missing_cohort':(cohort/'post_stage_memory_release.json').unlink()
    elif change=='source_sha':p['status']['latest_save']['sha256']='0'*64
    elif change=='pending_validation':p['status']['latest_save']['validation_pending']=1
    else:
        e=anchor if change.startswith('anchor_') else cohort
        file=e/('collection.json' if change in ('anchor_count','anchor_samples') else 'result.json')
        if change in ('cohort_ledger','current_samples'):
            if change=='current_samples':e=tmp_path/'round_01_episode_0008'
            file=e/'learning_decisions.jsonl';rows=[json.loads(x) for x in file.read_text().splitlines()]
            rows[0]['collected_samples' if change=='current_samples' else 'env_decisions']-=128
            file.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        else:
            x=json.loads(file.read_text())
            if change=='anchor_count':x['env_decisions']-=64
            elif change in ('anchor_seed','cohort_seed'):x['seed']-=64
            elif change=='anchor_driver':x['driver']='student'
            elif change=='anchor_samples':x['collected_samples']-=1
            elif change=='anchor_stop':x['requested_safe_stop']=True
            elif change=='anchor_steps':x['engine_controlled_steps']-=1
            elif change=='cohort_unresolved':x['episodes'][0]['reason']=None
            file.write_text(json.dumps(x))
    with pytest.raises(ValueError):resource_abort_plan(p,2)
