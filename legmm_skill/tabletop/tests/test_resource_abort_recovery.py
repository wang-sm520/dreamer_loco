"""Synthetic CPU learner-state tests; no claim of physical episode restoration."""
import copy,json,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tabletop.data import Reservoir
from tabletop.dagger_recovery import save_latest
from tabletop.resource_abort_recovery import resource_abort_plan
from tabletop.prepare_resource_abort_patch import recovery_source
from tabletop.storage import StorageGuard
from tabletop.tests.test_closed_collection_recovery import TinyStudent,add_samples

from tabletop.dagger_recovery import restore_latest as RESTORE
REASON="SafeStop('Stop signal 15 handled at safe boundary: collection action-block boundary after scene cleanup')"


def fixture(tmp_path,seed=7):
    random.seed(seed);np.random.seed(seed);student=TinyStudent(seed)
    session=SimpleNamespace(student=student,reservoir=Reservoir(4,seed),visual=SimpleNamespace(manifest=lambda:{}),seen=64000,budget=64000,round=1,teacher_drives=False,episode_serial=12,physical_env_decisions=64000,collected_samples=6)
    add_samples(session,256,1,0);add_samples(session,640,7,1)
    for _ in range(4):student.update(session.reservoir)
    base=dict(PID=99999999,formal_training=False,diagnostic_only=True,synthetic_CPU_test=True,num_envs=64,current_round_update_count=2,current_round_loss_sum=.2)
    previous=save_latest(session,base,2,1024,12,tmp_path,StorageGuard(tmp_path,tmp_path/'code'))
    session.round=2;session.budget=96000;session.seen=session.physical_env_decisions=67776;session.episode_serial=13;session.collected_samples+=3;add_samples(session,1024,13,2)
    episode=tmp_path/'round_02_episode_0012';episode.mkdir()
    result=dict(seed=1024,num_envs=64,requested_safe_stop=True,safety_stop=None,physical_steps=1244,engine_controlled_steps=1244,native_delay_queue_advances=1244,actual_standing_policy_calls=622,episodes=[dict(seed=i,reason=None,success=False) for i in range(1024,1088)])
    (episode/'result.json').write_text(json.dumps(result));(episode/'post_stage_memory_release.json').write_text('{}')
    (episode/'collection.json').write_text(json.dumps(dict(env_decisions=67776,physical_env_decisions=67776,collected_samples=9,round=2,interrupted=True)))
    (episode/'learning_decisions.jsonl').write_text(''.join(json.dumps(dict(frame=i*16,actual_envs=64,teacher_drives=False,env_decisions=64000+(i+1)*64))+'\n' for i in range(59)))
    status=dict(base,current_episode=str(episode),last_phase='dagger_student_collection',reason=REASON,current_round_update_count=0,current_round_loss_sum=0.,current_round_update_elapsed_s=0.,latest_save=previous,total_env_decision_budget=192000)
    saved=save_latest(session,status,2,1088,13,tmp_path,StorageGuard(tmp_path,tmp_path/'code'),complete_round_boundary=False)
    (tmp_path/'status.json').write_text(json.dumps(dict(PID=99999999,num_envs=64,completed=False,exit_code=0,reason=REASON,safe_stop_requested=dict(available_ram_bytes=11*1024**3,own_rss_bytes=1024**3),minimum_available_ram_bytes=9*1024**3)))
    return session,saved,episode


def continue_after_aborted_episode(session):
    add_samples(session,1088,14,2);remaining=session.budget-session.seen;assert remaining==28224
    session.seen+=remaining;session.physical_env_decisions+=remaining;session.collected_samples+=3;session.episode_serial+=1
    for _ in range(2):session.student.update(session.reservoir)


def test_explicit_resource_learner_resume_keeps_prefix_rng_adam_and_next_seed(tmp_path):
    torch.set_num_threads(2);original,saved,_=fixture(tmp_path);payload=torch.load(tmp_path/'latest.pt',weights_only=False);plan=resource_abort_plan(payload,2)
    assert plan['consumed_prefix_decisions']==3776 and plan['remaining_collection_decisions']==28224 and plan['remaining_round_updates']==2
    assert plan['next_seed']==1088 and plan['next_episode']==13 and not plan['physical_episode_resumed'] and not plan['aborted_initial_states_replayed']
    continue_after_aborted_episode(original);expected=[p.detach().clone() for p in original.student.model.parameters()];batch=original.reservoir.sample(3);rng=(random.random(),np.random.random(),torch.rand(3));adam=copy.deepcopy(original.student.optimizer.state_dict())
    other=tmp_path/'other';other.mkdir();recovered,_,_=fixture(other,93)
    _,cursor=RESTORE(recovered,tmp_path/'latest.pt',allow_resource_abort=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    assert recovered.seen==67776 and recovered.student.updates==4 and cursor['resource_abort_plan']['next_seed']==1088
    continue_after_aborted_episode(recovered);actual=recovered.reservoir.sample(3)
    assert recovered.seen==96000 and recovered.student.updates==6
    assert all(torch.equal(a,b) for a,b in zip(expected,recovered.student.model.parameters()))
    for key,values in adam['state'].items():
        for name,value in values.items():assert torch.equal(value,recovered.student.optimizer.state_dict()['state'][key][name])
    assert all(np.array_equal(batch[k],actual[k]) for k in batch)
    assert rng[0]==random.random() and rng[1]==np.random.random() and torch.equal(rng[2],torch.rand(3))


def test_old_flags_wrong_sha_and_diagnostic_identity_do_not_admit_resource_abort(tmp_path):
    session,saved,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='Interrupted collection'):RESTORE(session,tmp_path/'latest.pt',allow_closed_collection=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    with pytest.raises(ValueError,match='Unsupported partial stage'):RESTORE(session,tmp_path/'latest.pt',allow_partial=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    with pytest.raises(ValueError,match='Explicit SHA'):RESTORE(session,tmp_path/'latest.pt',allow_resource_abort=True)
    with pytest.raises(ValueError,match='SHA differs'):RESTORE(session,tmp_path/'latest.pt',allow_resource_abort=True,expected_sha256='wrong',expected_formal=False)
    with pytest.raises(ValueError,match='formal/diagnostic'):RESTORE(session,tmp_path/'latest.pt',allow_resource_abort=True,expected_sha256=saved['sha256'],expected_formal=True,updates_per_round=2)


def test_witness_interrupt_cause_cursor_and_unfinished_optimizer_are_strict(tmp_path):
    _,_,e=fixture(tmp_path);payload=torch.load(tmp_path/'latest.pt',weights_only=False)
    wrong=copy.deepcopy(payload);wrong['cursor']['next_seed']=1024
    with pytest.raises(ValueError,match='Advanced seed'):resource_abort_plan(wrong,2)
    wrong=copy.deepcopy(payload);wrong['cursor']['updates_completed_in_round']=1
    with pytest.raises(ValueError,match='optimizer work'):resource_abort_plan(wrong,2)
    path=e/'result.json';original=path.read_text();x=json.loads(original);x['requested_safe_stop']=False;path.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='explicit resource'):resource_abort_plan(payload,2)
    x['requested_safe_stop']=True;x['safety_stop']='base_fall';path.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='standing failure'):resource_abort_plan(payload,2)
    path.write_text(original);parent_path=tmp_path/'status.json';parent=json.loads(parent_path.read_text());parent['safe_stop_requested']['available_ram_bytes']=13*1024**3;parent_path.write_text(json.dumps(parent))
    with pytest.raises(ValueError,match='RAM soft-guard'):resource_abort_plan(payload,2)


def test_proposed_resource_branch_composes_with_lossless_raw_or_gzip_reader(tmp_path):
    import gzip,hashlib
    from tabletop.state_fingerprints import expected_fingerprints,session_fingerprints
    session,_,_=fixture(tmp_path);source=tmp_path/'latest.pt';payload=torch.load(source,weights_only=False)
    packed=tmp_path/'verbatim_source.pt.gz';raw=source.read_bytes();packed.write_bytes(gzip.compress(raw,compresslevel=3,mtime=0));assert gzip.decompress(packed.read_bytes())==raw
    other=tmp_path/'other';other.mkdir();restored,_,_=fixture(other,81)
    _,cursor=RESTORE(restored,packed,allow_resource_abort=True,expected_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),expected_formal=False,updates_per_round=2)
    assert session_fingerprints(restored)==expected_fingerprints(payload)
    assert cursor['resource_abort_plan']['next_seed']==1088 and restored.seen==67776 and restored.student.updates==4


def test_resource_entry_rejects_mixed_flags_stage_and_consumption(tmp_path):
    session,saved,_=fixture(tmp_path)
    for flags in (dict(allow_partial=True),dict(allow_closed_collection=True)):
        with pytest.raises(ValueError,match='mutually exclusive'):
            RESTORE(session,tmp_path/'latest.pt',allow_resource_abort=True,expected_sha256=saved['sha256'],expected_formal=False,**flags)
    with pytest.raises(ValueError,match='identity required'):
        RESTORE(session,tmp_path/'latest.pt',allow_resource_abort=True,expected_sha256=saved['sha256'])
    payload=torch.load(tmp_path/'latest.pt',weights_only=False)
    wrong=copy.deepcopy(payload);wrong['collection']['seen']+=64
    with pytest.raises(ValueError):resource_abort_plan(wrong,2)
    for phase in ('dagger_update','student_fixed_validation',None):
        import hashlib
        wrong=copy.deepcopy(payload);wrong['cursor']['partial_phase']=phase
        path=tmp_path/'wrong_stage.pt';torch.save(wrong,path)
        with pytest.raises(ValueError,match='student collection stage'):
            RESTORE(session,path,allow_resource_abort=True,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),expected_formal=False,updates_per_round=2)
