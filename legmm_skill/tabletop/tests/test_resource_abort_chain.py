"""Exact learner continuation across two witnessed RAM-soft-stop cohorts."""
import copy,json,shutil
from pathlib import Path
import pytest
import torch
from tabletop.tests.test_resource_abort_recovery import fixture,REASON,RESTORE
from tabletop.tests.test_closed_collection_recovery import add_samples
from tabletop.resource_abort_recovery import resource_abort_plan
from tabletop.dagger_recovery import save_latest
from tabletop.storage import StorageGuard
from tabletop.state_fingerprints import expected_fingerprints,session_fingerprints,expected_next_batch,digest


def chain_fixture(tmp_path):
    first=tmp_path/'first';first.mkdir();session,prior_saved,_=fixture(first)
    prior_payload=torch.load(first/'latest.pt',weights_only=False)
    prior_plan=resource_abort_plan(prior_payload,2)
    archive=tmp_path/'prior.pt';shutil.copy2(first/'latest.pt',archive)
    second=tmp_path/'second';second.mkdir();episode=second/'round_02_episode_0013';episode.mkdir()
    session.seen=session.physical_env_decisions=67904;session.episode_serial=14;session.collected_samples+=3;add_samples(session,1088,14,2)
    result=dict(seed=1088,num_envs=64,requested_safe_stop=True,safety_stop=None,physical_steps=332,engine_controlled_steps=332,native_delay_queue_advances=332,actual_standing_policy_calls=166,episodes=[dict(seed=i,reason=None,success=False) for i in range(1088,1152)])
    (episode/'result.json').write_text(json.dumps(result));(episode/'post_stage_memory_release.json').write_text('{}')
    (episode/'collection.json').write_text(json.dumps(dict(env_decisions=67904,physical_env_decisions=67904,collected_samples=12,round=2,interrupted=True)))
    (episode/'learning_decisions.jsonl').write_text(''.join(json.dumps(dict(frame=i*16,actual_envs=64,teacher_drives=False,env_decisions=67776+(i+1)*64))+'\n' for i in range(2)))
    status=dict(prior_payload['status'],PID=99999998,current_episode=str(episode),latest_save={},source_recovery=dict(path=str(archive),sha256=prior_saved['sha256'],env_decisions=67776,optimizer_updates=4,cursor=dict(prior_payload['cursor'],resource_abort_plan=prior_plan)))
    saved=save_latest(session,status,2,1152,14,first,StorageGuard(tmp_path,tmp_path/'code'),complete_round_boundary=False)
    (second/'status.json').write_text(json.dumps(dict(PID=99999998,num_envs=64,completed=False,exit_code=0,reason=REASON,safe_stop_requested=dict(available_ram_bytes=11*1024**3),minimum_available_ram_bytes=9*1024**3)))
    (second/'training_status.json').write_text(json.dumps(dict(PID=99999998,latest_save=saved)))
    return session,torch.load(first/'latest.pt',weights_only=False),saved,first,second,archive


def test_second_abort_restores_all_current_state_and_skips_both_consumed_prefixes(tmp_path):
    original,payload,saved,rolling,_,_=chain_fixture(tmp_path)
    plan=resource_abort_plan(payload,2)
    assert plan['resource_abort_chain_depth']==1 and plan['predecessor_consumed_decisions']==67776
    assert plan['consumed_prefix_decisions']==128 and plan['collected_in_round']==3904 and plan['remaining_collection_decisions']==28096
    assert plan['predecessor_plan']['consumed_prefix_decisions']==3776
    assert plan['next_seed']==1152 and plan['next_episode']==14 and not plan['aborted_initial_states_replayed']
    other=tmp_path/'other';other.mkdir();restored,_,_=fixture(other,71)
    _,cursor=RESTORE(restored,rolling/'latest.pt',allow_resource_abort=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    assert session_fingerprints(restored)==expected_fingerprints(payload)
    assert digest(restored.reservoir.sample(64))==expected_next_batch(payload)
    assert restored.seen==67904 and restored.student.updates==4 and cursor['next_seed']==1152


@pytest.mark.parametrize('mutation',['prior_sha','prior_counter','prior_cursor','model_changed','quota_gap','first_seed_replayed','old_log_changed','unsafe_stop','missing_receipt'])
def test_repeated_abort_never_invents_or_loses_predecessor_provenance(tmp_path,mutation):
    _,payload,_,rolling,second,archive=chain_fixture(tmp_path)
    if mutation=='prior_sha':payload['status']['source_recovery']['sha256']='0'*64
    elif mutation=='prior_counter':payload['status']['source_recovery']['env_decisions']=64000
    elif mutation=='prior_cursor':payload['status']['source_recovery']['cursor']['next_seed']=1024
    elif mutation=='model_changed':
        key=next(iter(payload['student']['model']));payload['student']['model'][key]=payload['student']['model'][key]+.125
    elif mutation=='quota_gap':payload['collection']['seen']+=64;payload['collection']['physical_env_decisions']+=64
    elif mutation=='first_seed_replayed':
        e=second/'round_02_episode_0013/result.json';x=json.loads(e.read_text());x['seed']=1024;x['episodes']=[dict(seed=i,reason=None,success=False) for i in range(1024,1088)];e.write_text(json.dumps(x))
    elif mutation=='old_log_changed':
        p=rolling/'latest_save_memory.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['sha256']='1'*64;p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    elif mutation=='unsafe_stop':
        p=second/'status.json';x=json.loads(p.read_text());x['safe_stop_requested']['available_ram_bytes']=7*1024**3;p.write_text(json.dumps(x))
    elif mutation=='missing_receipt':(second/'training_status.json').unlink()
    with pytest.raises(ValueError):resource_abort_plan(payload,2)
