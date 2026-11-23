import copy,hashlib,json,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tabletop.data import Reservoir
from tabletop.dagger_recovery import save_latest
from tabletop.closed_collection_recovery import closed_collection_plan
from tabletop.prepare_closed_collection_patch import recovery_source
from tabletop.storage import StorageGuard

ROOT=Path(__file__).parents[1]
namespace={'__name__':'tabletop._closed_collection_candidate','__package__':'tabletop'}
source=(ROOT/'dagger_recovery.py').read_text()
exec(compile(source if 'allow_closed_collection=False' in source else recovery_source(source),'closed_recovery_candidate','exec'),namespace)
RESTORE=namespace['restore_latest']


class TinyStudent:
    def __init__(self,seed):
        torch.manual_seed(seed);self.device=torch.device('cpu');self.model=torch.nn.Linear(1,1);self.model.config={'unit_fixture':1}
        self.mean=torch.zeros(7);self.scale=torch.ones(7);self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-3);self.updates=0
    def update(self,buffer):
        batch=buffer.sample(2);x=torch.tensor(batch['proprio'][:,:1]);target=torch.tensor(batch['label'][:,0,:1])
        self.model.train();self.optimizer.zero_grad();((self.model(x)-target)**2).mean().backward();self.optimizer.step();self.updates+=1;self.model.eval()


def add_samples(session,seed,episode,round_index):
    for i in range(3):
        value=random.random()+float(np.random.random())+float(torch.rand(()))
        q=np.full(7,value*.01,np.float32);labels=np.repeat((q+.002)[None],8,axis=0)
        session.reservoir.add(np.full((2,2,128,128),value*.1,np.float32),q,labels,
            dict(shape='box',outcome='timeout',round=round_index,seed=seed+i,episode=episode,frame=i*16))


def fixture(tmp_path,seed=7):
    random.seed(seed);np.random.seed(seed);student=TinyStudent(seed)
    session=SimpleNamespace(student=student,reservoir=Reservoir(4,seed),visual=SimpleNamespace(manifest=lambda:{}),
        seen=44032,budget=64000,round=1,teacher_drives=False,episode_serial=8,physical_env_decisions=44032,collected_samples=6)
    add_samples(session,256,6,0)
    for _ in range(2):student.update(session.reservoir)
    add_samples(session,704,8,1)
    episode=tmp_path/'closed_cohort';episode.mkdir()
    result=dict(seed=704,num_envs=64,requested_safe_stop=False,safety_stop=None,physical_steps=1800,
        engine_controlled_steps=1800,native_delay_queue_advances=1800,actual_standing_policy_calls=900,
        episodes=[dict(seed=i,reason='timeout') for i in range(704,768)])
    (episode/'result.json').write_text(json.dumps(result));(episode/'post_stage_memory_release.json').write_text('{}')
    (episode/'collection.json').write_text(json.dumps(dict(env_decisions=44032,round=1,interrupted=False,physical_env_decisions=44032,collected_samples=6)))
    (episode/'learning_decisions.jsonl').write_text(''.join(json.dumps(dict(frame=16*i,actual_envs=64,teacher_drives=False,env_decisions=38016+(i+1)*64))+'\n' for i in range(94)))
    status=dict(formal_training=False,diagnostic_only=True,synthetic_CPU_test=True,num_envs=64,current_episode=str(episode),
        last_phase='dagger_student_collection',current_round_update_count=0,current_round_loss_sum=0.,current_round_update_elapsed_s=0.)
    saved=save_latest(session,status,1,768,8,tmp_path,StorageGuard(tmp_path,tmp_path/'code'),complete_round_boundary=False)
    return session,saved,episode


def continuation(session):
    add_samples(session,768,9,1);remaining=session.budget-session.seen;session.seen+=remaining;session.physical_env_decisions+=remaining;session.collected_samples+=3;session.episode_serial+=1
    for _ in range(2):session.student.update(session.reservoir)


def test_closed_collection_next_samples_adam_rng_and_quota_match_uninterrupted(tmp_path):
    torch.set_num_threads(2);original,saved,episode=fixture(tmp_path)
    payload=torch.load(tmp_path/'latest.pt',weights_only=False);plan=closed_collection_plan(payload,updates_per_round=2)
    assert plan['collected_in_round']==12032 and plan['remaining_collection_decisions']==19968 and plan['remaining_round_updates']==2
    assert plan['next_seed']==768 and not plan['physical_episode_resumed']
    continuation(original);expected_sample=original.reservoir.sample(3);expected_random=(random.random(),float(np.random.random()),torch.rand(2))
    other=tmp_path/'other';other.mkdir();recovered,_,_=fixture(other,99)
    state,cursor=RESTORE(recovered,tmp_path/'latest.pt',allow_closed_collection=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    assert recovered.seen==44032 and recovered.student.updates==2 and cursor['next_round']==1
    assert cursor['closed_collection_plan']['remaining_collection_decisions']==19968
    continuation(recovered);actual_sample=recovered.reservoir.sample(3)
    assert recovered.seen==64000 and recovered.student.updates==4 and recovered.physical_env_decisions==64000
    assert all(torch.equal(a,b) for a,b in zip(original.student.model.parameters(),recovered.student.model.parameters()))
    first=original.student.optimizer.state_dict();second=recovered.student.optimizer.state_dict();assert first['param_groups']==second['param_groups']
    for key,values in first['state'].items():
        for name,value in values.items():assert torch.equal(value,second['state'][key][name])
    assert all(np.array_equal(expected_sample[k],actual_sample[k]) for k in expected_sample)
    actual=(random.random(),float(np.random.random()),torch.rand(2));assert actual[:2]==expected_random[:2] and torch.equal(actual[2],expected_random[2])
    assert original.reservoir.composition==recovered.reservoir.composition and original.reservoir.rng.getstate()==recovered.reservoir.rng.getstate()


def test_v6_flag_does_not_authorize_collection_or_wrong_sha_identity(tmp_path):
    session,saved,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='Unsupported partial stage'):
        RESTORE(session,tmp_path/'latest.pt',allow_partial=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    with pytest.raises(ValueError,match='SHA'):RESTORE(session,tmp_path/'latest.pt',allow_closed_collection=True)
    with pytest.raises(ValueError,match='SHA differs'):RESTORE(session,tmp_path/'latest.pt',allow_closed_collection=True,expected_sha256='wrong')
    with pytest.raises(ValueError,match='formal/diagnostic'):RESTORE(session,tmp_path/'latest.pt',allow_closed_collection=True,expected_sha256=saved['sha256'],expected_formal=True)


def test_interrupted_unresolved_missing_witness_and_bad_cursors_are_rejected(tmp_path):
    _,_,episode=fixture(tmp_path);payload=torch.load(tmp_path/'latest.pt',weights_only=False)
    wrong=copy.deepcopy(payload);wrong['cursor']['next_seed']+=64
    with pytest.raises(ValueError,match='cursor differs'):closed_collection_plan(wrong,2)
    wrong=copy.deepcopy(payload);wrong['student']['updates']+=1
    with pytest.raises(ValueError,match='Model counter'):closed_collection_plan(wrong,2)
    result_path=episode/'result.json';result=json.loads(result_path.read_text());result['requested_safe_stop']=True;result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='Interrupted collection'):closed_collection_plan(payload,2)
    result['requested_safe_stop']=False;result['episodes'][0]['reason']=None;result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='Unresolved'):closed_collection_plan(payload,2)
    result['episodes'][0]['reason']='timeout';result_path.write_text(json.dumps(result));(episode/'post_stage_memory_release.json').unlink()
    with pytest.raises(ValueError,match='Missing closed cohort witness'):closed_collection_plan(payload,2)


def test_prepared_loader_keeps_existing_update_and_full_round_recovery(tmp_path):
    session,_,episode=fixture(tmp_path)
    session.seen=session.budget;session.physical_env_decisions=session.budget;session.student.update(session.reservoir)
    status=dict(formal_training=False,diagnostic_only=True,synthetic_CPU_test=True,num_envs=64,current_episode=str(episode),
        last_phase='dagger_update',current_round_update_count=1,current_round_loss_sum=.2,current_round_update_elapsed_s=.1)
    update_root=tmp_path/'update';update_root.mkdir();saved=save_latest(session,status,1,768,8,update_root,StorageGuard(tmp_path,tmp_path/'code'),complete_round_boundary=False)
    other=tmp_path/'other';other.mkdir();recovered,_,_=fixture(other,18)
    _,cursor=RESTORE(recovered,update_root/'latest.pt',allow_partial=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    assert cursor['partial_phase']=='dagger_update' and recovered.student.updates==3
    recovered.student.update(recovered.reservoir);status['current_round_update_count']=2
    full_root=tmp_path/'full';full_root.mkdir();full=save_latest(recovered,status,2,768,8,full_root,StorageGuard(tmp_path,tmp_path/'code'))
    _,cursor=RESTORE(session,full_root/'latest.pt',expected_sha256=full['sha256'],expected_formal=False)
    assert cursor['next_round']==2 and cursor['partial_phase'] is None and session.student.updates==4
