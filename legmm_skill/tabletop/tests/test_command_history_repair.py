import copy
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tabletop.data import Reservoir
from tabletop.dagger_recovery import save_latest, restore_latest
from tabletop.learning import DaggerSession, EvaluationSession
from tabletop.spec import STOW, STEP_LENGTH, LOWER, UPPER
from tabletop.state_fingerprints import digest
from tabletop.storage import StorageGuard
from tabletop.student import Student

torch.set_num_threads(2)


def native(n=2):
    return np.tile(np.asarray([*STOW,.072],np.float32),(n,1))


def images(n=2):
    return np.random.default_rng(1816).random((n,2,2,128,128),dtype=np.float32)


def test_same_scene_different_pending_commands_are_distinct_actor_inputs():
    s=Student(seed=16,command_history=True)
    q=native();previous=q.copy();previous[1,1]+=.02
    x=s.training_state(q,previous)
    assert np.array_equal(x[:,:7],q) and np.array_equal(x[:,7:],previous)
    assert not np.array_equal(x[0],x[1])
    received=[]
    handle=s.model.state_encoder[0].register_forward_pre_hook(lambda m,a:received.append(a[0].detach().clone()))
    same=np.repeat(images(1),2,axis=0)
    with torch.no_grad():pred=s.prediction(same,x)
    handle.remove()
    expected=(torch.tensor(x)-torch.cat((s.mean,s.mean)))/torch.cat((s.scale,s.scale))
    assert torch.equal(received[0],expected)
    assert not torch.equal(pred[0],pred[1])
    assert not s.input_contract['privileged']
    with pytest.raises(TypeError):s.prediction(same,x,object_yaw=1.)
    with pytest.raises(ValueError):s.prediction(same,q)
    with pytest.raises(ValueError):s.training_state(q,np.zeros((2,14)))


def test_history_governor_uses_actual_previous_command_and_original_limits():
    s=Student(seed=3,command_history=True)
    q=native();last=q.copy();last[1,0]+=.02
    raw=np.tile(np.asarray([*UPPER[:6],0.],np.float32),(2,8,1))
    s.prediction=lambda im,st:torch.tensor(raw[:len(st)])
    out=s.act(images(),q,last,microbatch=2).numpy()
    delta=np.diff(np.concatenate((last[:,None],out),axis=1),axis=1)
    assert np.max(abs(delta)/STEP_LENGTH)<=1.00002
    assert np.all(out>=LOWER-1e-6) and np.all(out<=UPPER+1e-6)
    expected=last.copy();points=[]
    for j in range(8):
        d=raw[:,j]-expected
        d/=np.maximum(1.,np.max(abs(d)/STEP_LENGTH,axis=1,keepdims=True))
        expected=np.clip(expected+d,LOWER,UPPER);points.append(expected.copy())
    assert np.allclose(out,np.stack(points,1),atol=2e-7)


def test_collection_and_evaluation_use_same_own_history(tmp_path,monkeypatch):
    im=images();q=native();last=q.copy();last[:,1]+=.02
    visual=SimpleNamespace(manifest=lambda:{},observe=lambda scene:(torch.tensor(im),torch.zeros((2,8,8,3),dtype=torch.uint8),torch.ones((2,8,8))))
    monkeypatch.setattr('tabletop.learning.DualDepth',lambda device:visual)
    session=DaggerSession('cpu',2,capacity=4,command_history=True)
    states=[SimpleNamespace(seed=i,shape='box') for i in range(2)]
    session.budget=2;session.reset(states,None,None,tmp_path)
    joints=np.concatenate((q[:,:6],q[:,6,None]/2,-q[:,6,None]/2),axis=1)
    label=np.repeat(q[:,None,:],8,axis=1)
    outcomes=[SimpleNamespace(reason=None,success=False) for _ in states]
    result=session.decision(None,None,joints,list(range(6)),[6,7],torch.tensor(last),label,outcomes,0)
    assert np.array_equal(result,label)
    expected=session.student.training_state(q,last)
    for i,row in enumerate(session.reservoir.items):assert np.array_equal(row[1],expected[i])
    assert set(session.reservoir.sample(1))=={'images','proprio','label'}
    session.finish(outcomes)
    evaluation=EvaluationSession(session.student,visual)
    evaluation.reset(states,None,None,tmp_path)
    got=evaluation.decision(None,None,joints,list(range(6)),[6,7],torch.tensor(last),label,outcomes,0)
    direct=session.student.act(im,q,last).numpy()
    assert np.array_equal(got,direct)


def fixture(seed):
    s=Student('cpu',seed,command_history=True);b=Reservoir(3,seed,state_dim=14)
    q=native(1);last=q.copy();last[:,1]+=.02
    for i in range(3):
        state=s.training_state(q,last+i*.001)[0]
        label=np.repeat(q,8,axis=0);label[:,1]+=.01
        b.add(images(1)[0],state,label,dict(shape='box',outcome='timeout',round=0,seed=i))
    return SimpleNamespace(student=s,reservoir=b,visual=SimpleNamespace(manifest=lambda:{}),
        seen=32000,budget=32000,round=0,teacher_drives=True,episode_serial=1,physical_env_decisions=32000,collected_samples=3)


def test_history_checkpoint_full_model_adam_buffer_rng_and_contract(tmp_path):
    a=fixture(4);loss=a.student.update(a.reservoir,batch=2,microbatch=1)
    assert np.isfinite(loss) and a.student.updates==1
    p=tmp_path/'student.pt';a.student.save(p,dict(test=True))
    c=Student.from_checkpoint(p);assert c.command_history and c.updates==1
    assert digest(c.model.state_dict())==digest(a.student.model.state_dict())
    with pytest.raises(ValueError,match='contract'):Student().restore(p)
    payload=torch.load(p,weights_only=False);payload['input_contract']['state_order'].reverse()
    torch.save(payload,tmp_path/'wrong.pt')
    with pytest.raises(ValueError,match='contract'):Student(command_history=True).restore(tmp_path/'wrong.pt')
    state=dict(formal_training=False,current_round_update_count=1,current_round_loss_sum=loss)
    record=save_latest(a,state,1,320,1,tmp_path,StorageGuard(tmp_path,tmp_path/'code'),compress=True)
    expected_rng=torch.get_rng_state().clone();expected_random=random.getstate()
    expect=a.reservoir.sample(2)
    b=fixture(5);restore_latest(b,tmp_path/'latest.pt',expected_sha256=record['sha256'])
    assert torch.equal(torch.get_rng_state(),expected_rng) and random.getstate()==expected_random
    assert digest(a.student.model.state_dict())==digest(b.student.model.state_dict())
    assert digest(a.student.optimizer.state_dict())==digest(b.student.optimizer.state_dict())
    actual=b.reservoir.sample(2)
    assert all(np.array_equal(expect[k],actual[k]) for k in expect)
    assert all(np.array_equal(x[1],y[1]) for x,y in zip(a.reservoir.items,b.reservoir.items))
    old=fixture(6);old.student=Student();old.reservoir=Reservoir(3,6)
    with pytest.raises(ValueError,match='dimensions'):restore_latest(old,tmp_path/'latest.pt')


def test_legacy_buffer_rejects_history_and_new_buffer_rejects_missing_history():
    q=native(1)[0];im=images(1)[0];labels=np.repeat(q[None],8,axis=0)
    meta=dict(shape='box',outcome='timeout',round=0)
    with pytest.raises(ValueError):Reservoir(2,1).add(im,np.r_[q,q],labels,meta)
    with pytest.raises(ValueError):Reservoir(2,1,state_dim=14).add(im,q,labels,meta)


def test_ppo_and_frozen_feature_boundary_retains_history_without_actor_truth(tmp_path):
    from tabletop.residual_ppo import FrozenStudentFeatures
    s=Student('cpu',19,command_history=True)
    path=tmp_path/'history.pt';s.save(path,dict(test=True))
    loaded=Student.from_checkpoint(path,'cpu',19)
    q=native();last=q.copy();last[:,1]+=.02;im=images()
    features=FrozenStudentFeatures(loaded)
    policy,base=features.observe(im,q,torch.tensor(last))
    assert set(policy)=={'policy'} and policy['policy'].shape==(2,547)
    assert base.shape==(2,4,7)
    assert torch.equal(base,loaded.act(im,q,last)[:,:4])
    assert np.array_equal(last[:,1],q[:,1]+.02)
    features.close()


def test_default_legacy_outputs_gradients_and_rng_match_exact_original():
    path=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_command_history_repair_20261116/source_before/student.py')
    if not path.exists():pytest.skip('Exact bdi before-source fixture absent')
    spec=importlib.util.spec_from_file_location('tabletop.before_history_repair_student',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reports=[]
    for cls in [module.Student,Student]:
        torch.manual_seed(80016)
        s=cls('cpu',17);buf=Reservoir(2,5)
        q=native();im=images();label=np.repeat(q[:,None],8,axis=1);label[:,:,1]+=.01
        for i in range(2):buf.add(im[i],q[i],label[i],dict(shape='box',outcome='success',round=0))
        initial=digest(s.model.state_dict());act=s.act(im,q,q).numpy()
        loss=s.update(buf,batch=2,microbatch=1)
        reports.append(dict(initial=initial,parameters=sum(p.numel() for p in s.model.parameters()),
            action=digest(act),loss=loss,model=digest(s.model.state_dict()),
            optimizer=digest(s.optimizer.state_dict()),rng=digest(torch.get_rng_state())))
    assert reports[0]==reports[1]
    Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_command_history_repair_20261116/legacy_default_equivalence.json').write_text(json.dumps(dict(completed=True,all_exact=True,before=reports[0],after=reports[1]),indent=2)+'\n')
