import copy,hashlib,random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tabletop.data import Reservoir
from tabletop.dagger_recovery import save_latest,restore_latest,partial_update_plan
from tabletop.storage import StorageGuard


class TinyStudent:
    """Small real Adam model isolates recovery accounting from vision cost."""
    def __init__(self,seed):
        with torch.random.fork_rng():
            torch.manual_seed(seed);self.model=torch.nn.Linear(7,56)
        self.model.config=dict(synthetic_recovery_test=True);self.device=torch.device('cpu')
        self.mean=torch.zeros(7);self.scale=torch.ones(7);self.updates=0
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
    def update(self,buffer):
        batch=buffer.sample(8);self.optimizer.zero_grad(set_to_none=True)
        prediction=self.model(torch.tensor(batch['proprio'])).reshape(-1,8,7)
        loss=(prediction-torch.tensor(batch['label'])).square().mean();loss.backward()
        self.optimizer.step();self.updates+=1;return float(loss.detach())


def make_session(seed=9):
    buffer=Reservoir(3,9)
    for i in range(3):
        buffer.add(np.zeros((2,2,128,128),np.float16),np.ones(7,np.float32)*i,
            np.ones((8,7),np.float32)*(i+.1),dict(shape='box',outcome='synthetic_unit',round=0))
    return SimpleNamespace(student=TinyStudent(seed),reservoir=buffer,visual=SimpleNamespace(manifest=lambda:{}),
        seen=32000,budget=32000,round=0,teacher_drives=True,episode_serial=6,physical_env_decisions=32000,collected_samples=26664)


def test_partial_update_matches_uninterrupted_and_rejects_wrong_identity(tmp_path):
    torch.set_num_threads(2)
    reference=make_session();first_loss=reference.student.update(reference.reservoir)
    expected_rng=reference.reservoir.rng.getstate();probe=make_session().reservoir;probe.rng.setstate(expected_rng)
    expected_next=probe.sample(8);second_loss=reference.student.update(reference.reservoir)
    interrupted=make_session();assert interrupted.student.update(interrupted.reservoir)==first_loss
    state=dict(formal_training=False,synthetic_test=True,last_phase='dagger_update',current_round_update_count=1,
        current_round_loss_sum=first_loss,current_round_update_elapsed_s=.5)
    saved=save_latest(interrupted,state,0,640,6,tmp_path,StorageGuard(tmp_path,tmp_path/'code'),complete_round_boundary=False)
    restored=make_session(10)
    with pytest.raises(ValueError,match='SHA differs'):restore_latest(restored,tmp_path/'latest.pt',True,'0'*64)
    with pytest.raises(ValueError,match='formal/diagnostic'):restore_latest(restored,tmp_path/'latest.pt',True,saved['sha256'],True)
    _,cursor=restore_latest(restored,tmp_path/'latest.pt',True,saved['sha256'],False)
    probe=make_session().reservoir;probe.rng.setstate(restored.reservoir.rng.getstate());actual_next=probe.sample(8)
    assert all(np.array_equal(actual_next[key],expected_next[key]) for key in actual_next)
    assert restored.student.update(restored.reservoir)==second_loss
    assert restored.student.updates==reference.student.updates==2
    assert all(torch.equal(a,b) for a,b in zip(restored.student.model.parameters(),reference.student.model.parameters()))
    for key,values in reference.student.optimizer.state_dict()['state'].items():
        for name,value in values.items():assert torch.equal(value,restored.student.optimizer.state_dict()['state'][key][name])
    assert (cursor['loss_sum']+second_loss)/2==(first_loss+second_loss)/2
    payload=torch.load(tmp_path/'latest.pt',weights_only=False)
    plan=partial_update_plan(payload,2000)
    assert plan['remaining_updates']==1999 and plan['missing_collection_decisions']==0 and plan['next_seed']==640
    assert len(range(plan['completed_updates'],2000))==1999
    bad=copy.deepcopy(payload);bad['cursor']['partial_phase']='dagger_student_collection'
    with pytest.raises(ValueError,match='Unsupported partial stage'):partial_update_plan(bad)
    missing=copy.deepcopy(payload);missing['collection']['seen']=31936
    with pytest.raises(ValueError,match='missing collection'):partial_update_plan(missing)
