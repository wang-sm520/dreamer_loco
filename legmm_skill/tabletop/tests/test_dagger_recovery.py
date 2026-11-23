import os,random,signal
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tabletop.student import Student
from tabletop.data import Reservoir
from tabletop.spec import STOW
from tabletop.storage import StorageGuard
from tabletop.dagger_recovery import save_latest,restore_latest
from tabletop.lifecycle_cleanup import install_stop_handler,stop_requested,check_stop,SafeStop


def fixture_session(seed):
    buffer=Reservoir(3,seed)
    q=np.asarray([*STOW,.072],dtype=np.float32)
    for i in range(3):
        buffer.add(np.full((2,2,128,128),i/10,dtype=np.float32),q+i*.001,
            np.repeat(q[None],8,axis=0),dict(shape='box',outcome='synthetic_unit',round=0,seed=i,episode=1,frame=16*i))
    return SimpleNamespace(student=Student('cpu',seed),reservoir=buffer,visual=SimpleNamespace(manifest=lambda:{}),
        seen=32000,budget=32000,round=0,teacher_drives=True,episode_serial=6,physical_env_decisions=32000,collected_samples=27000)


def test_signal_inside_optimizer_is_deferred_and_full_state_roundtrips(tmp_path,monkeypatch):
    torch.set_num_threads(2);session=fixture_session(9)
    original_term=signal.getsignal(signal.SIGTERM);original_int=signal.getsignal(signal.SIGINT)
    try:
        install_stop_handler();real_step=session.student.optimizer.step
        def signalled_step(*args,**kwargs):
            os.kill(os.getpid(),signal.SIGTERM)
            assert stop_requested()
            return real_step(*args,**kwargs)
        monkeypatch.setattr(session.student.optimizer,'step',signalled_step)
        loss=session.student.update(session.reservoir,batch=8,microbatch=8)
        assert session.student.updates==1
        with pytest.raises(SafeStop):check_stop('after complete optimizer update')
        state=dict(current_round=0,last_phase='dagger_update',current_round_update_count=1,
            current_round_loss_sum=loss,current_round_update_elapsed_s=.1,formal_training=False,synthetic_test=True)
        guard=StorageGuard(tmp_path,tmp_path/'code')
        saved=save_latest(session,state,0,640,6,tmp_path,guard,complete_round_boundary=False)
        assert saved['bytes']==(tmp_path/'latest.pt').stat().st_size
        assert not (tmp_path/'latest.pt.pending').exists()
        expected_rng=torch.get_rng_state().clone();expected_python=random.getstate();expected_numpy=np.random.get_state()
        expected_sample=session.reservoir.sample(8)
        recovered=fixture_session(10)
        with pytest.raises(ValueError,match='Partial recovery'):restore_latest(recovered,tmp_path/'latest.pt')
        restored,cursor=restore_latest(recovered,tmp_path/'latest.pt',allow_partial=True,expected_sha256=saved['sha256'])
        assert cursor['partial_phase']=='dagger_update' and cursor['updates_completed_in_round']==1
        assert recovered.seen==32000 and recovered.budget==32000 and recovered.episode_serial==6
        assert recovered.physical_env_decisions==32000 and recovered.collected_samples==27000
        assert recovered.student.updates==1 and recovered.reservoir.composition==session.reservoir.composition
        assert torch.equal(torch.get_rng_state(),expected_rng) and random.getstate()==expected_python
        actual_numpy=np.random.get_state();assert actual_numpy[0]==expected_numpy[0] and np.array_equal(actual_numpy[1],expected_numpy[1]) and actual_numpy[2:]==expected_numpy[2:]
        actual_sample=recovered.reservoir.sample(8)
        assert all(np.array_equal(expected_sample[key],actual_sample[key]) for key in expected_sample)
        assert all(torch.equal(a,b) for a,b in zip(session.student.model.parameters(),recovered.student.model.parameters()))
        first=session.student.optimizer.state_dict();second=recovered.student.optimizer.state_dict()
        assert first['param_groups']==second['param_groups']
        for key,values in first['state'].items():
            for name,value in values.items():assert torch.equal(value,second['state'][key][name])
        for original,restored_row in zip(session.reservoir.items,recovered.reservoir.items):
            assert all(np.array_equal(a,b) for a,b in zip(original[:3],restored_row[:3])) and original[3]==restored_row[3]
        install_stop_handler()
        save_latest(recovered,state,1,640,6,tmp_path,guard)
        full=fixture_session(11);_,cursor=restore_latest(full,tmp_path/'latest.pt')
        assert cursor['next_round']==1 and cursor['partial_phase'] is None and full.student.updates==1
        assert len(list(tmp_path.glob('*.pt')))==1 and not (tmp_path/'latest.pt.pending').exists()
    finally:
        install_stop_handler();signal.signal(signal.SIGTERM,original_term);signal.signal(signal.SIGINT,original_int)
