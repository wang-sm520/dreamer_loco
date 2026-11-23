import numpy as np
import torch
from tabletop.audit_student_actions import slot_sample,metric
from tabletop.control import DelayedCommands
from tabletop.spec import STOW,LOWER,UPPER,STEP_LENGTH
from tabletop.student import Student


def test_fixed_slot_recovers_target_index_and_is_not_always_latest():
    ring=np.full(11,-1.)
    for tick in range(300+160):
        frame=tick-300
        if frame>=0 and frame%16==0:
            sample=slot_sample(frame)
            expected=-1 if sample['slot_write_action_frame']<0 else sample['slot_write_action_frame']//4
            assert ring[10]==expected
        ring[tick%11]=-1 if frame<0 else frame//4
    assert slot_sample(16)['target_index']==1
    assert slot_sample(32)['target_index']==3
    assert slot_sample(384)['target_index']==3
    assert metric(np.array([[0]*6+[.01]]))['opening_RMSE_mm']==10.


def test_actual_student_projection_four_step_hold_delay_and_finger_units():
    student=Student.__new__(Student);student.device=torch.device('cpu');student.model=torch.nn.Identity()
    student.mean=torch.tensor((LOWER+UPPER)/2,dtype=torch.float32);student.scale=torch.tensor((UPPER-LOWER)/2,dtype=torch.float32)
    initial=np.array([*STOW,.072],dtype=np.float32)
    desired=initial+np.array([.02,.01,0,0,0,0,-.005],dtype=np.float32)
    student.prediction=lambda images,q:torch.tensor(desired)[None,None].repeat(1,8,1)
    chunk=student.act(np.zeros((1,2,2,128,128),np.float32),initial[None],initial[None])[0].numpy()
    np.testing.assert_allclose(chunk[0]-initial,[.01,.005,0,0,0,0,-.0025],atol=2e-7)
    np.testing.assert_allclose(chunk[1],desired,atol=2e-7)
    assert np.max(abs(np.diff(np.concatenate((initial[None],chunk),axis=0),axis=0))/STEP_LENGTH)<1.0001
    issued=np.repeat(chunk[:4],4,axis=0)
    delay=DelayedCommands(initial);applied=np.array([delay.step(q) for q in issued])
    np.testing.assert_array_equal(applied[:10],np.repeat(initial[None],10,axis=0))
    np.testing.assert_array_equal(applied[10:14],np.repeat(chunk[:1],4,axis=0))
    motors=applied[:,6,None]*np.array([[.5,-.5]])
    np.testing.assert_allclose(motors[:,0]-motors[:,1],applied[:,6])
