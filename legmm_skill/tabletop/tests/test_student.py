import numpy as np
import pytest
import torch
from tabletop.student import Student
from tabletop.data import Reservoir
from tabletop.spec import STOW,STEP_LENGTH,LOWER,UPPER


def test_student_update_restore_and_deployment_contract(tmp_path):
    torch.set_num_threads(2)
    rng=np.random.default_rng(7)
    images=rng.random((2,2,128,128),dtype=np.float32)
    proprio=np.array([*STOW,.072],dtype=np.float32)
    labels=np.repeat(proprio[None],8,axis=0);labels[:,0]+=.004
    buffer=Reservoir(2,7)
    buffer.add(images,proprio,labels,dict(shape='box',outcome='collision',round=0,object_yaw=1.))
    student=Student(seed=8)
    loss=student.update(buffer,batch=2,microbatch=1)
    assert np.isfinite(loss) and student.updates==1
    before=student.act(images[None],proprio[None],proprio[None]).numpy()
    step=np.diff(np.concatenate((proprio[None,None],before),axis=1),axis=1)
    assert np.max(abs(step)/STEP_LENGTH)<=1.00001
    assert np.all(before>=LOWER-1e-6) and np.all(before<=UPPER+1e-6)
    path=tmp_path/'student.pt';student.save(path,dict(physics=[4,0]))
    restored=Student(seed=9);assert restored.restore(path)==dict(physics=[4,0])
    assert restored.updates==1
    assert restored.act(images[None],proprio[None],proprio[None]).numpy()==pytest.approx(before)
    with pytest.raises(TypeError):restored.prediction(images[None],proprio[None],object_yaw=1.)
    assert set(buffer.sample(1))=={'images','proprio','label'}
