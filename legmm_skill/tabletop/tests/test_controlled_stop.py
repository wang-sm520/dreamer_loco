"""Exercise the real OS signal/save/restore path before another long budget."""
import os,signal,subprocess,sys
from pathlib import Path
import torch


def test_sigterm_keeps_actual_optimizer_checkpoint(tmp_path):
    code='''
import sys,time
from pathlib import Path
import numpy as np
from tabletop.student import Student
from tabletop.data import Reservoir
from tabletop.spec import STOW
from tabletop.storage import StorageGuard
from tabletop.lifecycle_cleanup import install_stop_handler,check_stop
student=Student('cpu',9);buffer=Reservoir(1,9)
q=np.asarray([*STOW,.072],dtype=np.float32)
buffer.add(np.zeros((2,2,128,128),dtype=np.float32),q,np.repeat(q[None],8,axis=0),dict(shape='box',outcome='synthetic_unit',round=0))
student.update(buffer,batch=8,microbatch=8)
install_stop_handler()
try:
 print('READY',flush=True)
 while True:
  check_stop('CPU idle boundary')
  time.sleep(.01)
except RuntimeError as error:
 out=Path(sys.argv[1])
 student.save(out,dict(controlled_stop=True,formal_training=False,reason=str(error)),StorageGuard(out.parent,out.parent/'code'))
'''
    checkpoint=tmp_path/'controlled_stop.pt'
    process=subprocess.Popen([sys.executable,'-c',code,str(checkpoint)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    try:
        output=[]
        for line in process.stdout:
            output.append(line)
            if line.strip()=='READY':break
        else:raise AssertionError('CPU signal fixture failed: '+''.join(output))
        process.send_signal(signal.SIGTERM);assert process.wait(timeout=30)==0
        from tabletop.student import Student
        student=Student('cpu',0);metadata=student.restore(checkpoint)
        assert student.updates==1 and metadata['controlled_stop'] and not metadata['formal_training']
        assert '15' in metadata['reason']
        assert all(torch.isfinite(parameter).all() for parameter in student.model.parameters())
    finally:
        if process.poll() is None:process.kill();process.wait()
