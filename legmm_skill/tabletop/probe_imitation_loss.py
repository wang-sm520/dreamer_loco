"""One-variable CPU diagnostic; never changes the active formal learner."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from .student import Student
from .data import Reservoir
from .spec import STEP_LENGTH,OUTPUT,WORKTREE
from .storage import StorageGuard


def native_update(student,buffer):
    samples=buffer.sample(64);student.model.train();student.optimizer.zero_grad(set_to_none=True);total=0.
    scale=torch.as_tensor(STEP_LENGTH,dtype=torch.float32,device=student.device)
    for i in range(0,64,8):
        prediction=student.prediction(samples['images'][i:i+8],samples['proprio'][i:i+8])
        target=torch.as_tensor(samples['label'][i:i+8],device=student.device)
        loss=(((prediction-target)/scale)**2).mean()/8
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native-unit loss')
        loss.backward();total+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(student.model.parameters(),1.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite probe gradient')
    student.optimizer.step();student.updates+=1;student.model.eval();return total


@torch.no_grad()
def evaluate(student,data):
    prediction=torch.cat([student.prediction(data['images'][i:i+8],data['proprio'][i:i+8]) for i in range(0,len(data['proprio']),8)]).numpy()
    error=(prediction[:,:4]-data['labels'][:,:4])/STEP_LENGTH[None,None,:]
    return dict(first4_native_step_rmse=float(np.sqrt(np.mean(error**2))),
        first4_per_axis_native_step_rmse=np.sqrt(np.mean(error**2,axis=(0,1))).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--updates',type=int,default=250);a=p.parse_args()
    if not a.out.resolve().is_relative_to(OUTPUT):raise ValueError('W8 diagnostic output required')
    a.out.mkdir(exist_ok=False);guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');guard.check()
    with a.checkpoint.open('rb') as file:
        digest=hashlib.sha256()
        while chunk:=file.read(8*1024**2):digest.update(chunk)
        file.seek(0);source=torch.load(file,map_location='cpu',weights_only=False)
    if source['format']!='w8_dagger_full_round_v1' or not source['status']['formal_training']:raise ValueError('Actual formal full-round state required')
    if any(not 0<=row[3]['seed']<10000 for row in source['reservoir']['items']):raise ValueError('Only training data')
    students=[];buffers=[]
    for _ in range(2):
        student=Student('cpu',0);student.model.load_state_dict(source['student']['model'])
        student.optimizer.load_state_dict(copy.deepcopy(source['student']['optimizer']));student.updates=source['student']['updates'];student.model.eval()
        buffer=Reservoir(source['reservoir']['capacity'],source['reservoir']['seed']);buffer.items=source['reservoir']['items']
        buffer.rng.setstate(source['reservoir']['rng']);students.append(student);buffers.append(buffer)
    indices=np.random.default_rng(0).choice(len(buffers[0].items),size=256,replace=False)
    train=dict(images=np.stack([buffers[0].items[i][0] for i in indices]),proprio=np.stack([buffers[0].items[i][1] for i in indices]),labels=np.stack([buffers[0].items[i][2] for i in indices]))
    held=dict(np.load(a.cache/'observations.npz'));del source
    status=dict(PID=os.getpid(),phase='CPU_loss_units_pair',completed=False,formal_training=False,Isaac_App_started=False,
        environment_interactions=0,source_checkpoint=str(a.checkpoint),source_sha256=digest.hexdigest(),
        updates_per_arm=a.updates,effective_batch=64,microbatch=8,learning_rate=1e-4,changed_variable='loss denominator only: joint half-range vs native step',
        paired_initial_weights_optimizer_and_sample_rng=True,before=dict(train=evaluate(students[0],train),independent_train_cache=evaluate(students[0],held)),adopted=False)
    def save():(a.out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();started=time.monotonic()
    for update in range(a.updates):
        old=students[0].update(buffers[0],batch=64,microbatch=8);new=native_update(students[1],buffers[1])
        if buffers[0].rng.getstate()!=buffers[1].rng.getstate():raise RuntimeError('Paired sample streams diverged')
        if update%25==0:
            status.update(completed_updates_each=update+1,elapsed_s=time.monotonic()-started,last_losses_different_units=[old,new]);save()
    status.update(completed=True,phase='CPU_loss_pair_complete',completed_updates_each=a.updates,
        real_optimizer_updates_total=2*a.updates,elapsed_s=time.monotonic()-started,
        original=dict(train=evaluate(students[0],train),independent_train_cache=evaluate(students[0],held)),
        native=dict(train=evaluate(students[1],train),independent_train_cache=evaluate(students[1],held)))
    if status['native']['independent_train_cache']['first4_native_step_rmse']<.9*status['original']['independent_train_cache']['first4_native_step_rmse']:
        path=a.out/'diagnostic_native_candidate.pt';students[1].save(path,dict(diagnostic_only=True,source_sha256=digest.hexdigest(),loss_units='native_step',not_formal_rollout_qualified=True),guard=guard);status['diagnostic_candidate']=str(path)
    save()


if __name__=='__main__':main()
