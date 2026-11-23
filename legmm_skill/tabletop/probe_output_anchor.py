"""Paired fresh CPU diagnostic of one output anchor; no formal model changes."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from .student import Student
from .data import Reservoir,actor_inputs
from .probe_imitation_loss import evaluate
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard
from .lifecycle_cleanup import memory_snapshot


class MeasuredAnchorStudent(Student):
    def prediction(self,images,proprio):
        actor_inputs(images,proprio)
        images=torch.as_tensor(images,device=self.device,dtype=torch.float32)
        proprio=torch.as_tensor(proprio,device=self.device,dtype=torch.float32)
        tokens=torch.ones((len(proprio),1),device=self.device,dtype=torch.long)
        return self.model((proprio-self.mean)/self.scale,tokens,images)*self.scale+proprio[:,None,:]


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--updates',type=int,default=400);a=p.parse_args()
    if not a.out.resolve().is_relative_to(OUTPUT):raise ValueError('W8 diagnostic output required')
    a.out.mkdir(exist_ok=False);guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');guard.check()
    with a.checkpoint.open('rb') as file:
        digest=hashlib.sha256()
        while chunk:=file.read(8*1024**2):digest.update(chunk)
        file.seek(0);source=torch.load(file,map_location='cpu',weights_only=False)
    if source['format']!='w8_dagger_full_round_v1' or not source['status']['formal_training']:raise ValueError('Formal full-round source data required')
    if any(not 0<=row[3]['seed']<10000 for row in source['reservoir']['items']):raise ValueError('Only training observations')
    # Fresh identical weights and optimizer: the source weights are never reused.
    students=[Student('cpu',256),MeasuredAnchorStudent('cpu',256)];buffers=[]
    assert all(torch.equal(a,b) for a,b in zip(students[0].model.parameters(),students[1].model.parameters()))
    for _ in range(2):
        buffer=Reservoir(source['reservoir']['capacity'],source['reservoir']['seed']);buffer.items=source['reservoir']['items']
        buffer.rng.setstate(source['reservoir']['rng']);buffers.append(buffer)
    indices=np.random.default_rng(0).choice(len(buffers[0].items),256,replace=False)
    train=dict(images=np.stack([buffers[0].items[i][0] for i in indices]),proprio=np.stack([buffers[0].items[i][1] for i in indices]),labels=np.stack([buffers[0].items[i][2] for i in indices]))
    held=dict(np.load(a.cache/'observations.npz'));del source
    status=dict(PID=os.getpid(),completed=False,phase='CPU_output_anchor_pair',formal_training=False,Isaac_App_started=False,
        environment_interactions=0,formal_updates=0,source_checkpoint=str(a.checkpoint),source_sha256=digest.hexdigest(),
        source_used_as='exact real training reservoir only; fresh seed256 model/Adam for both arms',changed_variable='output anchor mean vs measured proprio; same normalized loss, scale, architecture, batch and LR',
        actor_inputs=['dual_depth','measured_joint_state'],teacher_truth_is_supervised_target_only=True,
        updates_each=a.updates,batch=64,microbatch=8,learning_rate=1e-4,adopted=False,
        before=[dict(train=evaluate(s,train),independent_train_cache=evaluate(s,held)) for s in students])
    def save():(a.out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();start=time.monotonic()
    for update in range(a.updates):
        memory=memory_snapshot()
        if memory['available_ram_bytes']<20*1024**3:
            # Evaluate existing weights before releasing them; this is a paired
            # partial diagnostic, never a completed400-update claim.
            status.update(phase='CPU_probe_yields_host_margin',completed_updates_each=update,
                diagnostic_optimizer_updates=2*update,elapsed_s=time.monotonic()-start,
                available_ram_bytes_at_yield=memory['available_ram_bytes'],
                original=dict(train=evaluate(students[0],train),independent_train_cache=evaluate(students[0],held)),
                measured_anchor=dict(train=evaluate(students[1],train),independent_train_cache=evaluate(students[1],held)))
            save();return
        first=students[0].update(buffers[0]);second=students[1].update(buffers[1])
        if buffers[0].rng.getstate()!=buffers[1].rng.getstate():raise RuntimeError('Paired sample stream differs')
        if update%25==0:
            status.update(completed_updates_each=update+1,last_loss=[first,second],elapsed_s=time.monotonic()-start);save()
    results=[dict(train=evaluate(s,train),independent_train_cache=evaluate(s,held)) for s in students]
    status.update(completed=True,phase='CPU_output_anchor_complete',completed_updates_each=a.updates,
        diagnostic_optimizer_updates=2*a.updates,elapsed_s=time.monotonic()-start,original=results[0],measured_anchor=results[1])
    better=all(results[1][key]['first4_native_step_rmse']<.8*results[0][key]['first4_native_step_rmse'] for key in ('train','independent_train_cache'))
    if better:
        import io
        stream=io.BytesIO();torch.save(dict(format='w8_cpu_output_anchor_diagnostic_v1',model=students[1].model.state_dict(),
            model_config=students[1].model.config,diagnostic_only=True,not_formal_rollout_qualified=True,source_sha256=digest.hexdigest(),updates=a.updates),stream)
        data=stream.getvalue();guard.check(len(data));path=a.out/'diagnostic_model_only.pt'
        with path.open('xb') as file:file.write(data);file.flush();os.fsync(file.fileno())
        status['diagnostic_model_only']=str(path)
    save()


if __name__=='__main__':main()
