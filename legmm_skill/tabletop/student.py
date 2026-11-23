"""W8 dual-depth student and bounded-batch DAgger updates.

Only measured arm/finger joints and the two preprocessed depth views enter the
actor. Privileged scene state and teacher memory are deliberately absent.
"""
from pathlib import Path
import io,os
import numpy as np
import torch
from airbot_depth.model import AirbotPolicyModel
from .spec import LOWER,UPPER,STEP_LENGTH
from .data import actor_inputs


class Student:
    command_history=False
    relative_delta=False
    _formats={
        (False,False):'w8_dual_depth_student_v1',
        (True,False):'w8_own_last_command_student_v1',
        (False,True):'w8_reldelta_student_v1',
        (True,True):'w8_own_last_command_reldelta_student_v1',
    }

    def __init__(self,device='cpu',seed=0,command_history=False,relative_delta=False):
        self.device=torch.device(device)
        self.command_history=bool(command_history)
        self.relative_delta=bool(relative_delta)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model=AirbotPolicyModel(vocabulary_size=2,state_dim=14 if self.command_history else 7).to(self.device)
        self.mean=torch.tensor((LOWER+UPPER)/2,dtype=torch.float32,device=self.device)
        self.scale=torch.tensor((UPPER-LOWER)/2,dtype=torch.float32,device=self.device)
        self.delta_scale=torch.tensor(4.0*STEP_LENGTH,dtype=torch.float32,device=self.device)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
        self.updates=0

    @property
    def input_contract(self):
        return dict(version='w8_own_last_command_v1',state_dim=14,
            state_order=['measured_joint1..6','measured_total_opening',
                         'last_command_joint1..6','last_command_total_opening'],
            source='actual last issued command before this decision',privileged=False)

    @property
    def output_contract(self):
        if self.relative_delta:
            return dict(version='w8_measured_q_delta_4step_v1',anchor='measured_proprio_first7',
                        delta_scale=(4.0*STEP_LENGTH).tolist())
        return None

    def training_state(self,proprio,last_command):
        if not self.command_history:return proprio
        q=proprio.detach().cpu().numpy() if isinstance(proprio,torch.Tensor) else np.asarray(proprio,dtype=np.float32)
        last=last_command.detach().cpu().numpy() if isinstance(last_command,torch.Tensor) else np.asarray(last_command)
        if q.ndim!=2 or q.shape[1]!=7 or last.shape!=q.shape:
            raise ValueError('Measured state and actual last command must each be [N,7]')
        if not np.isfinite(q).all() or not np.isfinite(last).all():
            raise FloatingPointError('Nonfinite command-history observation')
        return np.concatenate((q,last.astype(np.float32)),axis=1)

    def prediction(self,images,proprio):
        actor_inputs(images,proprio,state_dim=14 if self.command_history else 7)
        images=torch.as_tensor(images,device=self.device,dtype=torch.float32)
        proprio=torch.as_tensor(proprio,device=self.device,dtype=torch.float32)
        tokens=torch.ones((len(proprio),1),device=self.device,dtype=torch.long)
        mean=torch.cat((self.mean,self.mean)) if self.command_history else self.mean
        scale=torch.cat((self.scale,self.scale)) if self.command_history else self.scale
        output=self.model((proprio-mean)/scale,tokens,images)
        if self.relative_delta:
            return proprio[:,None,:7]+output*self.delta_scale
        return output*self.scale+self.mean

    @torch.no_grad()
    def act(self,images,proprio,last_command,microbatch=32):
        self.model.eval()
        if self.command_history:proprio=self.training_state(proprio,last_command)
        chunks=[]
        for i in range(0,len(proprio),microbatch):
            chunks.append(self.prediction(images[i:i+microbatch],proprio[i:i+microbatch]))
        desired=torch.cat(chunks);previous=torch.as_tensor(last_command,device=self.device,dtype=torch.float32)
        step=torch.as_tensor(STEP_LENGTH,device=self.device,dtype=torch.float32)
        low=self.mean-self.scale;high=self.mean+self.scale;result=[]
        for j in range(8):
            delta=desired[:,j]-previous
            delta=delta/torch.maximum(torch.ones_like(delta[:,:1]),(delta.abs()/step).amax(1,keepdim=True))
            previous=torch.maximum(low,torch.minimum(high,previous+delta));result.append(previous)
        return torch.stack(result,1)

    def update(self,reservoir,batch=64,microbatch=8):
        if batch%microbatch:raise ValueError('Microbatch must divide effective batch')
        samples=reservoir.sample(batch);self.model.train();self.optimizer.zero_grad(set_to_none=True);total=0.
        loss_scale=self.delta_scale if self.relative_delta else self.scale
        for i in range(0,batch,microbatch):
            prediction=self.prediction(samples['images'][i:i+microbatch],samples['proprio'][i:i+microbatch])
            target=torch.as_tensor(samples['label'][i:i+microbatch],device=self.device)
            loss=(((prediction-target)/loss_scale)**2).mean()*(microbatch/batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DAgger loss')
            loss.backward();total+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite DAgger gradient')
        self.optimizer.step();self.updates+=1;self.model.eval()
        return total

    def save(self,path,metadata,guard=None):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        stream=io.BytesIO()
        normalization=dict(mean=self.mean.cpu(),scale=self.scale.cpu())
        output_fields={}
        if self.relative_delta:
            normalization['delta_scale']=self.delta_scale.cpu()
            output_fields=dict(relative_delta=True,output_contract=self.output_contract)
        torch.save(dict(format=self._formats[self.command_history,self.relative_delta],model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            updates=self.updates,model_config=self.model.config,metadata=metadata,
            input_keys=['images','proprio','last_command'] if self.command_history else ['images','proprio'],
            **(dict(input_contract=self.input_contract) if self.command_history else {}),native_state_names=[f'joint{i}' for i in range(1,7)]+['opening'],
            normalization=normalization,horizon=8,execute=4,**output_fields),stream)
        data=stream.getvalue()
        if guard is not None:guard.check(len(data))
        with path.open('xb') as file:
            file.write(data);file.flush();os.fsync(file.fileno())
        return len(data)

    def restore(self,path):
        payload=torch.load(path,map_location=self.device,weights_only=False)
        return self._restore_payload(payload)

    def _restore_payload(self,payload):
        expected=self._formats[self.command_history,self.relative_delta]
        keys=['images','proprio','last_command'] if self.command_history else ['images','proprio']
        if payload['format']!=expected or payload['input_keys']!=keys:
            raise ValueError('Wrong student contract')
        self.validate_recovery_contract(payload)
        if payload['model_config']!=self.model.config:raise ValueError('Wrong student architecture')
        for name,value in [('mean',self.mean),('scale',self.scale)]:
            if not torch.equal(payload['normalization'][name].to(self.device),value):raise ValueError('Wrong normalization')
        self.model.load_state_dict(payload['model']);self.optimizer.load_state_dict(payload['optimizer'])
        self.updates=payload['updates'];self.model.eval()
        return payload['metadata']

    def validate_recovery_contract(self,source):
        if self.command_history and source.get('input_contract')!=self.input_contract:
            raise ValueError('Command-history input contract differs or is missing')
        if not self.command_history and source.get('input_contract'):
            raise ValueError('Command-history checkpoint cannot load as legacy student')

        delta_scales=[fields['delta_scale'] for fields in (source,source.get('normalization',{}))
                      if 'delta_scale' in fields]
        if self.relative_delta:
            if source.get('relative_delta') is not True or source.get('output_contract')!=self.output_contract:
                raise ValueError('Relative-delta output contract differs or is missing')
            if len(delta_scales)!=1 or not isinstance(delta_scales[0],torch.Tensor):
                raise ValueError('Relative-delta normalization is missing or ambiguous')
            delta_scale=delta_scales[0]
            if delta_scale.dtype!=self.delta_scale.dtype or not torch.equal(delta_scale.to(self.device),self.delta_scale):
                raise ValueError('Relative-delta normalization differs')
        elif source.get('relative_delta',False) is not False or 'output_contract' in source or delta_scales:
            raise ValueError('Relative-delta checkpoint cannot load as absolute student')

    @classmethod
    def from_checkpoint(cls,path,device='cpu',seed=0):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        for (history,relative_delta),format_name in cls._formats.items():
            if payload.get('format')==format_name:
                student=cls(device,seed,command_history=history,relative_delta=relative_delta)
                student._restore_payload(payload)
                return student
        raise ValueError('Wrong student contract')
