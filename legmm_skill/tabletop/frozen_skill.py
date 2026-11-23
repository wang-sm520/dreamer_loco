"""W10 callable actor boundary. No scene, target, teacher or critic parameter."""
import hashlib,json,copy
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from airbot_rl.ppo import TRAIN_CFG
from .spec import STOW,WORKTREE
from .student import Student
from .residual_ppo import FrozenStudentFeatures,residual_commands


class FrozenW8Skill:
    def __init__(self,manifest,device='cpu'):
        path=Path(manifest);self.manifest=json.loads(path.read_text());self.device=torch.device(device)
        if self.manifest['format']!='w8_frozen_skill_v1':raise ValueError('Unknown frozen skill format')
        import importlib.metadata
        for package,version in self.manifest.get('dependencies',{}).items():
            if importlib.metadata.version(package)!=version:raise ValueError('Frozen dependency version differs: '+package)
        for relative,digest in self.manifest['source_sha256'].items():
            if hashlib.sha256((WORKTREE/'legmm_skill'/relative).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen inference source changed: '+relative)
        for name,record in self.manifest['weights'].items():
            source=path.parent/record['file']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Frozen weight changed: '+name)
        payload=torch.load(path.parent/self.manifest['weights']['student']['file'],map_location=self.device,weights_only=False)
        format_name=payload.get('format','w8_frozen_student_v1')
        if format_name not in ('w8_frozen_student_v1','w8_frozen_reldelta_student_v1'):
            raise ValueError('Unknown frozen student format')
        self.student=Student(device,0,command_history=bool(payload.get('input_contract')),
                             relative_delta=format_name=='w8_frozen_reldelta_student_v1')
        self.student.validate_recovery_contract(payload)
        if payload['model_config']!=self.student.model.config:raise ValueError('Frozen student dimensions differ')
        self.student.model.load_state_dict(payload['model'])
        for name,value in [('mean',self.student.mean),('scale',self.student.scale)]:
            if not torch.equal(payload['normalization'][name].to(device),value):raise ValueError('Frozen normalization differs')
        self.features=FrozenStudentFeatures(self.student)
        self.actor=None;self.last_command=None;self.depth_transform=None
        if 'residual' in self.manifest['weights']:
            obs=TensorDict(dict(policy=torch.zeros(1,547,device=self.device)),batch_size=[1])
            cfg=copy.deepcopy({k:v for k,v in TRAIN_CFG['actor'].items() if k!='class_name'})
            self.actor=MLPModel(obs,TRAIN_CFG['obs_groups'],'actor',28,**cfg).to(self.device)
            residual=torch.load(path.parent/self.manifest['weights']['residual']['file'],map_location=self.device,weights_only=False)
            self.actor.load_state_dict(residual['actor_state_dict']);self.actor.eval().requires_grad_(False)

    def reset(self,batch_size,last_command=None):
        initial=torch.tensor([*STOW,.072],device=self.device,dtype=torch.float32).repeat(batch_size,1)
        if last_command is not None:
            value=torch.as_tensor(last_command,device=self.device,dtype=torch.float32)
            if value.shape!=initial.shape or not torch.isfinite(value).all():raise ValueError('Invalid initial native command')
            initial=value.clone()
        self.last_command=initial

    @torch.no_grad()
    def act_depth(self,images,proprio):
        """Return four native 25Hz targets, queried every .16s after reset.

        images [N,2,2,128,128] = [head,wrist] x [depth,validity]. Native
        proprio [N,7] = joint1..6 radians, total finger opening metres.
        Caller retains the physical100ms delay, PD and standing controller.
        """
        if self.last_command is None or len(self.last_command)!=len(proprio):raise RuntimeError('reset batch before acting')
        policy,base=self.features.observe(images,proprio,self.last_command)
        if self.actor is None:commands=base
        else:
            obs=TensorDict(policy,batch_size=[len(proprio)])
            commands=residual_commands(base,self.actor(obs),self.last_command)
        if not torch.isfinite(commands).all():raise FloatingPointError('Nonfinite frozen action')
        self.last_command=commands[:,-1].clone()
        return commands

    @torch.no_grad()
    def act_raw(self,head_rgb,wrist_depth_metres,proprio):
        """Source raw canvas RGB uint8 and distance_to_image_plane in metres."""
        from airbot_depth.depth import DepthAnythingTransform,DEFAULT_DEPTH_CONFIG
        from airbot_rl.scene import head_image,wrist_depth
        from airbot_rl.policy import head_depth
        from airbot_rl.depth import encode_wrist_depth
        head=torch.as_tensor(head_rgb,device=self.device)[...,:3]
        wrist=torch.as_tensor(wrist_depth_metres,device=self.device,dtype=torch.float32)
        hc=self.manifest['visual']['head']['canvas'];wc=self.manifest['visual']['wrist']['canvas']
        if head.dtype!=torch.uint8 or tuple(head.shape[1:])!=(hc[1],hc[0],3):raise ValueError('Raw head canvas/dtype differs from frozen camera')
        if tuple(wrist.shape[1:]) not in ((wc[1],wc[0]),(wc[1],wc[0],1)):raise ValueError('Raw wrist canvas differs from frozen camera')
        if wrist.ndim==3:wrist=wrist[...,None]
        if len(head)!=len(wrist) or len(head)!=len(proprio):raise ValueError('Camera/proprio batch differs')
        if self.depth_transform is None:
            candidate=DepthAnythingTransform(DEFAULT_DEPTH_CONFIG,device=self.device,local_files_only=True)
            actual=candidate.provenance();expected=self.manifest['visual']['depth_provenance']
            if actual['config']!=expected['config'] or actual['files_sha256']!=expected['files_sha256']:
                raise ValueError('Frozen DA2 configuration/weights differ')
            self.depth_transform=candidate
        rgb=head_image(head,torch.ones(len(head),device=self.device))
        images=torch.stack((head_depth(self.depth_transform,rgb,batch=8),encode_wrist_depth(wrist_depth(wrist))),1)
        return self.act_depth(images,proprio)
