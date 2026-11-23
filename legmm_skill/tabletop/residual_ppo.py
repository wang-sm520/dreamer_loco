"""W8 residual PPO building blocks using the installed source TimePPO/M6 config.

Post-terminal cohort padding is masked from optimization and normalization.
Actor inputs are frozen student features/proprio/base actions; privileged state
is confined to the critic key. Formal launch still requires student validation.
"""
import math,copy,weakref
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from airbot_rl.ppo import TimePPO,TRAIN_CFG


class ActiveMLP(MLPModel):
    active_rows=None

    def update_normalization(self,obs):
        selected=obs if self.active_rows is None else obs[self.active_rows]
        if selected.batch_size[0]:super().update_normalization(selected)


class ValidRollout(RolloutStorage):
    def __init__(self,num_envs,obs,device,batch_size=1024):
        # A cohort can have just one surviving row. Keep enough temporal slots
        # for a complete effective batch without dropping difficult episodes.
        self.capacity=max(batch_size,math.ceil(batch_size/num_envs)+96)
        super().__init__('rl',num_envs,self.capacity,obs,(28,),device=device)
        self.valid=torch.zeros(self.capacity,num_envs,dtype=torch.bool,device=device)
        self.effective_batch=batch_size

    def clear(self):
        super().clear();self.valid.zero_();self.num_transitions_per_env=self.capacity

    def selected_indices(self):
        indices=self.valid[:self.step].flatten().nonzero().flatten()
        if len(indices)<self.effective_batch:raise ValueError('Incomplete valid PPO rollout')
        return indices[:self.effective_batch]

    def mini_batch_generator(self,num_mini_batches,num_epochs=8):
        indices=self.selected_indices();batch=len(indices)
        if batch%num_mini_batches:raise ValueError('Effective PPO batch must divide minibatches')
        obs=self.observations[:self.step].reshape(-1)
        flat=lambda value:value[:self.step].flatten(0,1)
        for _ in range(num_epochs):
            order=indices[torch.randperm(batch,device=self.device)]
            for part in order.split(batch//num_mini_batches):
                yield self.Batch(observations=obs[part],actions=flat(self.actions)[part],
                    values=flat(self.values)[part],advantages=flat(self.advantages)[part],returns=flat(self.returns)[part],
                    old_actions_log_prob=flat(self.actions_log_prob)[part],
                    old_distribution_params=tuple(flat(v)[part] for v in self.distribution_params),
                    dones=flat(self.dones)[part])


class ValidTimePPO(TimePPO):
    def set_active(self,mask):
        self.active=mask.to(device=self.device,dtype=torch.bool).clone()

    def process_env_step(self,obs,rewards,dones,extras):
        self.storage.valid[self.storage.step]=self.active
        self.actor.active_rows=self.active;self.critic.active_rows=self.active
        super().process_env_step(obs,rewards,dones,extras)

    def compute_returns(self,obs):
        st=self.storage;st.num_transitions_per_env=st.step
        # Reuse the source time-discounted GAE. Normalize only valid samples.
        normalizer=self.normalize_advantage_per_mini_batch
        self.normalize_advantage_per_mini_batch=True
        super().compute_returns(obs)
        self.normalize_advantage_per_mini_batch=normalizer
        indices=st.selected_indices();valid=st.advantages[:st.step].flatten(0,1)[indices]
        if not normalizer:st.advantages=(st.advantages-valid.mean())/(valid.std()+1e-8)


def build_residual_ppo(observations,device='cpu'):
    obs=TensorDict(observations,batch_size=[len(observations['policy'])],device=device)
    groups=TRAIN_CFG['obs_groups']
    # rsl_rl resolves distribution_cfg by pop(); nested source config must not
    # be consumed by the first model and break later evaluation/restore.
    actor_cfg=copy.deepcopy({k:v for k,v in TRAIN_CFG['actor'].items() if k!='class_name'})
    critic_cfg=copy.deepcopy({k:v for k,v in TRAIN_CFG['critic'].items() if k!='class_name'})
    actor=ActiveMLP(obs,groups,'actor',28,**actor_cfg).to(device)
    critic=ActiveMLP(obs,groups,'critic',1,**critic_cfg).to(device)
    mean=[module for module in actor.modules() if isinstance(module,torch.nn.Linear)][-1]
    torch.nn.init.zeros_(mean.weight);torch.nn.init.zeros_(mean.bias)
    storage=ValidRollout(obs.batch_size[0],obs,device)
    config=copy.deepcopy({k:v for k,v in TRAIN_CFG['algorithm'].items() if k!='class_name'})
    algorithm=ValidTimePPO(actor,critic,storage,device=device,**config)
    return algorithm


def residual_commands(base_chunk,residual,last_command):
    """Source residual scales with the same native speed/limit governor as BC."""
    from .spec import RESIDUAL_SCALE,STEP_LENGTH,LOWER,UPPER
    if base_chunk.shape[1:]!=(4,7) or residual.shape!=(len(base_chunk),28):raise ValueError('Invalid residual action contract')
    scale=torch.as_tensor(RESIDUAL_SCALE,device=base_chunk.device,dtype=base_chunk.dtype)
    step=torch.as_tensor(STEP_LENGTH,device=base_chunk.device,dtype=base_chunk.dtype)
    low=torch.as_tensor(LOWER,device=base_chunk.device,dtype=base_chunk.dtype)
    high=torch.as_tensor(UPPER,device=base_chunk.device,dtype=base_chunk.dtype)
    raw=base_chunk+scale*torch.tanh(residual.reshape(-1,4,7));previous=last_command;points=[]
    for j in range(4):
        delta=raw[:,j]-previous
        delta=delta/torch.maximum(torch.ones_like(delta[:,:1]),(delta.abs()/step).amax(1,keepdim=True))
        previous=torch.maximum(low,torch.minimum(high,previous+delta));points.append(previous)
    result=torch.stack(points,1)
    # Deterministic iteration zero must be the frozen base exactly, including
    # its last-bit rounding through the native governor.
    zero=residual.abs().amax(1)==0
    return torch.where(zero[:,None,None],base_chunk,result)


class FrozenStudentFeatures:
    def __init__(self,student):
        self.student=student;student.model.eval().requires_grad_(False)
        self.parts=[];self.collecting=False
        reference=weakref.ref(self)
        def collect(_,inputs):
            owner=reference()
            if owner is not None and owner.collecting:owner.parts.append(inputs[0].detach())
        self.handle=student.model.action_head.register_forward_pre_hook(collect)
        self.finalizer=weakref.finalize(self,self.handle.remove)

    def close(self):
        self.finalizer();self.parts=[]

    @torch.no_grad()
    def observe(self,images,proprio,last_command):
        self.parts=[]
        self.collecting=True
        try:chunk=self.student.act(images,proprio,last_command)
        finally:self.collecting=False
        features=torch.cat(self.parts)
        state=torch.as_tensor(proprio,device=self.student.device,dtype=torch.float32)
        normalized=(state-self.student.mean)/self.student.scale
        base=chunk[:,:4]
        return dict(policy=torch.cat((features,normalized,base.flatten(1)),1)),base
