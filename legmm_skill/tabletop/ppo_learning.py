"""Residual learning hooks. Reward/critic truth never enters the actor key.

Design choice: bounded lift/hold potential, +20 success, -5 failure (a further
-2 for forbidden contact), and .001 per nominal decision. Terminal potential
is zero. Verdicts and their time remain exclusively EpisodeOutcome's decision.
"""
import io,json,os,time
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .residual_ppo import FrozenStudentFeatures,build_residual_ppo,residual_commands


def transition_reward(previous,previous_steps,outcomes,gamma=.99):
    steps=np.asarray([o.steps for o in outcomes]);duration=(steps-previous_steps)/16
    if (duration<0).any():raise ValueError('Outcome clock moved backwards')
    done=np.asarray([o.reason is not None for o in outcomes])
    potential=np.asarray([min(max(o.peak_lift,0.)/.05,1.)+min(o.held_steps/200,1.) for o in outcomes])
    potential[done]=0.
    reward=gamma**duration*potential-previous-.001*duration
    reward+=np.asarray([20. if o.success else -5. if o.reason else 0. for o in outcomes])
    reward-=np.asarray([2. if (o.reason or '').startswith('collision:') else 0. for o in outcomes])
    return reward,duration,done,potential,steps


class ResidualSession:
    driver_name='residual_ppo'
    diagnostic=False
    teacher_drives=False
    # Episodes run to their actual terminal outcomes; update budgets never
    # truncate an in-flight 15 s cohort or turn truncation into success.
    exhausted=False

    def __init__(self,student,visual,train=True,on_update=None,actor_state=None):
        self.student=student;self.visual=visual;self.features=FrozenStudentFeatures(student)
        self.device=student.device;self.train=train;self.on_update=on_update
        self.algorithm=None;self.updates=0;self.valid_samples=0;self.physical_env_decisions=0
        self.unselected_samples=0;self.pending=None;self.last_obs=None
        self.update_records=[]
        self.actor_state=actor_state

    def reset(self,states,scene,simulation,out):
        if self.pending is not None:raise RuntimeError('Unfinished PPO transition across scene reset')
        self.states=states;self.out=Path(out);self.started=time.monotonic()
        self.latest_critic=None;self.previous=np.zeros(len(states));self.previous_steps=np.zeros(len(states))
        self.log=(self.out/'ppo_decisions.jsonl').open('x')
        (self.out/'visual_manifest.json').write_text(json.dumps(self.visual.manifest(),indent=2))

    def critic(self,scene,robot):
        target=scene['target'];origins=scene.env_origins
        constants=torch.tensor([[s.support,*s.size,s.mass,s.friction] for s in self.states],device=self.device)
        return torch.cat((robot.data.joint_pos,robot.data.joint_vel,
            robot.data.root_link_pos_w-origins,robot.data.root_link_quat_w,robot.data.root_vel_w,
            target.data.root_pos_w-origins,target.data.root_quat_w,target.data.root_vel_w,constants),1).detach()

    def after_step(self,scene,robot,outcomes,frame):
        # Read only: physics, delay queue and per-step verdict implementation are
        # unchanged in ppo_episode. No image, teacher label or state is mutated.
        self.latest_critic=self.critic(scene,robot)

    def _finish_transition(self,obs,outcomes):
        if self.pending is None:return
        active=self.pending
        reward,duration,done,potential,steps=transition_reward(self.previous,self.previous_steps,outcomes)
        algorithm=self.algorithm
        if self.train and bool(active.any()):
            algorithm.set_active(active)
            algorithm.process_env_step(obs,torch.as_tensor(reward,device=self.device,dtype=torch.float32),
                torch.as_tensor(done,device=self.device),
                dict(duration=torch.as_tensor(duration,device=self.device,dtype=torch.float32)))
            self.valid_samples+=int(active.sum())
            if int(algorithm.storage.valid[:algorithm.storage.step].sum())==algorithm.storage.effective_batch:
                algorithm.compute_returns(obs)
                losses=algorithm.update();self.updates+=1
                record=dict(update=self.updates,valid_samples=self.valid_samples,losses=losses)
                self.update_records.append(record)
                if self.on_update:self.on_update(record)
                from .lifecycle_cleanup import check_stop
                check_stop('completed PPO optimizer iteration')
        self.previous=potential;self.previous_steps=steps;self.pending=None

    @torch.no_grad()
    def _actor_observation(self,images,proprio,last_command,critic):
        policy,base=self.features.observe(images,proprio,last_command)
        return TensorDict(dict(policy=policy['policy'],critic=critic),batch_size=[len(proprio)]),base

    def decision(self,scene,robot,joints,arm_ids,finger_ids,last_command,labels,outcomes,frame,phases=None):
        images,_,_=self.visual.observe(scene)
        proprio=np.column_stack((joints[:,arm_ids],joints[:,finger_ids[0]]-joints[:,finger_ids[1]]))
        obs,base=self._actor_observation(images,proprio,last_command,self.critic(scene,robot))
        self._finish_transition(obs,outcomes)
        if self.algorithm is None:
            self.algorithm=build_residual_ppo(obs,self.device)
            if self.actor_state is not None:self.algorithm.actor.load_state_dict(self.actor_state)
        algorithm=self.algorithm
        active=torch.tensor([o.reason is None for o in outcomes],device=self.device)
        if self.train:
            remaining=algorithm.storage.effective_batch-int(algorithm.storage.valid[:algorithm.storage.step].sum())
            indices=active.nonzero().flatten();self.unselected_samples+=max(0,len(indices)-remaining)
            active[indices[remaining:]]=False
            # Source algorithm samples the Gaussian. Only valid rows enter PPO;
            # all other physical rows and original failures stay in metrics.
            with torch.no_grad():residual=algorithm.act(obs)
        else:
            with torch.no_grad():residual=algorithm.actor(obs)
        commands=residual_commands(base,residual,last_command)
        self.pending=active;self.last_obs=obs
        self.physical_env_decisions+=len(outcomes)
        self.log.write(json.dumps(dict(frame=frame,physical_env_decisions=self.physical_env_decisions,
            valid_samples=self.valid_samples,active_selected=int(active.sum()),
            unselected_samples=self.unselected_samples,optimizer_updates=self.updates,
            actor_keys=['frozen_dual_depth_features','proprio','frozen_base_chunk'],critic_privileged=True))+'\n')
        self.log.flush()
        # Native runtime expects horizon8; only the first4 are executed.
        return torch.cat((commands,commands[:,-1:].repeat(1,4,1)),1).cpu().numpy()

    def finish(self,outcomes,aborted=None):
        if any(o.reason is None for o in outcomes) and aborted is None:raise RuntimeError('PPO cohort ended before actual terminal verdict')
        if aborted is None:
            obs=self.last_obs.clone();obs['critic']=self.latest_critic
            self._finish_transition(obs,outcomes)
        else:
            # Preserve the true safety verdict in runtime result.json. Do not
            # invent outcomes or optimize a partially observed transition.
            self.pending=None
        self.log.close()
        (self.out/'ppo_collection.json').write_text(json.dumps(dict(updates=self.updates,
            valid_samples=self.valid_samples,physical_env_decisions=self.physical_env_decisions,
            unselected_samples=self.unselected_samples,success=sum(o.success for o in outcomes),
            total=len(outcomes),safety_stop=aborted,unterminated=sum(o.reason is None for o in outcomes),
            elapsed_s=time.monotonic()-self.started),indent=2))

    def save(self,path,metadata,guard):
        path=Path(path);algorithm=self.algorithm
        stream=io.BytesIO()
        torch.save(dict(format='w8_residual_ppo_v1',actor_state_dict=algorithm.actor.state_dict(),
            critic_state_dict=algorithm.critic.state_dict(),optimizer_state_dict=algorithm.optimizer.state_dict(),
            updates=self.updates,valid_samples=self.valid_samples,physical_env_decisions=self.physical_env_decisions,
            metadata=metadata),stream)
        data=stream.getvalue();guard.check(len(data))
        with path.open('xb') as file:file.write(data);file.flush();os.fsync(file.fileno())
        return len(data)
