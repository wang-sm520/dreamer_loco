"""DAgger session hooks for a true-scene episode; no privileged actor channels."""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .student import Student
from .data import Reservoir, SplitReservoir
from .visual import DualDepth
from .segment_timing import timer_for
from .spec import PHYSICS_DT


def arm_step4_statistics(actions,labels,last_command,active):
    previous=last_command[active,:6]
    teacher=labels[active,3,:6]-previous
    teacher_norm=np.linalg.norm(teacher,axis=1)
    eligible=teacher_norm>1e-4
    count=int(eligible.sum())
    if not count:
        return dict(arm_step4_eligible=count,arm_step4_amp_ratio_median=None,arm_step4_cos_median=None)
    student=(actions[active,3,:6]-previous)[eligible]
    teacher=teacher[eligible]
    teacher_norm=teacher_norm[eligible]
    student_norm=np.linalg.norm(student,axis=1)
    cosine=np.sum(student*teacher,axis=1)/(student_norm*teacher_norm+1e-12)
    return dict(arm_step4_eligible=count,arm_step4_amp_ratio_median=float(np.median(student_norm/teacher_norm)),
                arm_step4_cos_median=float(np.median(cosine)))


class DaggerSession:
    supports_interrupted_collection=True
    def __init__(self,device,seed,capacity=10000,diagnostic=False,command_history=False,relative_delta=False,stateless_dagger=False):
        self.diagnostic=diagnostic
        self.student=Student(device,seed,command_history=command_history,relative_delta=relative_delta)
        self.visual=DualDepth(device)
        self.stateless_dagger=stateless_dagger
        state_dim=14 if command_history else 7
        if stateless_dagger:
            from .stateless_teacher import EpisodeLabels
            self.episode_labels=EpisodeLabels()
            self.reservoir=SplitReservoir(seed,state_dim)
        else:
            self.reservoir=Reservoir(capacity,seed,state_dim=state_dim)
        self.round=0;self.teacher_drives=True;self.budget=0;self.seen=0
        self.episode_serial=0;self.update_records=[];self.checkpoint=None
        self.physical_env_decisions=0;self.collected_samples=0
        self.shadow_student=False

    @property
    def exhausted(self):return self.seen>=self.budget

    @property
    def retained(self):
        return self.reservoir.retained if self.stateless_dagger else len(self.reservoir.items)

    @property
    def collection_pool(self):
        return self.reservoir.for_round(self.round) if self.stateless_dagger else self.reservoir

    def _select_supervision(self, labels, active, frame):
        if not self.stateless_dagger:
            return labels, active
        supervision = labels if self.teacher_drives else labels.copy()
        valid = []
        for index in active:
            teacher, measured, metadata = self.episode_labels.take(self.states[index].seed, frame * PHYSICS_DT)
            if not self.teacher_drives:
                supervision[index], metadata = teacher.stateless.label(
                    measured, initial_gate_reason=teacher.initial_gate_reason)
                self.stateless_phase_counts[metadata['phase']] += 1
            if metadata['reason'] is None:
                valid.append(index)
                self.valid_phase_counts[metadata['phase']] += 1
            else:
                self.filtered_reasons[metadata['reason']] += 1
        return supervision, valid

    def reset(self,states,scene,simulation,out):
        self.states=states;self.out=Path(out);self.episode_serial+=1
        self.episode_counts={s.seed:0 for s in states};self.started=time.monotonic()
        (self.out/'visual_manifest.json').write_text(json.dumps(self.visual.manifest(),indent=2))
        if self.stateless_dagger:
            self.stateless_phase_counts=Counter()
            self.valid_phase_counts=Counter()
            self.filtered_reasons=Counter()
        self.decision_log=(self.out/'learning_decisions.jsonl').open('x')
        if self.diagnostic:
            from .visual_diagnostics import VisualDiagnostics
            self.visual_diagnostics=VisualDiagnostics(out,states)

    def decision(self,scene,robot,joints,arm_ids,finger_ids,last_command,labels,outcomes,frame,phases=None):
        timings=timer_for(self)
        images,rgb,wrist=timings.call('visual_observe_total',self.visual.observe,scene)
        if self.diagnostic:self.visual_diagnostics.record(scene,images,rgb,wrist,phases,frame)
        proprio=np.column_stack((joints[:,arm_ids],joints[:,finger_ids[0]]-joints[:,finger_ids[1]]))
        # CPU reservoir stores only rounded dual depth, native proprio and labels.
        # Terminal failures are retained; only post-terminal duplicate frames stop.
        remaining=max(0,self.budget-self.seen)
        active=[i for i,o in enumerate(outcomes) if o.reason is None and i<remaining]
        self.physical_env_decisions+=len(proprio)
        self.seen+=min(remaining,len(proprio))
        stored_proprio=self.student.training_state(proprio,last_command)
        if self.student.command_history and frame in (0,16):
            with (self.out/'command_history_input_audit.jsonl').open('a') as audit:
                audit.write(json.dumps(dict(frame=frame,seeds=[s.seed for s in self.states[:4]],
                    measured_proprio=proprio[:4].tolist(),last_command=stored_proprio[:4,7:].tolist(),
                    stored_actor_state=stored_proprio[:4].tolist(),state_dim=14,
                    source='actual last issued command before decision'))+'\n')
        supervision, valid = self._select_supervision(labels, active, frame)
        for i in ([] if self.diagnostic else valid):
            state=self.states[i]
            self.collection_pool.add(images[i].cpu().numpy(),stored_proprio[i],supervision[i],
                dict(shape=state.shape,outcome='pending',round=self.round,seed=state.seed,
                     episode=self.episode_serial,frame=frame))
            self.collected_samples+=1;self.episode_counts[state.seed]+=1
        _timing_forward=timings.begin('student_forward')
        if self.teacher_drives:
            if self.shadow_student:self.student.act(images,proprio,last_command)
            actions=labels
        else:actions=self.student.act(images,proprio,last_command).cpu().numpy()
        timings.end('student_forward',_timing_forward)
        _timing_io=timings.begin('decision_io')
        # At least one real backward/update is part of a capacity short test.
        record=dict(frame=frame,actual_envs=len(proprio),images_shape=list(images.shape),
            head_rgb_shape=list(rgb.shape),wrist_depth_shape=list(wrist.shape),
            reservoir_frames=self.retained,reservoir_bytes=self.reservoir.tensor_bytes,
            env_decisions=self.seen,physical_env_decisions=self.physical_env_decisions,collected_samples=self.collected_samples,teacher_drives=self.teacher_drives,shadow_student_forward=self.shadow_student)
        if not self.teacher_drives:
            previous=stored_proprio[:,7:] if self.student.command_history else last_command
            if isinstance(previous,torch.Tensor):previous=previous.detach().cpu().numpy()
            record.update(arm_step4_statistics(actions,supervision,previous,valid))
        self.decision_log.write(json.dumps(record)+'\n');self.decision_log.flush()
        if frame==0:
            from PIL import Image
            Image.fromarray(rgb[0].cpu().numpy()).save(self.out/'head_preview.png')
            Image.fromarray((images[0,1,0].cpu().numpy()*255).astype(np.uint8)).save(self.out/'wrist_preview.png')
        timings.end('decision_io',_timing_io)
        return actions

    def finish(self,outcomes,interrupted=False):
        verdicts={s.seed:o.reason or ('collection_interrupted' if interrupted else 'collection_budget_truncated') for s,o in zip(self.states,outcomes)}
        pool=self.collection_pool
        for _,_,_,metadata in pool.items:
            if metadata['episode']==self.episode_serial:metadata['outcome']=verdicts[metadata['seed']]
        # Composition counts describe all seen samples, not just reservoir survivors.
        for s in self.states:
            count=self.episode_counts[s.seed]
            pool.composition[(s.shape,'pending',self.round)]-=count
            pool.composition[(s.shape,verdicts[s.seed],self.round)]+=count
        self.decision_log.close()
        extra = {}
        pools = [pool]
        if self.stateless_dagger:
            pools = [self.reservoir.demo, self.reservoir.dagger]
            extra = dict(stateless_phase_counts=dict(self.stateless_phase_counts),
                         valid_phase_counts=dict(self.valid_phase_counts),
                         filtered_reasons=dict(self.filtered_reasons),
                         demo_retained=len(pools[0].items), dagger_retained=len(pools[1].items))
        (self.out/'collection.json').write_text(json.dumps(dict(env_decisions=self.seen,round=self.round,interrupted=interrupted,
            physical_env_decisions=self.physical_env_decisions,collected_samples=self.collected_samples,
            elapsed_s=time.monotonic()-self.started,retained=self.retained,
            reservoir_bytes=self.reservoir.tensor_bytes,composition=[dict(shape=k[0],outcome=k[1],round=k[2],count=v)
                for pool in pools for k,v in pool.composition.items() if v],**extra),indent=2))


class EvaluationSession:
    """Student rollout without adding validation frames to the DAgger reservoir."""
    exhausted=False

    def __init__(self,student,visual):
        self.student,self.visual=student,visual

    def reset(self,states,scene,simulation,out):
        self.states=states;self.out=Path(out)
        (self.out/'visual_manifest.json').write_text(json.dumps(self.visual.manifest(),indent=2))
        self.started=time.monotonic();self.decisions=0

    def decision(self,scene,robot,joints,arm_ids,finger_ids,last_command,labels,outcomes,frame,phases=None):
        timings=timer_for(self)
        images,_,_=timings.call('visual_observe_total',self.visual.observe,scene)
        proprio=np.column_stack((joints[:,arm_ids],joints[:,finger_ids[0]]-joints[:,finger_ids[1]]))
        self.decisions+=len(proprio)
        return timings.call('student_forward',lambda:self.student.act(images,proprio,last_command).cpu().numpy())

    def finish(self,outcomes):
        (self.out/'student_evaluation.json').write_text(json.dumps(dict(env_decisions=self.decisions,
            total=len(outcomes),success=sum(o.success for o in outcomes),elapsed_s=time.monotonic()-self.started),indent=2))


class StandingAuditSession:
    diagnostic=False
    teacher_drives=False
    driver_name='standing_hold'

    def __init__(self,render_enabled):
        self.render_enabled=render_enabled;self.seen=0;self.budget=0
        self.physical_env_decisions=0;self.collected_samples=0

    @property
    def exhausted(self):return self.seen>=self.budget

    def reset(self,states,scene,simulation,out):
        self.out=Path(out);self.states=states

    def decision(self,scene,robot,joints,arm_ids,finger_ids,last_command,labels,outcomes,frame,phases=None):
        self.seen+=len(outcomes);self.physical_env_decisions+=len(outcomes)
        return np.repeat(last_command.cpu().numpy()[:,None,:],8,axis=1)

    def finish(self,outcomes):
        (self.out/'standing_audit.json').write_text(json.dumps(dict(render_enabled=self.render_enabled,
            env_decisions=self.seen,formal_training=False),indent=2))
