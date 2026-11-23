"""Continuation of a completed DAgger budget using its selected frozen student; explicit qualification required before launch."""
import argparse,gc,json,os,time
from pathlib import Path
from types import SimpleNamespace
from .spec import OUTPUT,WORKTREE
from .coordinator import acknowledge


def episode_args(out,num_envs,seed):
    return SimpleNamespace(out=out,num_envs=num_envs,seed=seed,mode='teacher',device='cuda:0',
        sampling=None,trace_teacher=False,closed_loop_teacher=True,residual_velocity=True,
        audit_no_gravity_compensation=False,audit_w2_model=False,audit_yaw_zero=False,audit_payload=None,
        audit_w2_self_collision_off=False,audit_reset_noise=False,audit_w2_robot_material=False,
        audit_scene_solver_4_0=False,audit_scene_solver_unclamped=False)


def evaluate_fixed(session,app,out,qualification,record_root):
    """Use exactly the teacher's fixed static initial states; never train on them."""
    from .learning import EvaluationSession
    from .learning_episode import run_episode
    from .evaluate_teacher import interval
    from isaacsim.core.utils.stage import create_new_stage
    import torch
    q=json.loads(qualification.read_text());evaluation=EvaluationSession(session.student,session.visual)
    episodes=[];sources=sorted({row['result_path'] for row in q['episodes']})
    for source in sources:
        print((record_root/'COORDINATOR_UPDATES.md').read_text(),flush=True)
        parent=Path(source).parent;data=json.loads(Path(source).read_text())
        seed=data['seed'];batch=out/f'seed_{seed}';batch.mkdir(exist_ok=False)
        args=episode_args(batch,len(data['states']),seed)
        args.residual_velocity=False
        args.sampling=Path(json.loads((parent/'sampling_source.json').read_text())['path'])
        result=run_episode(args,app,evaluation)
        if result['safety_stop']:raise RuntimeError('Student evaluation standing safety stop')
        episodes.extend(result['episodes'])
        create_new_stage();gc.collect();torch.cuda.empty_cache()
    if sorted(x['seed'] for x in episodes)!=list(range(10000,10256)):raise RuntimeError('Student validation seed mismatch')
    k=sum(x['success'] for x in episodes)
    summary=dict(driver='student',success_count=k,num_episodes=256,wilson95=interval(k,256),
        student_threshold=.85,student_gate_passed=k>=218,episodes=episodes)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-student',type=Path,required=True);p.add_argument('--resume-status',type=Path,required=True);p.add_argument('--start-round',type=int,default=6);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--record-root',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=32);p.add_argument('--seed',type=int,default=256)
    p.add_argument('--capacity-only',action='store_true');p.add_argument('--visual-diagnostic',action='store_true');p.add_argument('--standing-audit',choices=['none','no-render','render'],default='none');p.add_argument('--sampling',type=Path);p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--buffer',type=int,default=10000)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;a.enable_cameras=True
    previous=json.loads(a.resume_status.read_text())
    if not previous.get('completed') or previous.get('phase')!='dagger_budget_complete':raise ValueError('Complete previous budget before continuation')
    if previous.get('student_gate_passed'):raise ValueError('Qualified student should proceed to PPO')
    if Path(previous['best_checkpoint']).resolve()!=a.resume_student.resolve():raise ValueError('Resume the validation-selected student')
    q=json.loads(a.qualification.read_text())
    if (not q.get('teacher_gate_passed') and not a.capacity_only) or q.get('num_episodes')!=256 or q.get('teacher_kind')!='measured_state_feedback':
        raise ValueError('Full qualified production closed-loop teacher required')
    import hashlib
    critical={'geometry.py','closed_loop.py','control.py','spec.py','outcome.py','collision.py',
        'isaac_scene.py','physics_config.py','assets.py','physics_audit.py'}
    for parent in {Path(row['result_path']).parent for row in q['episodes']}:
        from .physics_config import verify_task_physics_audit
        verify_task_physics_audit(json.loads((parent/'physics_audit.json').read_text()))
        inventory={Path(x['path']).name:x['sha256'] for x in json.loads((parent/'source_inventory.json').read_text())}
        for name in critical:
            actual=hashlib.sha256((WORKTREE/'legmm_skill/tabletop'/name).read_bytes()).hexdigest()
            if not a.capacity_only and inventory.get(name)!=actual:raise ValueError('Teacher qualification source differs: '+name)
    app=AppLauncher(a).app
    from isaacsim.core.utils.stage import create_new_stage
    from .learning import DaggerSession
    from .learning_episode import run_episode
    from .storage import StorageGuard
    import torch
    from .learning import StandingAuditSession
    session=StandingAuditSession(a.standing_audit=='render') if a.standing_audit!='none' else DaggerSession('cuda:0',a.seed,a.buffer,diagnostic=a.visual_diagnostic)
    session.student.restore(a.resume_student)
    resumed_updates=session.student.updates
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    status=dict(PID=os.getpid(),phase='standing_render_audit' if a.standing_audit!='none' else 'visual_diagnostic' if a.visual_diagnostic else 'capacity_pipeline' if a.capacity_only else 'dagger',completed=False,
        formal_training=not (a.capacity_only or a.visual_diagnostic or a.standing_audit!='none'),dagger_started=not (a.capacity_only or a.visual_diagnostic or a.standing_audit!='none'),ppo_started=False,
        teacher_gate_passed=bool(q.get('teacher_gate_passed')),diagnostic_only=a.capacity_only or a.visual_diagnostic or a.standing_audit!='none',
        num_envs=a.num_envs,seed=a.seed,env_decisions=0,optimizer_updates=session.student.updates,checkpoint=str(a.resume_student),
        resumed_from=str(a.resume_status),cumulative_previous_env_decisions=previous.get('cumulative_env_decisions',previous['env_decisions']),
        previous_executed_optimizer_updates=previous.get('cumulative_executed_optimizer_updates',previous['optimizer_updates']),
        best_checkpoint=previous['best_checkpoint'],best_validation_success=previous['best_validation_success'],student_gate_passed=False,
        total_env_decision_budget=94*a.num_envs if a.visual_diagnostic or a.standing_audit!='none' else 2*a.num_envs if a.capacity_only else 192000,effective_batch=64,
        gradient_microbatch=8,learning_rate=1e-4,capacity_target=4096,rounds=[])
    def save():
        status.update(cumulative_env_decisions=status['cumulative_previous_env_decisions']+session.seen,
            cumulative_executed_optimizer_updates=status['previous_executed_optimizer_updates']+session.student.updates-resumed_updates)
        guard.check();(a.out/'training_status.json').write_text(json.dumps(status,indent=2)+'\n')
        for fn in ['progress.json','attention.json']:(a.record_root/fn).write_text(json.dumps(status,indent=2)+'\n')
    save();episode=0;seed=a.seed
    try:
        for round_index in range(1 if a.capacity_only or a.visual_diagnostic or a.standing_audit!='none' else a.rounds):
            acknowledge(a.record_root,'dagger_continuation',dagger_started=True,ppo_started=False)
            session.round=a.start_round+round_index;session.teacher_drives=False
            session.budget=session.seen+(94*a.num_envs if a.visual_diagnostic or a.standing_audit!='none' else 2*a.num_envs if a.capacity_only else 32000)
            while not session.exhausted:
                out=a.out/f'round_{a.start_round+round_index:02d}_episode_{episode:04d}';out.mkdir(exist_ok=False)
                args=episode_args(out,a.num_envs,seed)
                if a.sampling:args.sampling=a.sampling
                result=run_episode(args,app,session)
                if result['safety_stop']:raise RuntimeError('Standing stability safety stop')
                status.update(env_decisions=session.seen,physical_env_decisions=session.physical_env_decisions,
                    collected_samples=session.collected_samples,current_episode=str(out),
                    actual_envs=result['num_envs'],physical_steps=result['physical_steps'])
                save();seed+=a.num_envs;episode+=1
                if a.visual_diagnostic or a.standing_audit!='none':session.budget=session.seen
                if seed+a.num_envs>10000:seed=256
                create_new_stage();gc.collect();torch.cuda.empty_cache()
            if a.visual_diagnostic or a.standing_audit!='none':
                status.update(completed=True,phase='standing_audit_complete' if a.standing_audit!='none' else 'visual_diagnostic_complete',visual_correctness_pending=True);save();return
            losses=[];update_start=time.monotonic()
            for update in range(1 if a.capacity_only else a.updates):
                losses.append(session.student.update(session.reservoir,batch=64,microbatch=8))
                if update%100==0:
                    status.update(optimizer_updates=session.student.updates,phase='capacity_update' if a.capacity_only else 'dagger_update');save()
            record=dict(update_elapsed_s=time.monotonic()-update_start,round=a.start_round+round_index,teacher_drives=session.teacher_drives,env_decisions=session.seen,
                optimizer_updates=session.student.updates,mean_loss=sum(losses)/len(losses),
                retained=len(session.reservoir.items),reservoir_bytes=session.reservoir.tensor_bytes)
            status['rounds'].append(record)
            if a.capacity_only or (round_index+1)%3==0:
                checkpoint=a.out/('capacity_restore_test.pt' if a.capacity_only else f'round_{a.start_round+round_index+1:02d}.pt')
                session.student.save(checkpoint,dict(visual=session.visual.manifest(),qualification=str(a.qualification),
                    physics='scene_all_actors_4_0',round=a.start_round+round_index,env_decisions=session.seen),guard=guard)
                status['checkpoint']=str(checkpoint)
                if a.capacity_only:session.student.restore(checkpoint)
                else:
                    status.update(phase='student_fixed_validation');save()
                    validation=a.out/f'validation_round_{a.start_round+round_index+1:02d}';validation.mkdir(exist_ok=False)
                    summary=evaluate_fixed(session,app,validation,a.qualification,a.record_root)
                    record['validation_success']=summary['success_count']
                    record['student_gate_passed']=summary['student_gate_passed']
                    if summary['success_count']>status.get('best_validation_success',-1):
                        status.update(best_validation_success=summary['success_count'],best_checkpoint=str(checkpoint),
                            student_gate_passed=summary['student_gate_passed'])
            status.update(optimizer_updates=session.student.updates,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
        status.update(completed=True,phase='capacity_complete' if a.capacity_only else 'dagger_budget_complete');save()
    except BaseException as error:
        status.update(reason=repr(error),completed=False);save();raise
    finally:app.close(wait_for_replicator=False)


if __name__=='__main__':main()
