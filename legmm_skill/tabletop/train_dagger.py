"""True A2 dual-vision DAgger runner; explicit qualification required before launch."""
import argparse,gc,json,os,time
from contextlib import nullcontext
from collections import Counter
from statistics import median
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


def evaluate_fixed(session,app,out,qualification,record_root,audit_timing=False,audit_batched_foot_readback=False,reuse_manifest=None,stateless_teacher_smoke=False):
    """Use exactly the teacher's fixed static initial states; never train on them."""
    from .learning import EvaluationSession
    from .learning_episode import run_episode
    from .evaluate_teacher import interval
    from isaacsim.core.utils.stage import create_new_stage
    import torch
    q=json.loads(qualification.read_text())
    if stateless_teacher_smoke:
        from .stateless_teacher import EpisodeLabels, StatelessTeacherEvaluation
        evaluation=StatelessTeacherEvaluation()
    else:
        evaluation=EvaluationSession(session.student,session.visual)
    episodes=[];sources=sorted({row['result_path'] for row in q['episodes']})
    if stateless_teacher_smoke:
        selected=[]
        for source in sources:
            cohort=json.loads(Path(source).read_text())
            if cohort['seed'] in (10000,10032):
                if len(cohort['states'])!=32:
                    raise ValueError('Teacher smoke requires two certified 32-environment cohorts')
                selected.append((cohort['seed'],source))
        selected.sort()
        if [seed for seed,_ in selected]!=[10000,10032]:
            raise ValueError('Teacher smoke requires fixed cohorts 10000 and 10032')
        sources=[source for _,source in selected]
    reused={}
    if reuse_manifest:
        from .validation_recovery import reusable_results
        reused=reusable_results(reuse_manifest,qualification)
        (out/'reused_batches.json').write_text(json.dumps(dict(manifest=str(reuse_manifest),batches=list(reused.values())),indent=2)+'\n')
    for source in sources:
        print((record_root/'COORDINATOR_UPDATES.md').read_text(),flush=True)
        parent=Path(source).parent;data=json.loads(Path(source).read_text())
        seed=data['seed']
        if seed in reused:
            prior=json.loads(Path(reused[seed]['result_path']).read_text())
            episodes.extend(prior['episodes']);continue
        batch=out/f'seed_{seed}';batch.mkdir(exist_ok=False)
        args=episode_args(batch,len(data['states']),seed)
        args.audit_timing=audit_timing
        args.audit_batched_foot_readback=audit_batched_foot_readback
        args.residual_velocity=False
        args.sampling=Path(json.loads((parent/'sampling_source.json').read_text())['path'])
        labels_scope=EpisodeLabels(drive=True).install() if stateless_teacher_smoke else nullcontext()
        with labels_scope:
            result=run_episode(args,app,evaluation)
        from .lifecycle_cleanup import check_stop
        if result.get('requested_safe_stop'):check_stop('validation action-block boundary')
        if result['safety_stop']:raise RuntimeError('Student evaluation standing safety stop')
        episodes.extend(result['episodes'])
        create_new_stage();gc.collect();torch.cuda.empty_cache()
    expected=64 if stateless_teacher_smoke else 256
    if sorted(x['seed'] for x in episodes)!=list(range(10000,10000+expected)):
        raise RuntimeError('Fixed validation seed mismatch')
    k=sum(x['success'] for x in episodes)
    if stateless_teacher_smoke:
        collisions=sum((row['reason'] or '').startswith('collision') for row in episodes)
        summary=dict(driver='stateless_teacher',success_count=k,num_episodes=expected,wilson95=interval(k,expected),
                     collision_count=collisions,teacher_smoke_passed=k>=54 and not collisions,episodes=episodes)
    else:
        summary=dict(driver='student',success_count=k,num_episodes=256,wilson95=interval(k,256),
            student_threshold=.85,student_gate_passed=k>=218,episodes=episodes)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def summarize_collection(out,round_index,stateless_dagger=False):
    """Count terminal outcomes only; a cutoff can leave some peers unresolved."""
    sources=sorted(out.glob(f'round_{round_index:02d}_episode_*'))
    reasons=Counter()
    success=unresolved=0
    movement=[]
    phases, valid_phases, filtered = Counter(), Counter(), Counter()
    for cohort in sources:
        result=json.loads((cohort/'result.json').read_text())
        for row in result['episodes']:
            if row['reason'] in (None,'collection_budget_truncated','collection_interrupted'):
                unresolved+=1
                continue
            success+=row['success']
            reasons[row['reason']]+=1
        if stateless_dagger:
            collection=json.loads((cohort/'collection.json').read_text())
            phases.update(collection['stateless_phase_counts'])
            valid_phases.update(collection['valid_phase_counts'])
            filtered.update(collection['filtered_reasons'])
        if round_index==1 or (stateless_dagger and round_index>0):
            with (cohort/'learning_decisions.jsonl').open() as decisions:
                for line in decisions:
                    record=json.loads(line)
                    if record['arm_step4_eligible']:movement.append(record)
    extra=dict(stateless_phase_counts=dict(phases),valid_phase_counts=dict(valid_phases),filtered_reasons=dict(filtered)) if stateless_dagger else {}
    return dict(success_count=success,completed_episodes=sum(reasons.values()),unresolved_episodes=unresolved,
        outcome_counts=dict(reasons),cohort_count=len(sources),decision_records=len(movement),
        eligible_env_decisions=sum(row['arm_step4_eligible'] for row in movement),
        arm_step4_amp_ratio_median=median(row['arm_step4_amp_ratio_median'] for row in movement) if movement else None,
        arm_step4_cos_median=median(row['arm_step4_cos_median'] for row in movement) if movement else None,**extra)


def write_round1_fast_gate(source,out,stateless_dagger=False):
    result=summarize_collection(source,1,stateless_dagger)
    if not result['cohort_count']:raise ValueError('Round 1 episode evidence missing')
    result.update(passed=result['success_count']>0,source=str(source),
                  movement_aggregation='median of eligible per-decision medians')
    if stateless_dagger:
        required=('clear','pre','grasp','close','lift')
        phases=result['valid_phase_counts']
        result.update(required_valid_phases=list(required),
                      missing_valid_phases=[phase for phase in required if not phases.get(phase)],
                      direction_passed=(result['arm_step4_cos_median'] is not None
                                        and result['arm_step4_cos_median']>.50))
        result['passed'] = result['passed'] and not result['missing_valid_phases'] and result['direction_passed']
    path=out/'round_01_fast_gate.json'
    temporary=path.with_suffix('.json.pending')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    os.replace(temporary,path)
    return result



def stateless_round_three_stop(success_count):
    if success_count>=218:
        return 'round_03_student_gate_passed'
    if success_count<=44:
        return 'round_03_improvement_gate_stopped'
    return None

def planned_rounds(start,total,stop_after=None):
    if stop_after is not None and not 1<=stop_after<=total:
        raise ValueError('--stop-after-round must be between 1 and --rounds')
    return range(start,total if stop_after is None else stop_after)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--record-root',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=32);p.add_argument('--seed',type=int,default=256)
    p.add_argument('--capacity-only',action='store_true');p.add_argument('--visual-diagnostic',action='store_true');p.add_argument('--standing-audit',choices=['none','no-render','render'],default='none');p.add_argument('--sampling',type=Path);p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--buffer',type=int,default=10000)
    p.add_argument('--capacity-cohorts',type=int,default=1);p.add_argument('--replay-sampling-dir',type=Path)
    p.add_argument('--resume-student',type=Path);p.add_argument('--skip-teacher-first',action='store_true')
    p.add_argument('--resume-state',type=Path);p.add_argument('--recovery-root',type=Path)
    p.add_argument('--allow-partial-recovery',action='store_true')
    p.add_argument('--resume-sha256')
    p.add_argument('--compress-recovery',action='store_true')
    p.add_argument('--validation-recovery-manifest',type=Path)
    p.add_argument('--command-history',action='store_true')
    p.add_argument('--relative-delta',action='store_true')
    p.add_argument('--stateless-dagger',action='store_true')
    p.add_argument('--stateless-teacher-smoke',action='store_true')
    p.add_argument('--stop-after-round',type=int)
    p.add_argument('--round1-fast-gate',action='store_true')
    p.add_argument('--audit-timing',action='store_true')
    p.add_argument('--audit-batched-foot-readback',action='store_true')
    p.add_argument('--allow-closed-collection-recovery',action='store_true')
    p.add_argument('--allow-resource-aborted-student-collection-recovery',action='store_true')
    p.add_argument('--capacity-updates',type=int,default=1)
    p.add_argument('--capacity-decisions-per-env',type=int,default=2);p.add_argument('--capacity-teacher-control',action='store_true')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;a.enable_cameras=True
    planned_rounds(0,a.rounds,a.stop_after_round)
    if a.stateless_dagger and (not a.relative_delta or not a.command_history or not a.compress_recovery
                               or a.skip_teacher_first or a.capacity_teacher_control):
        p.error('Stateless DAgger requires command history, relative delta, compression and round-0 demonstrations')
    if a.stateless_teacher_smoke and (not a.stateless_dagger or not a.capacity_only or a.num_envs!=64):
        p.error('Stateless teacher smoke requires stateless diagnostic mode and 64 environments')
    if a.capacity_updates<1:p.error('--capacity-updates must be positive')
    if a.stop_after_round is not None and (a.capacity_only or a.visual_diagnostic or a.standing_audit!='none'):
        p.error('--stop-after-round requires formal DAgger')
    if a.round1_fast_gate and (a.capacity_only or a.visual_diagnostic or a.standing_audit!='none' or a.skip_teacher_first):
        p.error('--round1-fast-gate requires formal DAgger with a teacher first round')

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
            if (a.relative_delta or not a.capacity_only) and inventory.get(name)!=actual:raise ValueError('Teacher qualification source differs: '+name)
    app=AppLauncher(a).app
    from .lifecycle_cleanup import install_stop_handler,check_stop,SafeStop
    install_stop_handler()
    from isaacsim.core.utils.stage import create_new_stage
    from .learning import DaggerSession
    from .learning_episode import run_episode
    from .storage import StorageGuard,scoped_storage_guard
    import torch
    from .learning import StandingAuditSession
    session=StandingAuditSession(a.standing_audit=='render') if a.standing_audit!='none' else DaggerSession('cuda:0',a.seed,a.buffer,diagnostic=a.visual_diagnostic,command_history=a.command_history,relative_delta=a.relative_delta,stateless_dagger=a.stateless_dagger)
    if a.resume_student:session.student.restore(a.resume_student)
    if a.relative_delta:
        from .reldelta_admission import RelativeDeltaStorageGuard
        guard=RelativeDeltaStorageGuard(a.record_root,a.out,a.recovery_root)
    else:guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    def audit_storage():
        if a.relative_delta:
            return scoped_storage_guard(OUTPUT,WORKTREE/'legmm_skill/tabletop',guard)
        return nullcontext()
    status=dict(PID=os.getpid(),phase='standing_render_audit' if a.standing_audit!='none' else 'visual_diagnostic' if a.visual_diagnostic else 'capacity_pipeline' if a.capacity_only else 'dagger',completed=False,
        formal_training=not (a.capacity_only or a.visual_diagnostic or a.standing_audit!='none'),dagger_started=not (a.capacity_only or a.visual_diagnostic or a.standing_audit!='none'),ppo_started=False,
        teacher_gate_passed=bool(q.get('teacher_gate_passed')),diagnostic_only=a.capacity_only or a.visual_diagnostic or a.standing_audit!='none',
        num_envs=a.num_envs,seed=a.seed,env_decisions=0,optimizer_updates=getattr(getattr(session,'student',None),'updates',0),checkpoint=None,
        resumed_student=str(a.resume_student) if a.resume_student else None,buffer_restored=False,
        total_env_decision_budget=94*a.num_envs if a.visual_diagnostic or a.standing_audit!='none' else a.capacity_decisions_per_env*a.num_envs*a.capacity_cohorts if a.capacity_only else 32000*a.rounds,effective_batch=64,
        gradient_microbatch=8,learning_rate=1e-4,capacity_target=4096,rounds=[],
        command_history=a.command_history,relative_delta=a.relative_delta,
        output_contract=session.student.output_contract if hasattr(session,'student') else None,
        stop_after_round=a.stop_after_round,round1_fast_gate=a.round1_fast_gate,
        actor_state_dim=14 if a.command_history else 7,
        capacity_decisions_per_env=a.capacity_decisions_per_env,capacity_teacher_control=a.capacity_teacher_control)
    if a.stateless_dagger:
        status.update(stateless_dagger=True,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])
    def save():
        status.update(new_env_decisions_executed=session.seen-restored_seen,
            optimizer_updates_executed_this_attempt=getattr(getattr(session,'student',None),'updates',0)-restored_updates)
        guard.check();(a.out/'training_status.json').write_text(json.dumps(status,indent=2)+'\n')
        for fn in ['progress.json','attention.json']:(a.record_root/fn).write_text(json.dumps(status,indent=2)+'\n')
    from .dagger_recovery import save_latest,restore_latest
    recovery_root=a.recovery_root or a.out
    allowed_recovery= a.record_root if a.relative_delta and a.capacity_only else OUTPUT
    if not recovery_root.resolve().is_relative_to(allowed_recovery.resolve()):raise ValueError('W8-owned recovery root required')
    episode=0;seed=a.seed;start_round=0;pending_validation=None;resume_cursor={}
    if a.resume_state:
        recovered,cursor=restore_latest(session,a.resume_state,allow_partial=a.allow_partial_recovery,
            expected_sha256=a.resume_sha256,expected_formal=status['formal_training'],updates_per_round=a.updates,allow_closed_collection=a.allow_closed_collection_recovery,expected_num_envs=a.num_envs,allow_resource_abort=a.allow_resource_aborted_student_collection_recovery,validation_manifest=a.validation_recovery_manifest);resume_cursor=cursor
        if bool(recovered['diagnostic_only'])!=bool(status['diagnostic_only']):raise ValueError('Diagnostic weights cannot resume formal training')
        status=recovered;status.update(PID=os.getpid(),completed=False,phase='full_round_recovered',
            relative_delta=a.relative_delta,output_contract=session.student.output_contract,
            stop_after_round=a.stop_after_round,round1_fast_gate=a.round1_fast_gate,
            resumed_from=str(a.resume_state),buffer_restored=True,num_envs=a.num_envs,reason=None,
            recovery_partial=bool(cursor.get('partial_phase')),physical_episode_resumed=False,
            lost_current_round_requires_recollection=False,recovery_is_full_round_only=False,
            partial_checkpoint_requires_explicit_stage_recovery=bool(cursor.get('partial_phase')),
            current_round_remaining_updates=a.updates-cursor.get('updates_completed_in_round',0) if cursor.get('partial_phase') else a.updates,
            remaining_env_decisions=status['total_env_decision_budget']-session.seen)
        start_round=cursor['next_round'];seed=cursor['next_seed'];episode=cursor['episode'];pending_validation=cursor['validation_pending']
        status.update(source_recovery=dict(path=str(a.resume_state),sha256=a.resume_sha256,env_decisions=session.seen,optimizer_updates=session.student.updates,cursor=cursor),latest=None,latest_save={},
            remaining_optimizer_updates=a.rounds*a.updates-session.student.updates)
        if cursor.get('validation_recovery_plan'):
            status.update(validation_recovery_plan=cursor['validation_recovery_plan'],interrupted_prefix_classification='resource_interruption',unresolved_episodes_excluded_from_complete_outcomes=True)
        if cursor.get('resource_abort_plan'):
            status.update(phase='resource_aborted_student_collection_recovered',resource_abort_recovery_plan=cursor['resource_abort_plan'],
                interrupted_prefix_classification='resource_interruption',unresolved_episodes_excluded_from_complete_outcomes=True)
        if cursor.get('closed_collection_plan'):status.update(closed_collection_boundary_recovered=True,closed_collection_recovery_plan=cursor['closed_collection_plan'])
    restored_seen=session.seen;restored_updates=getattr(getattr(session,'student',None),'updates',0)
    def latest(next_round,pending=None):
        status.update(env_decisions=session.seen,optimizer_updates=session.student.updates,
            new_env_decisions_executed=session.seen-restored_seen,
            optimizer_updates_executed_this_attempt=session.student.updates-restored_updates,
            validation_pending=pending,current_round_remaining_updates=0,
            partial_checkpoint_requires_explicit_stage_recovery=False,
            remaining_env_decisions=status['total_env_decision_budget']-session.seen)
        record=save_latest(session,status,next_round,seed,episode,recovery_root,guard,validation_pending=pending,compress=a.compress_recovery)
        status.update(latest=str(recovery_root/'latest.pt'),latest_save=record);save()
        check_stop('atomic full-round checkpoint complete')
    def validate_round(number):
        checkpoint=a.out/f'round_{number:02d}.pt'
        session.student.save(checkpoint,dict(visual=session.visual.manifest(),qualification=str(a.qualification),
            physics='scene_all_actors_4_0',round=number-1,env_decisions=session.seen),guard=guard)
        status.update(checkpoint=str(checkpoint),phase='student_fixed_validation');save()
        validation=a.out/f'validation_round_{number:02d}';validation.mkdir(exist_ok=False)
        with audit_storage():
            summary=evaluate_fixed(session,app,validation,a.qualification,a.record_root,audit_timing=a.audit_timing,audit_batched_foot_readback=a.audit_batched_foot_readback,reuse_manifest=a.validation_recovery_manifest if number==pending_validation else None)
        record=next(row for row in status['rounds'] if row['round']==number-1)
        record.update(validation_success=summary['success_count'],student_gate_passed=summary['student_gate_passed'])
        if summary['success_count']>status.get('best_validation_success',-1):
            status.update(best_validation_success=summary['success_count'],best_checkpoint=str(checkpoint),
                best_validation_summary=str(validation/'summary.json'),student_gate_passed=summary['student_gate_passed'])
        milestone(number)
        stop_phase=stateless_round_three_stop(summary['success_count']) if a.stateless_dagger and number==3 else None
        if stop_phase:
            status.update(completed=True,phase=stop_phase)
        latest(number)
        return stop_phase is not None
    def milestone(number):
        if a.stop_after_round==number:
            status.update(completed=True,phase=f'round_{number:02d}_milestone_complete')
    def round1_stop(source):
        result=write_round1_fast_gate(source,a.out,a.stateless_dagger)
        status.update(round1_gate_result=result)
        if result['passed']:save();return False
        status.update(completed=True,phase='round_01_fast_gate_stopped')
        save()
        return True
    save()
    try:
        if a.stateless_teacher_smoke:
            with audit_storage():
                summary=evaluate_fixed(session,app,a.out,a.qualification,a.record_root,
                    audit_timing=a.audit_timing,audit_batched_foot_readback=a.audit_batched_foot_readback,
                    stateless_teacher_smoke=True)
            status.update(completed=True,phase='stateless_teacher_smoke_complete',
                          teacher_smoke_passed=summary['teacher_smoke_passed'],
                          teacher_smoke_success=summary['success_count'])
            save()
            return
        if pending_validation and validate_round(pending_validation):return
        if a.round1_fast_gate and start_round==2 and round1_stop(Path(status['current_episode']).parent):return
        total_rounds=a.capacity_cohorts if a.capacity_only else 1 if a.visual_diagnostic or a.standing_audit!='none' else a.rounds
        for round_index in planned_rounds(start_round,total_rounds,a.stop_after_round):
            acknowledge(a.record_root,'capacity_pipeline' if a.capacity_only else 'dagger_collection',formal_training=not a.capacity_only,ppo_started=False)
            session.round=round_index;session.teacher_drives=(round_index==0 and not a.skip_teacher_first) or (a.capacity_only and a.capacity_teacher_control)
            session.shadow_student=bool(a.capacity_only and a.capacity_teacher_control)
            restoring_update=(round_index==start_round and resume_cursor.get('partial_phase')=='dagger_update')
            initial_update=resume_cursor.get('updates_completed_in_round',0) if restoring_update else 0
            previous_loss_sum=resume_cursor.get('loss_sum',0.) if restoring_update else 0.
            previous_update_elapsed=resume_cursor.get('update_elapsed_s',0.) if restoring_update else 0.
            status.update(current_round=round_index,current_round_update_count=initial_update,
                current_round_loss_sum=previous_loss_sum,current_round_update_elapsed_s=previous_update_elapsed,
                current_round_remaining_updates=(a.capacity_updates if a.capacity_only else a.updates)-initial_update)
            session.budget=(round_index+1)*(94*a.num_envs if a.visual_diagnostic or a.standing_audit!='none' else a.capacity_decisions_per_env*a.num_envs if a.capacity_only else 32000)
            while not session.exhausted:
                acknowledge(a.record_root,'capacity_cohort' if a.capacity_only else 'dagger_collection',formal_training=not a.capacity_only,ppo_started=False)
                out=a.out/f'round_{round_index:02d}_episode_{episode:04d}';out.mkdir(exist_ok=False)
                status.update(phase='capacity_pipeline' if a.capacity_only else 'dagger_teacher_collection' if session.teacher_drives else 'dagger_student_collection',current_episode=str(out));save()
                args=episode_args(out,a.num_envs,seed)
                args.audit_timing=a.audit_timing
                args.audit_batched_foot_readback=a.audit_batched_foot_readback
                if a.sampling:args.sampling=a.sampling
                if a.replay_sampling_dir:
                    sample=a.replay_sampling_dir/f'sampling_{seed}.json'
                    if sample.exists():args.sampling=sample
                episode_seen_before=session.seen
                labels_scope=session.episode_labels.install() if a.stateless_dagger else nullcontext()
                with audit_storage(), labels_scope:
                    result=run_episode(args,app,session)
                if result['safety_stop']:raise RuntimeError('Standing stability safety stop')
                status.update(env_decisions=session.seen,physical_env_decisions=session.physical_env_decisions,
                    collected_samples=session.collected_samples,current_episode=str(out),
                    actual_envs=result['num_envs'],physical_steps=result['physical_steps'])
                save()
                if not result.get('requested_safe_stop') or session.seen>episode_seen_before:seed+=a.num_envs
                episode+=1
                if a.visual_diagnostic or a.standing_audit!='none':session.budget=session.seen
                if seed+a.num_envs>10000:seed=256
                create_new_stage();gc.collect();torch.cuda.empty_cache()
                from .lifecycle_cleanup import trim_unused
                (out/'post_stage_memory_release.json').write_text(json.dumps(trim_unused(),indent=2))
                from .compact_current_logs import finalize_closed_label_log
                finalize_closed_label_log(out,allowed_root=a.out,guard=guard)
                if result.get('requested_safe_stop'):check_stop('collection action-block boundary after scene cleanup')
            if a.visual_diagnostic or a.standing_audit!='none':
                status.update(completed=True,phase='standing_audit_complete' if a.standing_audit!='none' else 'visual_diagnostic_complete',visual_correctness_pending=True);save();return
            losses=[];update_start=time.monotonic()
            update_limit=a.capacity_updates if a.capacity_only else a.updates
            status['phase']='capacity_update' if a.capacity_only else 'dagger_update'
            for update in range(initial_update,update_limit):
                losses.append(session.student.update(session.reservoir,batch=64,microbatch=8))
                status.update(current_round_update_count=update+1,current_round_loss_sum=previous_loss_sum+sum(losses),
                    current_round_update_elapsed_s=previous_update_elapsed+time.monotonic()-update_start,
                    current_round_remaining_updates=update_limit-update-1)
                check_stop('completed optimizer update')
                if update%100==0:
                    status.update(optimizer_updates=session.student.updates,phase='capacity_update' if a.capacity_only else 'dagger_update');save()
            record=dict(update_elapsed_s=previous_update_elapsed+time.monotonic()-update_start,round=round_index,teacher_drives=session.teacher_drives,env_decisions=session.seen,
                optimizer_updates=session.student.updates,mean_loss=(previous_loss_sum+sum(losses))/update_limit,
                retained=session.retained,reservoir_bytes=session.reservoir.tensor_bytes,last_cohort_physical_steps=status['physical_steps'])
            if a.stateless_dagger:
                record.update(summarize_collection(a.out,round_index,True))
                record.update(demo_retained=len(session.reservoir.demo.items),dagger_retained=len(session.reservoir.dagger.items))
            status['rounds'].append(record)
            if not a.capacity_only or round_index==a.capacity_cohorts-1:
                latest(round_index+1,None if a.capacity_only or (round_index+1)%3 else round_index+1)
            if a.round1_fast_gate and round_index==1 and round1_stop(a.out):return
            if a.capacity_only and round_index==a.capacity_cohorts-1:
                checkpoint=a.out/(f'capacity_restore_test_{round_index:02d}.pt' if a.capacity_only and a.capacity_cohorts>1 else 'capacity_restore_test.pt' if a.capacity_only else f'round_{round_index+1:02d}.pt')
                session.student.save(checkpoint,dict(visual=session.visual.manifest(),qualification=str(a.qualification),
                    physics='scene_all_actors_4_0',round=round_index,env_decisions=session.seen),guard=guard)
                status['checkpoint']=str(checkpoint)
                session.student.restore(checkpoint)
                from .dagger_recovery import reservoir_state
                def cache_rng():
                    if a.stateless_dagger:
                        state=reservoir_state(session.reservoir)
                        return state['demo']['rng'],state['dagger']['rng']
                    return session.reservoir.rng.getstate()
                expected_updates=session.student.updates;expected_seen=session.seen;expected_rng=cache_rng()
                _,restored_cursor=restore_latest(session,recovery_root/'latest.pt')
                if session.student.updates!=expected_updates or session.seen!=expected_seen or cache_rng()!=expected_rng:
                    raise RuntimeError('Full capacity session restore differs')
                status.update(full_session_restore_passed=True,restored_reservoir_frames=session.retained,
                    restored_cursor=restored_cursor,full_restore_peak_rss_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024)
            elif not a.capacity_only and (round_index+1)%3==0 and validate_round(round_index+1):return
            status.update(optimizer_updates=session.student.updates,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
        if a.stop_after_round is not None:milestone(a.stop_after_round)
        else:status.update(completed=True,phase='capacity_complete' if a.capacity_only else 'dagger_budget_complete')
        save()
    except BaseException as error:
        status.update(reason=repr(error),completed=False,last_phase=status['phase'],phase='dagger_stopped',env_decisions=session.seen,
            physical_env_decisions=session.physical_env_decisions,collected_samples=session.collected_samples)
        status.update(new_env_decisions_executed=session.seen-restored_seen,
            lost_current_round_requires_recollection=True,recovery_is_full_round_only=True)
        student=getattr(session,'student',None)
        if student is not None:
            status['optimizer_updates']=student.updates
            previous=status.get('latest_save',{})
            unchanged=(previous.get('optimizer_updates')==student.updates and previous.get('env_decisions')==session.seen)
            if isinstance(error,SafeStop) and not unchanged:
                try:
                    partial_root=recovery_root
                    if a.relative_delta:
                        partial_root=a.out/'interrupted_recovery'
                        partial_root.mkdir(exist_ok=False)
                    record=save_latest(session,status,session.round,seed,episode,partial_root,guard,complete_round_boundary=False,compress=a.compress_recovery)
                    status.update(latest=record['path'],latest_save=record,partial_checkpoint_requires_explicit_stage_recovery=True)
                except BaseException as save_error:status['controlled_stop_save_error']=repr(save_error)
            if (recovery_root/'latest.pt').exists():status['consistent_recovery']=str(recovery_root/'latest.pt')
        save();raise
    finally:app.close(wait_for_replicator=False)


if __name__=='__main__':main()
