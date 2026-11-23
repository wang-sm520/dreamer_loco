"""Gated residual PPO runner; no Isaac execution before student qualification."""
import argparse,gc,hashlib,json,os,time,signal
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .coordinator import acknowledge


def teacher_qualification(path):
    from .physics_config import verify_task_physics_audit
    q=json.loads(path.read_text())
    if not q.get('teacher_gate_passed') or q.get('success_count',0)<231 or q.get('num_episodes')!=256:
        raise ValueError('True4/0 fixed256 teacher >=231 required')
    if sorted(row['seed'] for row in q['episodes'])!=list(range(10000,10256)):
        raise ValueError('Teacher qualification seed set differs')
    critical={'geometry.py','closed_loop.py','control.py','spec.py','outcome.py','collision.py',
        'isaac_scene.py','physics_config.py','assets.py','physics_audit.py'}
    for parent in {Path(row['result_path']).parent for row in q['episodes']}:
        verify_task_physics_audit(json.loads((parent/'physics_audit.json').read_text()))
        inventory={Path(row['path']).name:row['sha256'] for row in json.loads((parent/'source_inventory.json').read_text())}
        for name in critical:
            if inventory.get(name)!=hashlib.sha256((WORKTREE/'legmm_skill/tabletop'/name).read_bytes()).hexdigest():
                raise ValueError('Teacher qualification source changed: '+name)
    return q


def student_qualification(status_path,checkpoint):
    status=json.loads(status_path.read_text())
    if not status.get('student_gate_passed') or status.get('best_validation_success',0)<218:
        raise ValueError('Frozen256 student >=218/256 qualification required')
    if Path(status['best_checkpoint']).resolve()!=checkpoint.resolve():
        raise ValueError('PPO must use the qualified best frozen student')
    validation=Path(status.get('best_validation_summary',status_path.parent/('validation_'+checkpoint.stem)/'summary.json'))
    v=json.loads(validation.read_text()) if validation.exists() else {}
    if not (v.get('success_count',0)==status['best_validation_success'] and v.get('success_count',0)>=218 and sorted(x['seed'] for x in v['episodes'])==list(range(10000,10256))):
        raise ValueError('Missing original256 student validation evidence')
    return dict(path=str(status_path),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        validation=str(validation),success_count=status['best_validation_success'],denominator=256)


def evaluate_fixed(session,app,out,qualification,record_root,limit=None):
    from .ppo_learning import ResidualSession
    from .ppo_episode import run_episode
    from .train_dagger import episode_args
    from .evaluate_teacher import interval
    from isaacsim.core.utils.stage import create_new_stage
    import torch
    evaluation=ResidualSession(session.student,session.visual,train=False,
        actor_state={k:v.detach().clone() for k,v in session.algorithm.actor.state_dict().items()})
    q=json.loads(qualification.read_text());episodes=[]
    for source in sorted({row['result_path'] for row in q['episodes']}):
        acknowledge(record_root,'ppo_validation',ppo_started=True,dagger_started=True)
        parent=Path(source).parent;data=json.loads(Path(source).read_text());seed=data['seed']
        batch=out/f'seed_{seed}';batch.mkdir(exist_ok=False)
        args=episode_args(batch,len(data['states']),seed);args.residual_velocity=False
        args.sampling=Path(json.loads((parent/'sampling_source.json').read_text())['path'])
        result=run_episode(args,app,evaluation)
        from .lifecycle_cleanup import check_stop
        if result.get('requested_safe_stop'):check_stop('PPO validation action-block boundary')
        if result['safety_stop']:raise RuntimeError('PPO evaluation standing safety stop')
        episodes.extend(result['episodes']);create_new_stage();gc.collect();torch.cuda.empty_cache()
        if limit is not None and len(episodes)>=limit:break
    total=256 if limit is None else limit
    if sorted(row['seed'] for row in episodes)!=list(range(10000,10000+total)):raise ValueError('PPO validation seeds differ')
    k=sum(row['success'] for row in episodes)
    result=dict(driver='residual_ppo_deterministic',success_count=k,num_episodes=total,wilson95=interval(k,total),
        student_gate_passed=k/total>=.85,episodes=episodes)
    evaluation.features.close()
    (out/'summary.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--student-status',type=Path,required=True);p.add_argument('--student-checkpoint',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--record-root',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=128);p.add_argument('--seed',type=int,default=4096)
    p.add_argument('--updates',type=int,default=1000);p.add_argument('--capacity-only',action='store_true')
    p.add_argument('--physical-decisions',type=int,default=1000*32*32)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;a.enable_cameras=True
    if a.updates<=0 or a.physical_decisions<=0:raise ValueError('Positive PPO budgets required')
    qualified=student_qualification(a.student_status,a.student_checkpoint)
    base_rows=[row for row in json.loads(Path(qualified['validation']).read_text())['episodes'] if row['seed']<10032]
    baseline_success=sum(row['success'] for row in base_rows)
    baseline_collisions=sum((row['reason'] or '').startswith('collision:') for row in base_rows)
    q=teacher_qualification(a.qualification)
    app=AppLauncher(a).app
    from .lifecycle_cleanup import install_stop_handler,trim_unused
    install_stop_handler()
    from .student import Student
    from .visual import DualDepth
    from .ppo_learning import ResidualSession
    from .ppo_episode import run_episode
    from .train_dagger import episode_args
    from .storage import StorageGuard
    from isaacsim.core.utils.stage import create_new_stage
    import torch
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    student=Student.from_checkpoint(a.student_checkpoint,'cuda:0',a.seed)
    status=dict(PID=os.getpid(),phase='ppo',completed=False,formal_training=not a.capacity_only,
        dagger_started=True,ppo_started=not a.capacity_only,capacity_only=a.capacity_only,
        student_qualification=qualified,num_envs=a.num_envs,optimizer_updates=0,valid_samples=0,
        physical_env_decisions=0,update_budget=1 if a.capacity_only else a.updates,effective_batch=1024,
        physical_env_decision_budget=a.physical_decisions,physical_env_steps=0,
        maximum_complete_cohort_tail_decisions=a.num_envs*94,
        budget_semantics='original1000x32x32 environment decisions; masked terminal padding reduces actual optimizer updates',
        minibatch=256,learning_epochs=5,learning_rate=3e-4,seed=a.seed,checkpoints=[])
    def save_status():
        guard.check();(a.out/'training_status.json').write_text(json.dumps(status,indent=2)+'\n')
        for name in ('progress.json','attention.json'):(a.record_root/name).write_text(json.dumps(status,indent=2)+'\n')
    def on_update(record):
        if not all(torch.isfinite(torch.tensor(value)) for value in record['losses'].values()):raise FloatingPointError('Nonfinite PPO loss')
        status.update(optimizer_updates=record['update'],valid_samples=record['valid_samples'],last_losses=record['losses'],
            physical_env_decisions=session.physical_env_decisions,optimized_valid_samples=record['update']*1024)
        with (a.out/'updates.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        save_status()
    session=ResidualSession(student,DualDepth('cuda:0'),on_update=on_update)
    # Finish the current cohort at the budget boundary, but stop optimizer
    # updates exactly at the recorded budget and disclose additional decisions.
    original_finish=session._finish_transition;next_validation=100
    def bounded_finish(obs,outcomes):
        if session.updates>=min(status['update_budget'],next_validation):session.train=False
        original_finish(obs,outcomes)
    session._finish_transition=bounded_finish
    episode=0;seed=a.seed;last_saved=0;best_actor=None;best_success=-1;save_status()
    try:
        while session.updates<status['update_budget'] and (a.capacity_only or session.physical_env_decisions<a.physical_decisions):
            acknowledge(a.record_root,'ppo_capacity' if a.capacity_only else 'ppo_training',ppo_started=not a.capacity_only,dagger_started=True)
            out=a.out/f'episode_{episode:04d}';out.mkdir(exist_ok=False)
            result=run_episode(episode_args(out,a.num_envs,seed),app,session)
            from .lifecycle_cleanup import check_stop
            if result.get('requested_safe_stop'):check_stop('PPO collection action-block boundary')
            if result['safety_stop']:raise RuntimeError('PPO standing safety stop')
            status.update(physical_env_decisions=session.physical_env_decisions,valid_samples=session.valid_samples,
                unselected_samples=session.unselected_samples,last_episode=str(out),physical_steps=result['physical_steps'],
                physical_env_steps=status['physical_env_steps']+result['physical_steps']*a.num_envs)
            if session.updates//500>last_saved//500 or session.updates>=status['update_budget']:
                checkpoint=a.out/f'ppo_{session.updates:04d}.pt'
                session.save(checkpoint,dict(student=qualified,physics='scene_all_actors_4_0',execute=4,horizon=8),guard)
                payload=torch.load(checkpoint,map_location=session.device,weights_only=False)
                if payload['format']!='w8_residual_ppo_v1':raise ValueError('Wrong PPO save format')
                for name,value in session.algorithm.actor.state_dict().items():
                    if not torch.equal(value,payload['actor_state_dict'][name]):raise RuntimeError('PPO actor save roundtrip differs')
                session.algorithm.actor.load_state_dict(payload['actor_state_dict'])
                session.algorithm.critic.load_state_dict(payload['critic_state_dict'])
                session.algorithm.optimizer.load_state_dict(payload['optimizer_state_dict'])
                del payload
                status['checkpoints'].append(str(checkpoint));last_saved=session.updates
            create_new_stage();gc.collect();torch.cuda.empty_cache()
            (out/'post_stage_memory_release.json').write_text(json.dumps(trim_unused(),indent=2));save_status()
            if not a.capacity_only and session.updates>=next_validation:
                status['phase']='ppo_validation_smoke';save_status()
                validation=a.out/f'validation_update_{session.updates:04d}';validation.mkdir(exist_ok=False)
                # Same predeclared first32 fixed val seeds every100 updates;
                # full256 qualifies the selected policy after this budget.
                summary=evaluate_fixed(session,app,validation,a.qualification,a.record_root,limit=32)
                collisions=sum((row['reason'] or '').startswith('collision:') for row in summary['episodes'])
                if summary['success_count']<baseline_success-6 or collisions>baseline_collisions+3:
                    raise RuntimeError('PPO validation collapse/contact increase; diagnose before more updates')
                if summary['success_count']>best_success:
                    best_success=summary['success_count']
                    best_actor={k:v.detach().cpu().clone() for k,v in session.algorithm.actor.state_dict().items()}
                    status.update(best_smoke_success=best_success,best_smoke_update=session.updates)
                next_validation+=100;session.train=True;status['phase']='ppo';status['last_smoke_update']=session.updates;save_status()
            episode+=1;seed+=a.num_envs
            if seed+a.num_envs>10000:seed=4096
        if not a.capacity_only:
            status['incomplete_rollout_samples_not_optimized']=int(session.algorithm.storage.valid[:session.algorithm.storage.step].sum())
            # The fixed physical interaction budget can end with fewer1000
            # updates after masking terminal padding. Save that actual endpoint.
            if session.updates!=last_saved:
                checkpoint=a.out/f'ppo_{session.updates:04d}.pt'
                session.save(checkpoint,dict(student=qualified,physics='scene_all_actors_4_0',budget_end=True),guard)
                status['checkpoints'].append(str(checkpoint))
            if status.get('last_smoke_update')!=session.updates:
                validation=a.out/f'validation_budget_end_{session.updates:04d}';validation.mkdir(exist_ok=False)
                summary=evaluate_fixed(session,app,validation,a.qualification,a.record_root,limit=32)
                if summary['success_count']>best_success:
                    best_success=summary['success_count']
                    best_actor={k:v.detach().cpu().clone() for k,v in session.algorithm.actor.state_dict().items()}
                    status.update(best_smoke_success=best_success,best_smoke_update=session.updates)
            status['phase']='ppo_fixed_validation';save_status()
            if best_actor is not None:session.algorithm.actor.load_state_dict(best_actor)
            validation=a.out/'validation_final';validation.mkdir(exist_ok=False)
            summary=evaluate_fixed(session,app,validation,a.qualification,a.record_root)
            status.update(validation_success=summary['success_count'],ppo_gate_passed=summary['student_gate_passed'])
            selected=a.out/'selected_actor.pt';stream=__import__('io').BytesIO()
            torch.save(dict(format='w8_residual_actor_v1',actor_state_dict=session.algorithm.actor.state_dict(),
                student_qualification=qualified,validation=str(validation/'summary.json'),
                selected_at_update=status.get('best_smoke_update',session.updates)),stream)
            data=stream.getvalue();guard.check(len(data))
            with selected.open('xb') as file:file.write(data)
            status['selected_actor']=str(selected)
            status['physical_budget_tail_decisions']=max(0,session.physical_env_decisions-a.physical_decisions)
        status.update(completed=True,phase='ppo_capacity_complete' if a.capacity_only else 'ppo_budget_complete');save_status()
    except BaseException as error:
        status.update(reason=repr(error),completed=False,physical_env_decisions=session.physical_env_decisions,
            valid_samples=session.valid_samples,optimizer_updates=session.updates)
        from .lifecycle_cleanup import SafeStop
        if isinstance(error,SafeStop) and session.algorithm is not None and session.updates>0:
            stopped=a.out/f'controlled_stop_{session.updates:04d}.pt'
            try:
                session.save(stopped,dict(student=qualified,reason=repr(error),
                    incomplete_rollout_samples=int(session.algorithm.storage.valid[:session.algorithm.storage.step].sum())),guard)
                status['controlled_stop_checkpoint']=str(stopped)
            except BaseException as save_error:status['controlled_stop_save_error']=repr(save_error)
        save_status();raise
    finally:app.close(wait_for_replicator=False)


if __name__=='__main__':main()
