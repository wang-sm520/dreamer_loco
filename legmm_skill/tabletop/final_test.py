"""Paired independent test. This entry is gated until actual PPO has completed."""
import argparse,gc,hashlib,json,os,time,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .coordinator import acknowledge


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_samples(folder):
    """Check a precommitted manifest; never write/choose test states here."""
    path=folder/'fixed_manifest.json';manifest=json.loads(path.read_text())
    if not manifest.get('completed') or manifest['seed_range']!=[20000,20255]:raise ValueError('Precommitted independent256 required')
    for seed in range(20000,20256,32):
        name=f'sampling_{seed}.json'
        if manifest['files'].get(name)!=sha(folder/name):raise ValueError('Independent initial states changed')
        rows=json.loads((folder/name).read_text());accepted=[r['state']['seed'] for r in rows if r['accepted']]
        if accepted!=list(range(seed,seed+32)):raise ValueError('Independent sampling seed mismatch')
    return sha(path)


def summarize(rows):
    from .evaluate_teacher import interval
    if sorted(row['seed'] for row in rows)!=list(range(20000,20256)):raise ValueError('Incomplete independent driver test')
    groups=defaultdict(lambda:[0,0])
    for row in rows:
        s=row['state'];h=s['table_size'][2];distance=row['gate']['contract']['dist']
        for key in (s['shape'],'height:'+('low' if h<.55 else 'mid' if h<.60 else 'high'),
            'distance:'+('near' if distance<.6 else 'mid' if distance<.7 else 'far')):
            groups[key][0]+=int(row['success']);groups[key][1]+=1
    k=sum(row['success'] for row in rows)
    return dict(success_count=k,total=256,wilson95=interval(k,256),failures=dict(Counter(row['reason'] for row in rows)),
        strata={key:dict(success=a,total=b,wilson95=interval(a,b)) for key,(a,b) in groups.items()},episodes=rows)


class TeacherEvaluation:
    driver_name='teacher';teacher_drives=True;exhausted=False
    def __init__(self,visual):self.visual=visual
    def reset(self,states,scene,simulation,out):
        (out/'visual_manifest.json').write_text(json.dumps(self.visual.manifest(),indent=2))
    def decision(self,scene,robot,joints,arm_ids,finger_ids,last_command,labels,outcomes,frame,phases=None):
        self.visual.observe(scene)
        return labels
    def finish(self,outcomes):pass


def main():
    p=argparse.ArgumentParser()
    for name in ('out','sampling-dir','student-status','student-checkpoint','ppo-status','qualification','record-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--num-envs',type=int,choices=[32],default=32);p.add_argument('--seed',type=int,default=20000)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;a.enable_cameras=True
    from .train_ppo import student_qualification,teacher_qualification
    qualified=student_qualification(a.student_status,a.student_checkpoint);teacher_qualification(a.qualification)
    ps=json.loads(a.ppo_status.read_text())
    if not ps.get('completed') or ps.get('phase')!='ppo_budget_complete':raise ValueError('Actual completed PPO required before final test access')
    if ps['student_qualification']['checkpoint_sha256']!=qualified['checkpoint_sha256']:raise ValueError('PPO used different student')
    residual=Path(ps['selected_actor']);manifest_sha=frozen_samples(a.sampling_dir)
    committed=json.loads((a.sampling_dir/'fixed_manifest.json').read_text())
    if committed['models']!=dict(student_sha256=sha(a.student_checkpoint),residual_sha256=sha(residual)):
        raise ValueError('Models differ from precommitted independent test')
    status=dict(PID=os.getpid(),phase='independent_test',formal_training=False,completed=False,
        precommitted_initial_states_sha256=manifest_sha,models=dict(student_sha256=sha(a.student_checkpoint),residual_sha256=sha(residual)),
        seed_range=[20000,20255],num_envs=32,residual_velocity=False,drivers={})
    from .storage import StorageGuard
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    def save():
        guard.check();(a.out/'training_status.json').write_text(json.dumps(status,indent=2)+'\n')
        for fn in ('progress.json','attention.json'):(a.record_root/fn).write_text(json.dumps(status,indent=2)+'\n')
    save();app=AppLauncher(a).app
    from .lifecycle_cleanup import install_stop_handler,check_stop
    install_stop_handler()
    from .student import Student
    from .visual import DualDepth
    from .learning import EvaluationSession
    from .ppo_learning import ResidualSession
    from .video_evidence import RecordingVisual
    from .train_dagger import episode_args
    from . import learning_episode,ppo_episode
    from isaacsim.core.utils.stage import create_new_stage
    from .physics_config import verify_task_physics_audit
    import torch
    student=Student.from_checkpoint(a.student_checkpoint,'cuda:0',0);visual=RecordingVisual(DualDepth('cuda:0'),guard)
    actor=torch.load(residual,map_location='cuda:0',weights_only=False)['actor_state_dict']
    try:
        for name,session,runtime in [('teacher',TeacherEvaluation(visual),learning_episode),
            ('student',EvaluationSession(student,visual),learning_episode),
            ('residual_ppo',ResidualSession(student,visual,train=False,actor_state=actor),ppo_episode)]:
            rows=[];driver_out=a.out/name;driver_out.mkdir(exist_ok=False);started=time.monotonic();sim_seconds=0.
            for seed in range(20000,20256,32):
                acknowledge(a.record_root,'independent_test_'+name,dagger_started=True,ppo_started=True)
                if frozen_samples(a.sampling_dir)!=manifest_sha:raise ValueError('Independent manifest changed in flight')
                batch=driver_out/f'seed_{seed}';batch.mkdir(exist_ok=False)
                visual.begin(batch,name,seed)
                args=episode_args(batch,32,seed);args.residual_velocity=False;args.sampling=a.sampling_dir/f'sampling_{seed}.json';args.trace_teacher=True
                result=runtime.run_episode(args,app,session)
                if result.get('requested_safe_stop'):check_stop('independent test action-block boundary')
                sim_seconds+=result['elapsed_sim_s']
                visual.finish(result)
                from .audit_io import read_physics_audit
                verify_task_physics_audit(read_physics_audit(batch))
                if result['safety_stop']:raise RuntimeError('Independent test standing safety stop; preserve all failures')
                latched={}
                with (batch/'closed_loop_labels.jsonl').open() as label_log:
                    for _ in range(32):
                        label=json.loads(next(label_log));reason=label['metadata'].get('reason') or ''
                        latched[label['seed']]=reason if reason.startswith('initial_gate:') else None
                for state,outcome,gate in zip(result['states'],result['episodes'],result['gate_reports']):
                    rows.append(dict(outcome,state=state,gate=gate,initial_gate_latched_reason=latched[outcome['seed']],result_path=str(batch/'result.json')))
                status.update(current_driver=name,completed_episodes=len(rows),current_episode=str(batch));save()
                create_new_stage();gc.collect();torch.cuda.empty_cache()
            summary=summarize(rows);summary.update(elapsed_wall_s=time.monotonic()-started,batch_simulation_seconds=sim_seconds,
                initial_gate_reject_count=sum(bool(row['initial_gate_latched_reason']) for row in rows),
                raw_contact_episode_count=sum(bool(row['contacts']) for row in rows))
            narrow=driver_out/'narrow_contact_audit.json'
            subprocess.run([sys.executable,'-m','tabletop.audit_narrow_contacts',
                *[str(driver_out/f'seed_{seed}') for seed in range(20000,20256,32)],'--out',str(narrow)],check=True)
            audit=json.loads(narrow.read_text())
            successful=[row for row in audit['episodes'] if row['shape']=='box' and row['success']]
            evidence=[row.get('first_success') or row.get('last_before_success') for row in successful]
            angles=[row['closing_axis_error_deg'] for row in evidence if row]
            approach_angles=[row['approach_normal_error_deg'] for row in evidence if row]
            summary['narrow_audit']=dict(path=str(narrow),successful_boxes=len(successful),
                pose_evidence_count=len(angles),missing_success_frame_count=audit['success_frames_not_sampled'],
                over15_degrees=sum(angle>15+1e-3 for angle in angles),max_degrees=max(angles) if angles else None,
                approach_over15_degrees=sum(angle>15+1e-3 for angle in approach_angles),
                approach_max_degrees=max(approach_angles) if approach_angles else None,
                teacher_reference_phase_is_not_learned_stage=True)
            from .final_pose_audit import summarize_successful_poses
            posture=summarize_successful_poses(driver_out,rows)
            posture_path=driver_out/'actual_successful_postures.json';posture_path.write_text(json.dumps(posture,indent=2))
            summary['actual_posture_audit']=dict(path=str(posture_path),**{key:value for key,value in posture.items() if key!='episodes'})
            (driver_out/'summary.json').write_text(json.dumps(summary,indent=2))
            status['drivers'][name]=summary;save()
        status.update(completed=True,phase='independent_test_complete');save()
        (a.out/'summary.json').write_text(json.dumps(status,indent=2)+'\n')
    except BaseException as error:
        status.update(reason=repr(error));save();raise
    finally:app.close(wait_for_replicator=False)


if __name__=='__main__':main()
