"""Bounded same-real-state full contact classification benchmark; diagnostic only."""
import argparse,ast,copy,json,os,time
from pathlib import Path
import numpy as np

class ContactProof:
    def __init__(self,out):
        from .prepare_v8_foot_patch import candidate_source,contact_loop
        original=Path(__file__).with_name('learning_episode.py').read_text()
        self.codes=[compile(ast.fix_missing_locations(ast.Module(body=[contact_loop(s)],type_ignores=[])),'real_GPU_contact_loop','exec') for s in (original,candidate_source(original))]
        self.rows=[];self.out=out
    def check(self,scope):
        tick=scope['tick']
        if tick not in (301,302,303,304,309,310,317,318,325,326,331,332):return
        import torch
        from .foot_readback_candidate import batched_foot_centers
        robot=scope['robot'];scene=scope['scene'];n=scope['n']
        before=robot.root_physx_view.get_link_transforms().clone();clock=float(scope['simulation'].current_time);steps=scope['engine_steps'].copy()
        ids={b:robot.find_bodies(b)[0][0] for b in scope['DOG_BODIES'] if b.endswith('_calf')}
        results={};elapsed={};foot_error=0.
        for body,bid in ids.items():
            batch=batched_foot_centers(robot.data.body_pos_w,robot.data.body_quat_w,bid,scope['tensor'],scope['maths'].quat_apply)
            for i in range(n):
                original=robot.data.body_pos_w[i,bid]+scope['maths'].quat_apply(robot.data.body_quat_w[i,bid][None],scope['tensor']([[0,0,-.275]]))[0]
                foot_error=max(foot_error,float(np.abs(batch[i]-original.cpu().numpy()).max()))
        for method in ((0,1) if len(self.rows)%2==0 else (1,0)):
            local=dict(scope);local.update(use_batched_foot_readback=bool(method),calf_body_ids=ids,batched_foot_centers=batched_foot_centers,
                bad=[set() for _ in range(n)],touch=[[False,False] for _ in range(n)],event_counts=copy.deepcopy(scope['event_counts']),contact_peaks=copy.deepcopy(scope['contact_peaks']))
            torch.cuda.synchronize();start=time.perf_counter_ns();exec(self.codes[method],local);torch.cuda.synchronize();elapsed[method]=(time.perf_counter_ns()-start)/1e6
            keys=('bad','touch','feet_peak','finger_peak','event_counts','contact_peaks');result={k:local[k] for k in keys}
            outcomes=copy.deepcopy(scope['outcomes'])
            for i,o in enumerate(outcomes):
                s=scope['states'][i];bottom=scope['lowest_point'](float(scope['position'][i,2]),scope['rotations'][i],s.shape,s.size)
                o.update(bottom,float(scope['target'].data.root_lin_vel_w[i,2]),local['touch'][i],local['bad'][i],bool(scope['heights'][i]<.2 or scope['upz'][i]<=.5))
            result['outcomes']=outcomes;results[method]=result
        assert results[0]==results[1],'GPU contact classification/outcome differs'
        assert foot_error==0.,foot_error
        assert torch.equal(before,robot.root_physx_view.get_link_transforms()) and clock==float(scope['simulation'].current_time) and steps==scope['engine_steps']
        self.rows.append(dict(tick=tick,order=[0,1] if len(self.rows)%2==0 else [1,0],original_ms=elapsed[0],batched_ms=elapsed[1],foot_max_abs_error=foot_error,all_classifications_counts_peaks_outcomes_equal=True,engine_unchanged=True,raw_forbidden_events=sum(sum(v.values()) for v in results[0]['event_counts'])))
        (self.out/'same_state_comparisons.json').write_text(json.dumps(self.rows,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--record-root',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a,unused=p.parse_known_args();a.headless=True;a.enable_cameras=True;app=AppLauncher(a).app
    from .lifecycle_cleanup import install_stop_handler
    install_stop_handler()
    from .learning import DaggerSession
    from .train_dagger import episode_args
    from . import learning_episode
    proof=ContactProof(a.out)
    source=Path(learning_episode.__file__).read_text();needle='                for body in DOG_BODIES+ARM_BODIES:';assert source.count(needle)==1
    source=source.replace(needle,'                GPU_CONTACT_PROOF.check(dict(dict(globals(),**locals()),engine_steps=engine_steps))\n'+needle)
    ns=dict(learning_episode.__dict__,GPU_CONTACT_PROOF=proof);exec(compile(source,'diagnostic_only_same_state_runtime','exec'),ns)
    session=DaggerSession('cuda:0',640,capacity=4);session.budget=128;session.shadow_student=True
    episode=a.out/'diagnostic_train_seed640';episode.mkdir()
    args=episode_args(episode,64,640)
    args.sampling=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_1_random_reachable_20261114/dagger_true4_visual64_retry1_resume_01/round_01_episode_0006/sampling.json')
    result=ns['run_episode'](args,app,session)
    assert len(proof.rows)==12 and not result['safety_stop']
    from .physics_config import verify_task_physics_audit
    verify_task_physics_audit(json.loads((episode/'physics_audit.json').read_text()))
    original=np.array([r['original_ms'] for r in proof.rows]);batched=np.array([r['batched_ms'] for r in proof.rows])
    report=dict(PID=os.getpid(),completed=True,diagnostic_only=True,formal_training=False,formal_env_decisions_added=0,optimizer_updates_added=0,num_envs=64,actual_diagnostic_env_decisions=session.seen,source_formal_model_used=False,physical_steps=result['physical_steps'],engine_steps=result['engine_controlled_steps'],same_state_pairs=len(proof.rows),all_pairs_exact=True,original_median_ms=float(np.median(original[2:])),batched_median_ms=float(np.median(batched[2:])),paired_median_speedup=float(np.median(original[2:]/batched[2:])),timing_includes_GPU_CPU_transfers_and_full_classifier=True,alternating_order=True,rows=proof.rows,source_runtime_unchanged=True,short_diagnostic_not_success_evaluation=True)
    (a.out/'training_status.json').write_text(json.dumps(report,indent=2)+'\n');(a.record_root/'v9_GPU_foot_same_state_proof.json').write_text(json.dumps(report,indent=2)+'\n');app.close()
if __name__=='__main__':main()
