"""Prepared v10 same-real-state outcome transport proof; launch only after formal exit."""
import argparse,ast,copy,json,os,time
from pathlib import Path
import numpy as np

class OutcomeProof:
    def __init__(self,out):
        from .outcome_readback_candidate import candidate_source,outcome_loop
        original=Path(__file__).with_name('learning_episode.py').read_text();candidate=candidate_source(original)
        assignments=[n for n in ast.walk(ast.parse(candidate)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('outcome_position_z','outcome_velocity_z','outcome_height','outcome_upz') for t in n.targets)]
        assert len(assignments)==4
        self.codes=[compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'real_GPU_outcome_transport','exec') for body in ([outcome_loop(original)],assignments+[outcome_loop(candidate)])]
        self.out=out;self.rows=[]
    def synthetic_priority(self,scope):
        # Explicit GPU tensor copies, not contacts or states produced by PhysX.
        import torch
        from types import SimpleNamespace
        from .outcome import EpisodeOutcome
        n=scope['n'];assert n>=13
        local=dict(scope);position=scope['position'].clone();velocity=torch.zeros_like(scope['target'].data.root_lin_vel_w)
        heights=torch.full_like(scope['heights'],.38);upz=torch.ones_like(scope['upz'])
        for i,state in enumerate(scope['states']):position[i,2]=state.support+.2
        hv=np.float32(.2);uv=np.float32(.5)
        for i,v in enumerate([np.nextafter(hv,-np.inf,dtype=np.float32),hv,np.nextafter(hv,np.inf,dtype=np.float32)]):heights[i]=float(v)
        for i,v in enumerate([np.nextafter(uv,-np.inf,dtype=np.float32),uv,np.nextafter(uv,np.inf,dtype=np.float32)],3):upz[i]=float(v)
        outcomes=[EpisodeOutcome(state.support,held_steps=199,lifted=True) for state in scope['states']]
        bad=[set() for _ in range(n)];touch=[[True,True] for _ in range(n)]
        bad[6]={'link6:table'};position[7,2]=scope['states'][7].support+.001;velocity[7,2]=-.2
        outcomes[9].reason='success';outcomes[9].success=True;bad[9]={'link6:object'}
        outcomes[10].reason='timeout';bad[10]={'link6:table'};touch[11]=[False,False]
        state=scope['states'][11];position[11,2]=state.support+.03-scope['lowest_point'](0.,scope['rotations'][11],state.shape,state.size)
        heights[12]=.1;bad[12]={'link6:table'}
        local.update(position=position,target=SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=velocity)),heights=heights,upz=upz,bad=bad,touch=touch)
        results=[]
        for code in self.codes:
            env=dict(local,outcomes=copy.deepcopy(outcomes));exec(code,env);results.append(env)
        a,b=results[0]['outcomes'],results[1]['outcomes'];assert a==b
        assert [o.reason=='base_fall' for o in a[:6]]==[True,False,False,True,True,False]
        assert a[6].reason.startswith('collision:') and a[7].reason=='drop' and a[8].success and a[9].success and a[10].reason=='timeout' and a[11].held_steps==0
        return dict(all_equal=True,device=str(heights.device),synthetic_tensor_cases_not_real_PhysX_contacts=True,thresholds_one_ULP_verified=True,first13_reasons=[o.reason for o in a[:13]],already_terminal_and_collision_drop_hold_success_priority_verified=True)

    def check(self,scope):
        tick=scope['tick']
        if tick not in (301,302,303,304,309,310,317,318,325,326,331,332):return
        import torch
        robot=scope['robot'];target=scope['target'];simulation=scope['simulation'];n=scope['n']
        assert scope['heights'].device.type=='cuda' and scope['position'].device.type=='cuda'
        state=[robot.root_physx_view.get_link_transforms().clone(),target.root_physx_view.get_transforms().clone(),robot.root_physx_view.get_link_velocities().clone(),target.root_physx_view.get_velocities().clone()]
        clock=(float(simulation.current_time),int(simulation.current_time_step_index),scope['engine_steps'].copy())
        results={};elapsed={};order=(0,1) if len(self.rows)%2==0 else (1,0)
        for method in order:
            local=dict(scope);local['outcomes']=copy.deepcopy(scope['outcomes'])
            torch.cuda.synchronize();start=time.perf_counter_ns();exec(self.codes[method],local);torch.cuda.synchronize();elapsed[method]=(time.perf_counter_ns()-start)/1e6;results[method]=local
        assert results[0]['outcomes']==results[1]['outcomes'],'Outcome state/priority mismatch'
        data={
            'outcome_position_z':lambda i:float(scope['position'][i,2]),
            'outcome_velocity_z':lambda i:float(target.data.root_lin_vel_w[i,2]),
            'outcome_height':lambda i:float(scope['heights'][i]),
            'outcome_upz':lambda i:float(scope['upz'][i])}
        for key,scalar in data.items():
            batch=results[1][key];assert batch.dtype==np.float32 and batch.shape==(n,)
            np.testing.assert_array_equal(batch,np.asarray([scalar(i) for i in range(n)],dtype=np.float32))
        if not hasattr(self,'synthetic_GPU_cases'):self.synthetic_GPU_cases=self.synthetic_priority(scope)
        after=[robot.root_physx_view.get_link_transforms(),target.root_physx_view.get_transforms(),robot.root_physx_view.get_link_velocities(),target.root_physx_view.get_velocities()]
        assert all(torch.equal(a,b) for a,b in zip(state,after))
        assert clock==(float(simulation.current_time),int(simulation.current_time_step_index),scope['engine_steps'])
        self.rows.append(dict(tick=tick,order=list(order),original_ms=elapsed[0],batched_ms=elapsed[1],four_arrays_bitwise_equal=True,all_64_outcome_states_equal=True,engine_and_world_state_unchanged=True,actual_bad_envs=sum(bool(v) for v in scope['bad']),actual_touch_envs=sum(any(v) for v in scope['touch']),already_terminal=sum(o.reason is not None for o in scope['outcomes'])))
        (self.out/'same_state_outcome_pairs.json').write_text(json.dumps(self.rows,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--record-root',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a,unused=p.parse_known_args();a.headless=True;a.enable_cameras=True
    preceding=json.loads((a.record_root/'v11_real_GPU_restore_proof.json').read_text());assert preceding['completed'] and preceding['model_updates']==4000 and preceding['chain_seen']==67776
    app=AppLauncher(a).app
    from .lifecycle_cleanup import install_stop_handler
    install_stop_handler()
    from .learning import DaggerSession
    from .train_dagger import episode_args
    from . import learning_episode
    proof=OutcomeProof(a.out);source=Path(learning_episode.__file__).read_text();needle='                for i,o in enumerate(outcomes):';assert source.count(needle)==1
    source=source.replace(needle,'                GPU_OUTCOME_PROOF.check(dict(dict(globals(),**locals()),engine_steps=engine_steps))\n'+needle)
    ns=dict(learning_episode.__dict__,GPU_OUTCOME_PROOF=proof);exec(compile(source,'diagnostic_same_state_outcome_runtime','exec'),ns)
    session=DaggerSession('cuda:0',640,capacity=4);session.budget=128;session.shadow_student=True
    episode=a.out/'diagnostic_train_seed640';episode.mkdir();args=episode_args(episode,64,640);args.audit_batched_foot_readback=True
    args.sampling=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_1_random_reachable_20261114/dagger_true4_visual64_retry1_resume_01/round_01_episode_0006/sampling.json')
    result=ns['run_episode'](args,app,session);assert len(proof.rows)==12 and not result['safety_stop']
    from .physics_config import verify_task_physics_audit
    verify_task_physics_audit(json.loads((episode/'physics_audit.json').read_text()))
    original=np.array([r['original_ms'] for r in proof.rows]);batch=np.array([r['batched_ms'] for r in proof.rows])
    report=dict(PID=os.getpid(),completed=True,diagnostic_only=True,formal_training=False,formal_env_decisions_added=0,optimizer_updates_added=0,num_envs=64,actual_diagnostic_env_decisions=session.seen,physical_steps=result['physical_steps'],engine_steps=result['engine_controlled_steps'],same_state_pairs=12,all_pairs_exact=True,original_median_ms=float(np.median(original[2:])),batched_median_ms=float(np.median(batch[2:])),paired_median_speedup=float(np.median(original[2:]/batch[2:])),timing_includes_four_GPU_CPU_transfers_and_full_outcome_loop=True,alternating_order=True,rows=proof.rows,synthetic_GPU_priority=proof.synthetic_GPU_cases,source_runtime_unchanged=True,uncovered_actual_GPU_cases='Early real train states may not contain collision/drop/hold-success/already-terminal/base-fall threshold negatives; actual counts shown per pair. CPU multi-branch one-ULP tests cover those state-machine cases.',short_diagnostic_not_success_evaluation=True)
    (a.out/'training_status.json').write_text(json.dumps(report,indent=2)+'\n');(a.record_root/'v11_GPU_outcome_same_state_proof.json').write_text(json.dumps(report,indent=2)+'\n');app.close()
if __name__=='__main__':main()
