"""Read-only reuse of the historical W2 standing runner, outputs redirected.

No original source/cache is written: bytecode is disabled by the launcher and
the robot converter's destination is replaced before constructing the scene.
"""
import argparse,json,sys
from pathlib import Path
from .spec import W8,OUTPUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=32);parser.add_argument('--device',default='cuda:0');parser.add_argument('--kit_args',default='')
    a=parser.parse_args()
    sys.path.insert(0,str(W8/'stand_probe/legmm_stand_probe'))
    sys.argv=['isaac_dr_probe.py','--part','reference_readonly','--out',str(a.out),
        '--pos-iters','4','--vel-iters','0','--device',a.device,'--kit_args',a.kit_args]
    import isaac_dr_probe as w2
    import numpy as np
    from isaaclab.sim import SimulationContext
    original=w2.robot_cfg
    def redirected():
        cfg=original();cfg.spawn.usd_dir=str(OUTPUT/'generated_usd_w2_exact_reference');return cfg
    w2.robot_cfg=redirected
    try:
        p=w2.Probe(a.num_envs)
        p.reset(seed=1,noise=.01,q_arm=w2.Q_STOW_NEW)
        prm=w2.table([dict(cond={},payload=.25) for _ in range(a.num_envs)],np.random.default_rng(12345))
        p.apply_dr(prm)
        w2.settle(p)
        initial=p.snapshot();obs_initial=p.obs().cpu().tolist()
        heights=[]
        for _ in range(750):
            p.step(w2.stow_t(a.num_envs));heights.append(p.snapshot()['pos'][:,2].tolist())
        final=p.snapshot();robot=p.robot
        result=dict(mode='w2-reference',diagnostic_only=True,num_envs=a.num_envs,physical_steps=p.step_count*2+1,
            base_fixed=bool(robot.is_fixed_base),initial_base=initial['pos'].tolist(),final_base=final['pos'].tolist(),
            height_series=heights,initial_obs=obs_initial,final_obs=p.obs().cpu().tolist(),joint_names=robot.joint_names,
            joint_stiffness=robot.data.joint_stiffness.cpu().tolist(),joint_damping=robot.data.joint_damping.cpu().tolist(),
            joint_armature=robot.data.joint_armature.cpu().tolist(),robot_masses=robot.root_physx_view.get_masses().cpu().tolist(),
            robot_inertias=robot.root_physx_view.get_inertias().cpu().tolist(),robot_coms=robot.root_physx_view.get_coms().cpu().tolist(),
            robot_materials=robot.root_physx_view.get_material_properties().cpu().tolist(),
            episodes=[dict(seed=i,success=False,reason='diagnostic_stand',fell=bool(p.fell[i])) for i in range(a.num_envs)])
        (a.out/'result.json').write_text(json.dumps(result,indent=2))
    except BaseException as error:
        import traceback
        (a.out/'error.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc())))
        raise
    finally:
        SimulationContext.clear_instance();w2.app.close(wait_for_replicator=False)


if __name__=='__main__':main()
