"""Explicit opt-in W8 scene/standing/teacher physical probe; no student training."""
import argparse,json,math,os,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .spec import OUTPUT,W8,STOW,PHYSICS_DT,PlannerConfig,STEP_LENGTH,InitialState,gate_diagnostics,validate_reused_sampling


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--mode',choices=['stand','teacher','contact','self-contact'],default='stand')
    parser.add_argument('--audit-no-gravity-compensation',action='store_true')
    parser.add_argument('--audit-w2-model',action='store_true')
    parser.add_argument('--audit-yaw-zero',action='store_true')
    parser.add_argument('--audit-payload',type=float)
    parser.add_argument('--audit-w2-self-collision-off',action='store_true')
    parser.add_argument('--audit-reset-noise',action='store_true')
    parser.add_argument('--audit-w2-robot-material',action='store_true')
    parser.add_argument('--audit-scene-solver-4-0',action='store_true')
    parser.add_argument('--audit-scene-solver-unclamped',action='store_true')
    parser.add_argument('--residual-velocity',action='store_true')
    parser.add_argument('--sampling',type=Path)
    parser.add_argument('--trace-teacher',action='store_true')
    parser.add_argument('--closed-loop-teacher',action='store_true')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();args.headless=True
    if (args.audit_no_gravity_compensation or args.audit_w2_model or args.audit_yaw_zero or args.audit_payload is not None) and args.mode!='stand':
        raise ValueError('Control-variable variants are standing diagnostics only')
    if (args.audit_w2_self_collision_off or args.audit_reset_noise or args.audit_w2_robot_material) and (args.mode!='stand' or not args.audit_w2_model):
        raise ValueError('Historical reset/self-collision controls require W2 standing diagnostics')
    if args.audit_scene_solver_4_0 and args.mode!='stand':raise ValueError('Solver control is standing diagnostic only')
    if args.audit_scene_solver_unclamped and (args.mode!='stand' or args.audit_scene_solver_4_0):
        raise ValueError('Legacy solver comparison is standing diagnostic only')
    from .assets import build_robot
    from .geometry import GeometryPlanner,sample_reachable
    model=build_robot(OUTPUT/'models'); planner=GeometryPlanner(model)
    if args.closed_loop_teacher:planner.cache_models=True
    states=[];nominal_plans=[];rejects=[]
    # This is a diagnostic candidate-selection height from W2, not a live pose.
    # Every candidate is rechecked against the actual pose after free settling.
    def nominal(s): return [*s.base_xy,.3763,math.cos(s.base_yaw/2),0,0,math.sin(s.base_yaw/2)]
    if args.sampling:
        import hashlib
        if not args.sampling.resolve().is_relative_to(OUTPUT): raise ValueError('Only this W8 task sampling records may be reused')
        raw=args.sampling.read_bytes();rejects=json.loads(raw)
        states=[InitialState(**r['state']) for r in rejects if r['accepted']]
        validate_reused_sampling(states,args.seed,args.num_envs,args.residual_velocity)
        (args.out/'sampling_source.json').write_text(json.dumps(dict(path=str(args.sampling),sha256=hashlib.sha256(raw).hexdigest())))
        print('Reused exact accepted initial states: '+str(args.sampling),flush=True)
    else:
        for seed in range(args.seed,args.seed+args.num_envs):
            def record(s,p): rejects.append(dict(state=s.json(),accepted=p.accepted,reason=p.reason))
            s,p=sample_reachable(seed,planner,nominal,record,args.residual_velocity)
            states.append(s);nominal_plans.append(p)
            print(json.dumps(dict(phase='cpu_sampling',seed=seed,attempt=s.attempt,shape=s.shape)),flush=True)
    (args.out/'sampling.json').write_text(json.dumps(rejects,indent=1))
    print('Full regression runs independently under w8_parallel_regression_20261114.md',flush=True)
    app=AppLauncher(args).app
    import torch
    import isaaclab.sim as sim
    import isaaclab.utils.math as maths
    from isaaclab.scene import InteractiveScene
    from .isaac_scene import make_scene,LEG_ORDER,HOME_Q,DOG_BODIES,ARM_BODIES
    from .outcome import EpisodeOutcome,lowest_point
    from .collision import excluded_pair
    try:
        simulation_cfg=sim.SimulationCfg(dt=PHYSICS_DT,device=args.device,render_interval=1000)
        if not args.audit_scene_solver_unclamped:
            from .physics_config import apply_task_solver_limits
            apply_task_solver_limits(simulation_cfg)
        simulation=sim.SimulationContext(simulation_cfg)
        scene_cfg=make_scene(states,model,legacy_rigid_solver=args.audit_scene_solver_unclamped)
        if args.audit_w2_model:
            scene_cfg.robot.spawn.asset_path=str(W8/'stand_probe/model/a2_arm.xml')
            scene_cfg.robot.spawn.usd_dir=str(OUTPUT/'generated_usd_w2_control_audit')
            # W2 deliberately had no arm colliders. Do not attach sensors to
            # collider-free links or add geometry to the control model.
            for body in ARM_BODIES:setattr(scene_cfg,'contact_'+body,None)
        if args.audit_w2_self_collision_off:
            scene_cfg.robot.spawn.self_collision=False
            scene_cfg.robot.spawn.usd_dir=str(OUTPUT/'generated_usd_w2_self_off_audit')
        scene=InteractiveScene(scene_cfg);simulation.reset()
        robot,target,table=scene['robot'],scene['target'],scene['table']
        device=robot.device;n=len(states);origins=scene.env_origins
        leg_ids=robot.find_joints(LEG_ORDER,preserve_order=True)[0]
        arm_ids=robot.find_joints([f'joint{i}' for i in range(1,7)],preserve_order=True)[0]
        finger_ids=robot.find_joints(['endleft','endright'],preserve_order=True)[0]
        tensor=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
        root=robot.data.default_root_state.clone()
        for i,s in enumerate(states):
            root[i,:3]=origins[i]+tensor([*s.base_xy,.4])
            root[i,3:7]=tensor([math.cos(s.base_yaw/2),0,0,math.sin(s.base_yaw/2)])
        if args.audit_yaw_zero:root[:,3:7]=tensor([1,0,0,0])
        robot.write_root_state_to_sim(root)
        if args.audit_w2_robot_material:
            material=robot.root_physx_view.get_material_properties().clone()
            material[:,:,0]=1.;material[:,:,1]=1.;material[:,:,2]=0.
            robot.root_physx_view.set_material_properties(material,torch.arange(n,dtype=torch.int32))
        if args.audit_payload is not None:
            masses=robot.root_physx_view.get_masses().clone()
            masses[:,robot.find_bodies('payload')[0][0]]=args.audit_payload
            robot.root_physx_view.set_masses(masses,torch.arange(n,dtype=torch.int32))
        q=robot.data.default_joint_pos.clone()
        if args.audit_reset_noise:
            q[:,leg_ids]+=tensor(np.random.default_rng(1).uniform(-.01,.01,(n,12)))
        robot.write_joint_state_to_sim(q,torch.zeros_like(q))
        table_state=table.data.default_root_state.clone(); object_state=target.data.default_root_state.clone()
        for i,s in enumerate(states):
            table_state[i,:3]=origins[i]+tensor([0,0,s.support/2])
            object_state[i,:3]=origins[i]+tensor(s.center)
            object_state[i,3:7]=tensor([math.cos(s.object_yaw/2),0,0,math.sin(s.object_yaw/2)])
        table.write_root_state_to_sim(table_state);target.write_root_state_to_sim(object_state)
        policy=torch.jit.load(str(W8/'stand_probe/from_mac/policy.pt'),map_location=device).eval()
        default=tensor(HOME_Q);last=torch.zeros(n,12,device=device);phase=torch.zeros(n,device=device)
        cmd=torch.zeros(n,3,device=device);up=tensor([0,0,1]).expand(n,3)
        native=tensor([*STOW,.072]).expand(n,7).clone()
        queue=native[None].repeat(11,1,1);ids=torch.arange(n,device=device)
        tick=0;peak_torque=torch.zeros(6,device=device)
        outcomes=[EpisodeOutcome(s.support) for s in states]
        event_counts=[{} for _ in states];feet_peak=0.;finger_peak=0.;initial=None
        contact_peaks=[{} for _ in states]
        safety_stop=None
        trajectory_peak=np.zeros(n);min_height=np.full(n,np.inf);min_upz=np.full(n,np.inf)
        def step(command,measure=False):
            nonlocal tick,last,peak_torque,feet_peak,finger_peak,safety_stop
            d=robot.data
            if tick%2==0:
                feet=torch.stack([phase,(phase+.5)%1,(phase+.5)%1,phase],1)
                obs=torch.cat([d.root_ang_vel_b,-maths.quat_apply(d.root_link_quat_w,up),
                    d.joint_pos[:,leg_ids]-default,d.joint_vel[:,leg_ids],last,cmd,feet],1)
                with torch.no_grad(): last=policy(obs)
                q[:,leg_ids]=default+.25*last
            queue[tick%11]=command
            delayed=queue[(tick-10)%11]
            q[:,arm_ids]=delayed[:,:6]
            q[:,finger_ids]=delayed[:,6,None]*tensor([.5,-.5])
            robot.set_joint_position_target(q)
            gravity=robot.root_physx_view.get_gravity_compensation_forces()[:,6+torch.tensor(arm_ids,device=device)]
            if args.audit_no_gravity_compensation:gravity=torch.zeros_like(gravity)
            robot.set_joint_effort_target(gravity,joint_ids=arm_ids)
            scene.write_data_to_sim();simulation.step(render=False);scene.update(PHYSICS_DT);tick+=1
            peak_torque=torch.maximum(peak_torque,robot.data.applied_torque[:,arm_ids].abs().amax(0))
            if not torch.isfinite(robot.data.joint_pos).all(): raise FloatingPointError('Nonfinite robot joints')
            if measure:
                position=target.data.root_pos_w-origins
                rotations=maths.matrix_from_quat(target.data.root_quat_w).cpu().numpy()
                heights=robot.data.root_link_pos_w[:,2]
                upz=maths.quat_apply(robot.data.root_link_quat_w,up)[:,2]
                if bool(((heights<.2)|(upz<=.5)).any()):
                    safety_stop='base_fall'
                displacement=np.linalg.norm((robot.data.root_link_pos_w-origins).cpu().numpy()-initial,axis=1)
                np.maximum(trajectory_peak,displacement,out=trajectory_peak)
                np.minimum(min_height,heights.cpu().numpy(),out=min_height)
                np.minimum(min_upz,upz.cpu().numpy(),out=min_upz)
                bad=[set() for _ in states];touch=[[False,False] for _ in states]
                for body in DOG_BODIES+ARM_BODIES:
                    if args.audit_w2_model and body in ARM_BODIES:continue
                    sensor=scene['contact_'+body]
                    force=sensor.data.force_matrix_w[:,0].norm(dim=-1).cpu().numpy()
                    points=sensor.data.contact_pos_w[:,0].cpu().numpy()
                    for i in range(n):
                        for j in np.flatnonzero(force[i]>1.):
                            if j==1 and body in ('left','right'):
                                touch[i][body=='right']=True;finger_peak=max(finger_peak,float(force[i,j]));continue
                            if j==2 and body.endswith('_calf'):
                                # The calf carries the foot collider. Only actual floor contacts
                                # at the foot's sphere are legal, not arbitrary calf hits.
                                bid=robot.find_bodies(body)[0][0]
                                foot=robot.data.body_pos_w[i,bid]+maths.quat_apply(robot.data.body_quat_w[i,bid][None],tensor([[0,0,-.275]]))[0]
                                if np.linalg.norm(points[i,j]-foot.cpu().numpy()) <= .04:
                                    feet_peak=max(feet_peak,float(force[i,j]));continue
                            filters=['table','object','floor']
                            if body in ARM_BODIES:
                                filters+=DOG_BODIES+[b for b in ['arm_base']+ARM_BODIES
                                    if b!=body and not excluded_pair(body,b,DOG_BODIES)]
                            name=f'{body}:'+filters[j]
                            bad[i].add(name);event_counts[i][name]=event_counts[i].get(name,0)+1
                            contact_peaks[i][name]=max(contact_peaks[i].get(name,0.),float(force[i,j]))
                for i,o in enumerate(outcomes):
                    bottom=lowest_point(float(position[i,2]),rotations[i],states[i].shape,states[i].size)
                    o.update(bottom,float(target.data.root_lin_vel_w[i,2]),touch[i],bad[i],bool(heights[i]<.2 or upz[i]<=.5))
        # W2 controller acts from the very first physical step. Root writes above
        # occur only at episode reset; no body/leg/controller resets inside grasp.
        for _ in range(300): step(native)
        initial=(robot.data.root_link_pos_w-origins).cpu().numpy().copy()
        actual_pose=torch.cat((robot.data.root_link_pos_w-origins,robot.data.root_link_quat_w),1).cpu().numpy()
        live_plans=[];gate_reports=[];shoulders=[]
        actual_velocity=robot.data.root_vel_w.cpu().numpy().copy()
        object_pose=torch.cat((target.data.root_pos_w-origins,target.data.root_quat_w),1).cpu().numpy().copy()
        joint_positions=robot.data.joint_pos.cpu().numpy().copy()
        feet=torch.stack([phase,(phase+.5)%1,(phase+.5)%1,phase],1)
        standing_obs=torch.cat([robot.data.root_ang_vel_b,-maths.quat_apply(robot.data.root_link_quat_w,up),
            robot.data.joint_pos[:,leg_ids]-default,robot.data.joint_vel[:,leg_ids],last,cmd,feet],1)
        adapter_readback=dict(joint_names=robot.joint_names,initial_obs=standing_obs.cpu().tolist(),
            joint_stiffness=robot.data.joint_stiffness.cpu().tolist(),joint_damping=robot.data.joint_damping.cpu().tolist(),
            joint_armature=robot.data.joint_armature.cpu().tolist(),robot_masses=robot.root_physx_view.get_masses().cpu().tolist(),
            robot_inertias=robot.root_physx_view.get_inertias().cpu().tolist(),robot_coms=robot.root_physx_view.get_coms().cpu().tolist(),
            robot_materials=robot.root_physx_view.get_material_properties().cpu().tolist())
        (args.out/'standing_adapter_readback.json').write_text(json.dumps(adapter_readback,indent=2))
        joint_states=[dict(zip(robot.joint_names,row.tolist())) for row in joint_positions]
        if args.mode=='contact':
            # Explicit positive self-contact diagnostic, NEVER validation or data
            # for distillation. PhysX collisions and failure verdicts stay active.
            # Intentionally execute the predicted arm_self path to measure the
            # disputed mount collision with the physical pair sensor.
            checked_collision=planner.collision
            def contact_diagnostic_path(q,object_delta=None):
                reason=checked_collision(q,object_delta)
                return None if reason=='arm_self' else reason
            planner.collision=contact_diagnostic_path
        shoulder_id=robot.find_bodies('link2')[0][0]
        physx_shoulders=(robot.data.body_link_pos_w[:,shoulder_id]-origins).cpu().numpy().copy()
        for i,s in enumerate(states):
            velocity=actual_velocity[i,[0,1,5]]
            ow,ox,oy,oz=object_pose[i,3:7]
            object_yaw=math.atan2(2*(ow*oz+ox*oy),1-2*(oy*oy+oz*oz))
            from dataclasses import replace
            live_state=replace(s,object_xy=tuple(object_pose[i,:2]),object_yaw=object_yaw)
            planner.configure(live_state,actual_pose[i],object_pose[i],joint_states[i])
            shoulder=planner.d.xpos[planner.m.body('link2').id].copy()
            live_plans.append(planner.plan(live_state,actual_pose[i],velocity,object_pose[i],joint_states[i]))
            shoulders.append(shoulder.tolist())
            w,x,y,z=actual_pose[i,3:7]
            yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            gate_reports.append(gate_diagnostics(live_state,shoulder[:2],actual_pose[i,:2],yaw,velocity,object_pose[i,:3]))
        from .physics_audit import physics_audit
        audit=physics_audit(simulation,robot,target,table,states)
        (args.out/'physics_audit.json').write_text(json.dumps(audit,indent=2))
        injected_velocity_readback=None
        if args.residual_velocity:
            velocity=robot.data.root_vel_w.clone()
            velocity[:,:2]=tensor([s.velocity[:2] for s in states]);velocity[:,5]=tensor([s.velocity[2] for s in states])
            robot.write_root_velocity_to_sim(velocity)  # reset boundary only
            injected_velocity_readback=robot.data.root_vel_w.cpu().tolist()
        trajectories=[];schedules=[]
        from .control import teacher_commands
        for i,p in enumerate(live_plans):
            schedule=[]
            points=[];previous=np.r_[joint_positions[i,arm_ids],joint_positions[i,finger_ids[0]]-joint_positions[i,finger_ids[1]]];closed=False
            if p.accepted and args.mode in ('teacher','contact'):
                points=teacher_commands(p,previous,schedule).tolist()
            trajectories.append(points or [previous]);schedules.append(schedule)
        if args.mode=='self-contact':
            # Dedicated physical positive control from the archived deterministic
            # MuJoCo contact target. Not a grasp planner or a validation episode.
            target_q=np.array([-.48680862635107347,-1.8594969666314412,-.087,
                1.2793502182800909,1.76,-1.468057760151292,.072])
            for i in range(n):
                start=np.asarray(trajectories[i][0]);count=int(np.ceil(np.max(abs(target_q-start)/STEP_LENGTH)))
                trajectories[i]=[start.copy()]*25+[start+(target_q-start)*j/count for j in range(1,count+1)]+[target_q.copy()]*100
                schedules[i]=[dict(phase='negative_stow' if j<25 else 'self_contact_positive') for j in range(len(trajectories[i]))]
        trace=None
        if args.trace_teacher:
            from .diagnosis import TeacherTrace
            trace=TeacherTrace(args.out,robot,target,origins,states,arm_ids,finger_ids)
            (args.out/'teacher_plans.json').write_text(json.dumps([asdict(p) for p in live_plans],default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))
        if args.closed_loop_teacher:
            if args.mode!='teacher':raise ValueError('Closed-loop teacher requires teacher mode')
            from .closed_loop import ClosedLoopTeacher,MeasuredState
            teachers=[ClosedLoopTeacher(planner,s) for s in states]
            schedules=[[] for _ in states]
            live_labels=None
            last_command=native.clone()
            label_log=(args.out/'closed_loop_labels.jsonl').open('x')
        for frame in range(1500):
            if args.closed_loop_teacher:
                if frame%16==0:
                    base=torch.cat((robot.data.root_link_pos_w-origins,robot.data.root_link_quat_w),1).cpu().numpy()
                    objects=torch.cat((target.data.root_pos_w-origins,target.data.root_quat_w),1).cpu().numpy()
                    joints=robot.data.joint_pos.cpu().numpy()
                    velocities=robot.data.root_vel_w.cpu().numpy()
                    live_labels=[]
                    for i,teacher in enumerate(teachers):
                        touches=tuple(float(scene['contact_'+b].data.force_matrix_w[i,0,1].norm())>1. for b in ('left','right'))
                        measured=MeasuredState(base[i],objects[i],dict(zip(robot.joint_names,joints[i])),
                            velocities[i,[0,1,5]],last_command[i].cpu().numpy(),queue[:,i].cpu().numpy(),touches,
                            frame*PHYSICS_DT,any(outcomes[i].reason and outcomes[i].reason.startswith(x) for x in ('collision','fall','drop')))
                        label,metadata=teacher.label(measured)
                        live_labels.append(label)
                        schedules[i].extend([dict(phase=metadata['phase'],command_time_s=frame*PHYSICS_DT)]*4)
                        label_log.write(json.dumps(dict(seed=states[i].seed,frame=frame,measured_object_pose=objects[i].tolist(),
                            measured_arm=joints[i,arm_ids].tolist(),label=label.tolist(),metadata=metadata))+'\n')
                    label_log.flush()
                command=tensor(np.array(live_labels)[:,(frame%16)//4])
                last_command=command.clone()
            else:
                command=tensor([trajectories[i][min(frame//4,len(trajectories[i])-1)] for i in range(n)])
            step(command,measure=True)
            if trace and frame%4==0:trace.record(frame,command,scene,outcomes,schedules)
            if frame%100==0:
                (args.out/'heartbeat.json').write_text(json.dumps(dict(physical_steps=tick,
                    min_base_height=float(min_height.min()),min_up_z=float(min_upz.min()),
                    safety_stop=safety_stop,finished=sum(o.reason is not None for o in outcomes))))
            if safety_stop is not None: break
            if args.mode in ('teacher','contact') and all(o.reason is not None for o in outcomes): break
        final=(robot.data.root_link_pos_w-origins).cpu().numpy()
        result=dict(mode=args.mode,diagnostic_only=args.mode in ('contact','self-contact') or args.audit_no_gravity_compensation or args.audit_w2_model or args.audit_yaw_zero or args.audit_payload is not None or args.audit_scene_solver_4_0 or args.audit_scene_solver_unclamped,
            audit_no_gravity_compensation=args.audit_no_gravity_compensation,audit_w2_model=args.audit_w2_model,
            audit_yaw_zero=args.audit_yaw_zero,audit_payload=args.audit_payload,
            audit_w2_self_collision_off=args.audit_w2_self_collision_off,audit_reset_noise=args.audit_reset_noise,
            audit_w2_robot_material=args.audit_w2_robot_material,
            audit_scene_solver_4_0=args.audit_scene_solver_4_0,
            audit_scene_solver_unclamped=args.audit_scene_solver_unclamped,
            closed_loop_teacher=args.closed_loop_teacher,
            safety_stop=safety_stop,residual_velocity=args.residual_velocity,
            injected_reset_velocity=[list(s.velocity) for s in states] if args.residual_velocity else None,
            injected_velocity_readback=injected_velocity_readback,
            gate_measurement_phase="settled_before_optional_reset_velocity_injection",seed=args.seed,num_envs=n,physical_steps=tick,elapsed_sim_s=tick*PHYSICS_DT,
            base_fixed=bool(robot.is_fixed_base),solver=[4,0],initial_base=initial.tolist(),final_base=final.tolist(),
            initial_base_pose=actual_pose.tolist(),initial_base_velocity=actual_velocity.tolist(),
            initial_object_pose=object_pose.tolist(),geometry_shoulders=shoulders,gate_reports=gate_reports,
            physx_shoulders=physx_shoulders.tolist(),initial_joint_positions=joint_positions.tolist(),
            joint_names=robot.joint_names,trajectory_control_points=[len(p) for p in trajectories],
            forbidden_contact_peaks=contact_peaks,
            trajectory_max_base_displacement=trajectory_peak.tolist(),trajectory_min_base_height=min_height.tolist(),
            trajectory_min_up_z=min_upz.tolist(),
            max_base_displacement=float(np.linalg.norm(final-initial,axis=1).max()),peak_arm_applied_torque=peak_torque.cpu().tolist(),
            foot_force_peak=feet_peak,finger_force_peak=finger_peak,object_masses=target.root_physx_view.get_masses().cpu().tolist(),
            object_inertias=target.root_physx_view.get_inertias().cpu().tolist(),states=[s.json() for s in states],
            live_geometry=[dict(accepted=p.accepted,reason=p.reason) for p in live_plans],
            episodes=[asdict(o)|dict(seed=s.seed,shape=s.shape,contacts=contacts) for o,s,contacts in zip(outcomes,states,event_counts)])
        (args.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k in ('mode','num_envs','physical_steps','max_base_displacement','base_fixed','foot_force_peak','finger_force_peak')}),flush=True)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        (args.out/'error.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc()),indent=2))
        raise
    finally:
        sim.SimulationContext.clear_instance();app.close(wait_for_replicator=False)

if __name__=='__main__': main()
