"""Prepare an unapplied telemetry-only patch for a later consistent boundary."""
import ast,difflib,hashlib,json
from pathlib import Path


def revised_sources(root):
    result={};path=root/'learning_episode.py';old=path.read_text();s=old
    s=s.replace('    from .lifecycle_cleanup import check_stop,stop_requested,clear_planner_cache,trim_unused',
        "    from .segment_timing import SegmentTimings\n    timings=SegmentTimings(enabled=bool(getattr(args,'audit_timing',False)))\n    _timing_cohort=timings.begin('cohort_wall_including_setup_cleanup')\n    learning.segment_timings=timings\n    if hasattr(learning,'visual'):learning.visual.segment_timings=timings\n    from .lifecycle_cleanup import check_stop,stop_requested,clear_planner_cache,trim_unused",1)
    s=s.replace('            nonlocal tick,last,peak_torque,feet_peak,finger_peak,safety_stop','            nonlocal tick,last,peak_torque,feet_peak,finger_peak,safety_stop\n            _timing_control=timings.begin(\'standing_and_targets\')',1)
    s=s.replace('            scene.write_data_to_sim();simulation.step(render=False);scene.update(PHYSICS_DT);tick+=1',"            timings.end('standing_and_targets',_timing_control)\n            _timing_physics=timings.begin('physics_step')\n            scene.write_data_to_sim();simulation.step(render=False);scene.update(PHYSICS_DT);tick+=1\n            timings.end('physics_step',_timing_physics)\n            _timing_contacts=timings.begin('contact_outcome_readback')",1)
    needle="                    o.update(bottom,float(target.data.root_lin_vel_w[i,2]),touch[i],bad[i],bool(heights[i]<.2 or upz[i]<=.5))"
    s=s.replace(needle,needle+"\n            timings.end('contact_outcome_readback',_timing_contacts)",1)
    needle='                    base=torch.cat((robot.data.root_link_pos_w-origins,robot.data.root_link_quat_w),1).cpu().numpy()'
    s=s.replace(needle,"                    _timing_state=timings.begin('teacher_global_readback')\n"+needle,1)
    needle='                    velocities=robot.data.root_vel_w.cpu().numpy()'
    s=s.replace(needle,needle+"\n                    timings.end('teacher_global_readback',_timing_state)",1)
    needle="                        touches=tuple(float(scene['contact_'+b].data.force_matrix_w[i,0,1].norm())>1. for b in ('left','right'))"
    s=s.replace(needle,"                        _timing_label=timings.begin('teacher_label_and_readback')\n"+needle,1)
    needle='                        label,metadata=teacher.label(measured)'
    s=s.replace(needle,needle+"\n                        timings.end('teacher_label_and_readback',_timing_label)",1)
    needle="                        label_log.write(json.dumps(dict(seed=states[i].seed,frame=frame,measured_object_pose=objects[i].tolist(),"
    s=s.replace(needle,"                        _timing_io=timings.begin('label_io')\n"+needle,1)
    needle="                            measured_arm=joints[i,arm_ids].tolist(),label=label.tolist(),metadata=metadata))+'\\n')"
    replacement="                            measured_arm=joints[i,arm_ids].tolist(),label=label.tolist(),metadata=metadata,**(dict(actual_previous_command=measured.last_command.tolist(),queue_tick_next=tick) if timings.enabled and i<4 else {})))+'\\n')\n                        timings.end('label_io',_timing_io)"
    assert needle in s;s=s.replace(needle,replacement,1)
    needle='                    observation_engine_before=engine_steps.copy()'
    s=s.replace(needle,"                    _timing_render=timings.begin('camera_render_and_guard')\n"+needle,1)
    needle='                    live_labels=learning.decision(scene,robot,joints,arm_ids,finger_ids,last_command,'
    s=s.replace(needle,"                    timings.end('camera_render_and_guard',_timing_render)\n"+needle,1)
    needle="        (args.out/'cohort_cleanup.json').write_text(json.dumps(cleanup,indent=2))"
    assert needle in s;s=s.replace(needle,needle+"\n        timings.end('cohort_wall_including_setup_cleanup',_timing_cohort)\n        timings.write(args.out/'segment_timings.json')",1)
    foot_check='if np.linalg.norm(points[i,j]-foot.cpu().numpy()) <= .04:'
    assert foot_check in s
    s=s.replace(foot_check,"if timings.call('foot_cpu_readback_distance',lambda:np.linalg.norm(points[i,j]-foot.cpu().numpy())) <= .04:",1)
    result[path.name]=(old,s)
    path=root/'learning.py';old=path.read_text();s=old.replace('from .visual import DualDepth','from .visual import DualDepth\nfrom .segment_timing import timer_for',1)
    s=s.replace('        images,rgb,wrist=self.visual.observe(scene)',"        timings=timer_for(self)\n        images,rgb,wrist=timings.call('visual_observe_total',self.visual.observe,scene)",1)
    s=s.replace('        if self.teacher_drives:',"        _timing_forward=timings.begin('student_forward')\n        if self.teacher_drives:",1)
    s=s.replace('        # At least one real backward/update is part of a capacity short test.',"        timings.end('student_forward',_timing_forward)\n        _timing_io=timings.begin('decision_io')\n        # At least one real backward/update is part of a capacity short test.",1)
    s=s.replace('        return actions',"        timings.end('decision_io',_timing_io)\n        return actions",1)
    s=s.replace('        images,_,_=self.visual.observe(scene)',"        timings=timer_for(self)\n        images,_,_=timings.call('visual_observe_total',self.visual.observe,scene)",1)
    s=s.replace('        return self.student.act(images,proprio,last_command).cpu().numpy()',"        return timings.call('student_forward',lambda:self.student.act(images,proprio,last_command).cpu().numpy())",1)
    result[path.name]=(old,s)
    path=root/'visual.py';old=path.read_text();s=old.replace('from airbot_rl.depth import encode_wrist_depth','from airbot_rl.depth import encode_wrist_depth\nfrom .segment_timing import timer_for',1)
    needle='        images=torch.stack((head_depth(self.depth,rgb,batch=8).to(self.device),encode_wrist_depth(wrist)),1)'
    replacement="        timings=timer_for(self)\n        images=torch.stack((timings.call('da2_head',head_depth,self.depth,rgb,batch=8).to(self.device),timings.call('wrist_preprocess',encode_wrist_depth,wrist)),1)"
    assert needle in s;s=s.replace(needle,replacement,1);result[path.name]=(old,s)
    path=root/'train_dagger.py';old=path.read_text();s=old
    s=s.replace("    p.add_argument('--resume-sha256')","    p.add_argument('--resume-sha256')\n    p.add_argument('--audit-timing',action='store_true')",1)
    s=s.replace('def evaluate_fixed(session,app,out,qualification,record_root):','def evaluate_fixed(session,app,out,qualification,record_root,audit_timing=False):',1)
    s=s.replace("        args=episode_args(batch,len(data['states']),seed)","        args=episode_args(batch,len(data['states']),seed)\n        args.audit_timing=audit_timing",1)
    s=s.replace('                args=episode_args(out,a.num_envs,seed)','                args=episode_args(out,a.num_envs,seed)\n                args.audit_timing=a.audit_timing',1)
    s=s.replace('summary=evaluate_fixed(session,app,validation,a.qualification,a.record_root)','summary=evaluate_fixed(session,app,validation,a.qualification,a.record_root,audit_timing=a.audit_timing)',1)
    result[path.name]=(old,s)
    path=root/'learning_launch.py';old=path.read_text();s=old
    s=s.replace("    p.add_argument('--resume-sha256')","    p.add_argument('--resume-sha256')\n    p.add_argument('--audit-timing',action='store_true')",1)
    needle="            if inventory.get(name)!=current:raise ValueError('Visual capacity source changed: '+name)"
    replacement="            if inventory.get(name)!=current:\n                from .segment_timing import observer_source_compatible\n                if not observer_source_compatible(name,inventory.get(name),current,a.record_root):raise ValueError('Visual capacity source changed: '+name)"
    assert needle in s;s=s.replace(needle,replacement,1)
    s=s.replace("    if a.resume_state:command.extend(['--resume-state',str(a.resume_state)])","    if a.audit_timing:command.append('--audit-timing')\n    if a.resume_state:command.extend(['--resume-state',str(a.resume_state)])",1)
    result[path.name]=(old,s)
    for _,s in result.values():ast.parse(s)
    return result


class StripTiming(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        return None if node.module=='segment_timing' else node
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and (t.id.startswith('_timing_') or t.id=='timings') for t in node.targets):return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='timings' and node.func.attr=='call':
            function=node.args[1]
            if isinstance(function,ast.Lambda):return function.body
            return ast.Call(func=function,args=node.args[2:],keywords=node.keywords)
        return node
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='timings':return None
        return self.generic_visit(node)


def step_ast(source,strip=False):
    tree=ast.parse(source);node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')
    if strip:node=StripTiming().visit(node)
    return ast.dump(node,include_attributes=False)


def main():
    root=Path(__file__).parent;changes=revised_sources(root);target=root/'diagnostics';target.mkdir(exist_ok=True)
    patch=''.join(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/legmm_skill/tabletop/'+name,tofile='b/legmm_skill/tabletop/'+name)) for name,(old,new) in changes.items())
    assert step_ast(changes['learning_episode.py'][0])==step_ast(changes['learning_episode.py'][1],strip=True)
    for name in ('learning.py','visual.py'):
        old,new=changes[name]
        assert ast.dump(ast.parse(old),include_attributes=False)==ast.dump(StripTiming().visit(ast.parse(new)),include_attributes=False),name
    (target/'v7_segment_timing_unapplied.patch').write_text(patch)
    proof=dict(applied=False,enabled_by_default=False,active_process_untouched=True,qualified_physics_step_AST_unchanged_after_timing_removal=True,
        helper_sha256=hashlib.sha256((root/'segment_timing.py').read_bytes()).hexdigest(),
        functional_observation_and_policy_AST_unchanged=True,exact_previous_command_metadata_env_limit=4,
        files={name:dict(before_sha256=hashlib.sha256(old.encode()).hexdigest(),prepared_sha256=hashlib.sha256(new.encode()).hexdigest()) for name,(old,new) in changes.items()})
    (target/'v7_segment_timing_proof.json').write_text(json.dumps(proof,indent=2)+'\n')


if __name__=='__main__':main()
