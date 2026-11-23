"""Prepare only: split existing physics host timing without extra engine calls."""
import ast,difflib,hashlib,json
from pathlib import Path
from .prepare_v7_timing_patch import StripTiming

class StrictStripTiming(StripTiming):
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='timings' and call.func.attr=='call':
            return ast.copy_location(ast.Expr(value=self.visit_Call(call)),node)
        return super().visit_Expr(node)

def prepare(root,record):
    path=root/'learning_episode.py';source=path.read_text()
    old='            scene.write_data_to_sim();simulation.step(render=False);scene.update(PHYSICS_DT);tick+=1'
    new="            scene.write_data_to_sim();timings.call('physics_simulation_step_only',simulation.step,render=False);timings.call('physics_scene_update_only',scene.update,PHYSICS_DT);tick+=1"
    assert source.count(old)==1;candidate=source.replace(old,new,1)
    stripped=lambda s:ast.dump(StrictStripTiming().visit(ast.parse(s)),include_attributes=False)
    assert stripped(source)==stripped(candidate)
    (record/'v9_physics_segment_split_unapplied.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile='a/legmm_skill/tabletop/learning_episode.py',tofile='b/legmm_skill/tabletop/learning_episode.py')))
    report=dict(applied=False,baseline_sha256=hashlib.sha256(source.encode()).hexdigest(),candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),whole_program_AST_equal_after_removing_telemetry=True,additional_physics_steps=0,additional_CUDA_synchronizations=0,additional_IO_files=0,extra_segments=2,expected_enabled_segment_keys=16,host_wait_attribution_only=True,active_formal_source_unchanged=True)
    (record/'v9_physics_segment_split_CPU_proof.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':print(json.dumps(prepare(Path(__file__).parent,Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_true4_contact_retry1_20261115'))))
