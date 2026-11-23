"""Create a new frozen deliverable only after all paired independent tests exist."""
import argparse,hashlib,io,json
from pathlib import Path
import torch
from .spec import OUTPUT,WORKTREE,LOWER,UPPER,STEP_LENGTH,RESIDUAL_SCALE
from .storage import StorageGuard
from .student import Student


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(student,residual,test_report,visual_manifest,out):
    report=json.loads(test_report.read_text())
    if not report.get('completed') or not report.get('precommitted_initial_states_sha256'):
        raise ValueError('Completed paired independent test and frozen states required')
    for name in ('teacher','student','residual_ppo'):
        rows=report['drivers'][name]['episodes']
        if sorted(row['seed'] for row in rows)!=list(range(20000,20256)):
            raise ValueError('Independent test seed set differs: '+name)
        narrow=report['drivers'][name]['narrow_audit']
        if narrow['over15_degrees'] or narrow['approach_over15_degrees'] or narrow['pose_evidence_count']!=narrow['successful_boxes']:
            raise ValueError('Resolve successful-box narrow-grasp evidence before freezing: '+name)
    if report['models']['student_sha256']!=digest(student) or report['models']['residual_sha256']!=digest(residual):
        raise ValueError('Final test weights differ from export request')
    if not out.resolve().is_relative_to(OUTPUT):raise ValueError('Export outside W8 output root')
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');guard.check();out.mkdir(exist_ok=False)
    source=torch.load(student,map_location='cpu',weights_only=False)
    payload=dict(format='w8_frozen_student_v1',model=source['model'],model_config=source['model_config'],normalization=source['normalization'],**(dict(input_contract=source['input_contract']) if source.get('input_contract') else {}))
    relative_delta=source.get('relative_delta',False)
    contract=Student(command_history=bool(source.get('input_contract')),relative_delta=relative_delta)
    contract.validate_recovery_contract(source)
    if relative_delta:
        payload.update(format='w8_frozen_reldelta_student_v1',relative_delta=True,output_contract=source['output_contract'])

    weights={}
    for name,value in [('student',payload),('residual',torch.load(residual,map_location='cpu',weights_only=False))]:
        stream=io.BytesIO();torch.save(value,stream);data=stream.getvalue();guard.check(len(data))
        destination=out/f'{name}.pt'
        with destination.open('xb') as file:file.write(data)
        weights[name]=dict(file=destination.name,sha256=digest(destination),bytes=len(data))
    # Include the transitive in-repository preprocessing/architecture sources,
    # not just the small W10 wrapper. Installed package versions are recorded too.
    sources=sorted([WORKTREE/'legmm_skill/tabletop'/name for name in
        ('frozen_skill.py','student.py','residual_ppo.py','spec.py','data.py')]+
        [path for folder in ('airbot_depth','airbot_rl','depth_policy')
        for path in (WORKTREE/'legmm_skill'/folder).glob('*.py')])
    import importlib.metadata
    manifest=dict(format='w8_frozen_skill_v1',weights=weights,
        source_sha256={str(path.relative_to(WORKTREE/'legmm_skill')):digest(path) for path in sources},
        dependencies={name:importlib.metadata.version(name) for name in ('torch','rsl-rl-lib','tensordict')},
        visual=json.loads(visual_manifest.read_text()),
        physics=dict(global_position_iterations=[4,4],global_velocity_iterations=[0,0],actor_iterations=[4,0],free_base=True,
            physics_dt=.01,standing_dt=.02,native_control_dt=.04,command_delay_s=.1,self_collision=True),
        policy=dict(input_keys=['images','proprio'],images_shape=[2,2,128,128],depth_channels=['depth','validity'],
            views=['head','wrist'],proprio_names=[f'joint{i}' for i in range(1,7)]+['total_finger_opening'],
            proprio_units=['rad']*6+['m'],horizon=8,execute=4,decision_dt=.16,
            lower=LOWER.tolist(),upper=UPPER.tolist(),native_step=STEP_LENGTH.tolist(),residual_scale=RESIDUAL_SCALE.tolist(),
            internal_actor_state_dim=source['model_config']['state_dim'],
            command_history=source.get('input_contract'),
            residual_stochastic=False,initialization='fresh W8 student seed256; zero residual mean, init_std0.2 for training'),
        task=dict(timeout_s=15,minimum_lowest_point_clearance_m=.05,continuous_hold_s=2,
            forbidden_contact_threshold_N=1,box_grasp='narrow face, horizontal',wrist_compensation_degrees=15),
        independent_test=dict(path=str(test_report),sha256=digest(test_report),
            initial_states_sha256=report['precommitted_initial_states_sha256'],
            results={name:dict(success=entry['success_count'],total=256,wilson95=entry['wilson95']) for name,entry in report['drivers'].items()}),
        interface='tabletop.frozen_skill.FrozenW8Skill',acceptance_claim=False)
    if relative_delta:
        manifest['policy']['output_contract']=source['output_contract']
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser()
    for name in ('student','residual','test-report','visual-manifest','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();export(a.student,a.residual,a.test_report,a.visual_manifest,a.out)


if __name__=='__main__':main()
