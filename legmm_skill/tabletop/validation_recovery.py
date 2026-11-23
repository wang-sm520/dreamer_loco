"""Explicit learner-only recovery of a resource-aborted fixed validation.

Completed cohorts remain immutable evidence. An interrupted evaluation has no
restorable physics state: its prefix is recorded, and its fixed seed is evaluated
again from the original initial state. This never adds training interactions.
"""
import hashlib,json
from pathlib import Path
from .spec import OUTPUT,WORKTREE


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def owned(path):
    p=Path(path)
    if p.is_symlink() or not p.resolve().is_relative_to(OUTPUT):
        raise ValueError('W8-owned immutable evidence required')
    return p


def classify_result(result,expected_seeds):
    episodes=result.get('episodes',[])
    if result.get('driver')!='student' or len(episodes)!=len(expected_seeds) or result.get('num_envs')!=len(expected_seeds):
        raise ValueError('Validation driver/count differs')
    if sorted(x['seed'] for x in episodes)!=expected_seeds:
        raise ValueError('Validation seed identity differs')
    if result.get('safety_stop') is not None:
        raise ValueError('Standing safety stop cannot use resource recovery')
    resolved=[x.get('reason') is not None for x in episodes]
    if result.get('requested_safe_stop'):
        if all(resolved):raise ValueError('Resource interruption has no unresolved episodes')
        return 'interrupted'
    if not all(resolved):raise ValueError('Unresolved result is not a completed evaluation')
    if any(bool(x['success'])!=(x['reason']=='success') for x in episodes):
        raise ValueError('Success and terminal reason disagree')
    return 'complete'


def validation_ledger(root,qualification):
    """Bind fixed qualification, physics, accepted states and every finished row."""
    root=owned(root);qualification=owned(qualification)
    q=json.loads(qualification.read_text())
    if not q.get('teacher_gate_passed') or q.get('num_episodes')!=256 or q.get('teacher_kind')!='measured_state_feedback':
        raise ValueError('Qualified fixed teacher required')
    sources=sorted({row['result_path'] for row in q['episodes']})
    complete=[];interrupted=[];missing=[];seeds=[];evidence={str(qualification):sha(qualification)}
    from .physics_config import verify_task_physics_audit
    for source in sources:
        teacher_path=owned(source);teacher=json.loads(teacher_path.read_text())
        seed=teacher['seed'];expected=list(range(seed,seed+len(teacher['states'])))
        seeds.extend(expected);batch=root/f'seed_{seed}';path=batch/'result.json'
        if not path.exists():missing.extend(expected);continue
        owned(path);result=json.loads(path.read_text());kind=classify_result(result,expected)
        if result['states']!=teacher['states']:raise ValueError('Fixed initial states differ')
        sample=json.loads((batch/'sampling_source.json').read_text())['path']
        if sample!=json.loads((teacher_path.parent/'sampling_source.json').read_text())['path']:
            raise ValueError('Fixed sampling source differs')
        verify_task_physics_audit(json.loads((batch/'physics_audit.json').read_text()))
        evaluation=json.loads((batch/'student_evaluation.json').read_text())
        if evaluation['total']!=len(expected) or evaluation['success']!=sum(bool(x['success']) for x in result['episodes']):
            raise ValueError('Evaluation counter witness differs')
        row=dict(seed=seed,seeds=expected,result_path=str(path),sha256=sha(path),
            env_decisions=evaluation['env_decisions'],physical_steps=result['physical_steps'],
            success=sum(bool(x['success']) for x in result['episodes']),
            unresolved=[x['seed'] for x in result['episodes'] if x.get('reason') is None])
        (complete if kind=='complete' else interrupted).append(row)
        for name in ['result.json','student_evaluation.json','physics_audit.json','sampling_source.json','sampling.json']:
            evidence[str(batch/name)]=sha(batch/name)
    if sorted(seeds)!=list(range(10000,10256)) or len(set(seeds))!=256:
        raise ValueError('Fixed validation coverage differs')
    if len(interrupted)!=1 or not complete or not missing:
        raise ValueError('This protocol requires completed prefix, one interrupted cohort, and untouched suffix')
    ordered=sorted(complete,key=lambda x:x['seed']);abort=interrupted[0]
    prefix=[s for x in ordered for s in x['seeds']]
    if prefix!=list(range(10000,abort['seed'])) or sorted(missing)!=list(range(max(abort['seeds'])+1,10256)):
        raise ValueError('Validation prefix ledger has a gap')
    return dict(complete=ordered,interrupted=interrupted,missing_seeds=sorted(missing),
        completed_episodes=len(prefix),target_episodes=256,evidence_sha256=evidence)


def recovery_plan(payload,manifest_path,source_sha):
    """Reject unsupported phases before any learner state is mutated."""
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest.get('format')!='w8_validation_resource_resume_v1' or manifest['source_sha256']!=source_sha:
        raise ValueError('Validation recovery manifest/source SHA differs')
    if payload.get('format')!='w8_dagger_partial_v1' or payload.get('complete_round_boundary'):
        raise ValueError('Explicit partial validation state required')
    c=payload['collection'];u=payload['cursor'];s=payload['status'];b=payload['reservoir']
    number=c['round']+1
    if u.get('partial_phase')!='student_fixed_validation' or s.get('last_phase')!='student_fixed_validation':
        raise ValueError('Only interrupted fixed validation is supported')
    if number!=3 or u['next_round']!=2 or s.get('validation_pending')!=3:
        raise ValueError('Unsupported fixed validation boundary')
    if not s.get('formal_training') or s.get('diagnostic_only') or s['num_envs']!=64 or s['total_env_decision_budget']!=192000:
        raise ValueError('Formal64 original budget required')
    if c['seen']!=96000 or c['physical_env_decisions']!=96000 or c['budget']!=96000 or payload['student']['updates']!=6000:
        raise ValueError('Validation consumed training budget or changed updates')
    if u['updates_completed_in_round']!=2000 or b['seen']!=c['collected_samples'] or sum(b['composition'].values())!=b['seen'] or len(b['items'])!=b['capacity']:
        raise ValueError('Incomplete learner counters/buffer')
    if not s.get('reason','').startswith("SafeStop('Stop signal 15 handled at safe boundary: validation action-block boundary"):
        raise ValueError('Not a graceful validation action-block stop')
    attempt=owned(manifest['attempt_root']);parent=json.loads((attempt/'status.json').read_text())
    state=json.loads((attempt/'training_status.json').read_text())
    if parent.get('exit_code')!=0 or parent.get('completed') or not parent.get('reason','').startswith('RuntimeError("Shared GPU external occupancy exceeds4GiB:'):
        raise ValueError('Witnessed external GPU protection stop required')
    if state.get('latest_save',{}).get('sha256')!=source_sha or parent['PID']!=s['PID'] or state['PID']!=s['PID']:
        raise ValueError('Stopped source identity differs')
    if Path('/proc',str(s['PID'])).exists() or Path('/proc',str(parent['supervisor_pid'])).exists():
        raise ValueError('Prior task still alive')
    for path,expected in manifest['evidence_sha256'].items():
        if sha(owned(path))!=expected:raise ValueError('Recovery evidence SHA differs: '+path)
    for name,expected in manifest['unchanged_code_sha256'].items():
        if Path(name).name!=name or sha(WORKTREE/'legmm_skill/tabletop'/name)!=expected:
            raise ValueError('Learning/physics source differs: '+name)
    from .state_fingerprints import expected_fingerprints
    if expected_fingerprints(payload)!=manifest['source_fingerprints']:
        raise ValueError('Learner/RNG fingerprint differs')
    previous=s['source_recovery']['cursor']
    if any(u[k]!=previous[k] for k in ['next_seed','episode']):raise ValueError('Next collection cursor differs')
    ledger=validation_ledger(manifest['validation_root'],manifest['qualification'])
    if ledger!=manifest['ledger']:raise ValueError('Validation ledger changed')
    return dict(phase='resource_aborted_fixed_validation_learner_only',next_round=number,
        next_seed=u['next_seed'],episode=u['episode'],validation_pending=number,
        manifest=str(manifest_path),completed_episodes=ledger['completed_episodes'],
        interrupted_evaluation_decisions=sum(x['env_decisions'] for x in ledger['interrupted']),
        training_interactions_added=0,optimizer_updates_added=0,physical_episode_resumed=False,
        completed_results_reused=True,interrupted_fixed_seed_restarted=True)


def reusable_results(manifest_path,qualification):
    m=json.loads(Path(manifest_path).read_text())
    if str(qualification)!=m['qualification'] or sha(qualification)!=m['evidence_sha256'][str(qualification)]:
        raise ValueError('Qualification differs from recovery manifest')
    for path,expected in m['evidence_sha256'].items():
        if sha(owned(path))!=expected:raise ValueError('Recovery evidence SHA differs: '+path)
    ledger=validation_ledger(m['validation_root'],qualification)
    if ledger!=m['ledger']:raise ValueError('Validation ledger changed')
    return {row['seed']:row for row in ledger['complete']}


def require_cuda_proof(record_root,source,source_sha,manifest_path):
    """Formal admission requires successful CPU and resource-supervised CUDA proof."""
    root=Path(record_root)
    cpu=json.loads((root/'source_cpu_audit.json').read_text())
    gpu=json.loads((root/'validation_recovery_GPU_proof.json').read_text())
    manifest=json.loads(Path(manifest_path).read_text())
    if not cpu.get('completed') or not gpu.get('completed'):
        raise ValueError('Validation recovery proof is incomplete')
    if any(x!=source_sha for x in [cpu['source_sha256'],gpu['source_sha256'],manifest['source_sha256']]):
        raise ValueError('Validation recovery proof source differs')
    if str(source)!=cpu['source'] or str(source)!=gpu['source_path'] or str(source)!=manifest['source']:
        raise ValueError('Validation recovery proof path differs')
    parent=json.loads((owned(gpu['proof_output_root'])/'status.json').read_text())
    if not parent.get('completed') or parent.get('exit_code')!=0 or parent.get('reason'):
        raise ValueError('CUDA proof resource supervisor did not succeed')
    if not all(gpu['state_checks'].values()) or not all(gpu['next_random_checks'].values()) or not gpu['next_sample64_exact']:
        raise ValueError('CUDA recovery equivalence failed')
    if gpu['formal_env_decisions_added'] or gpu['optimizer_updates_added'] or gpu['physics_steps']:
        raise ValueError('Recovery proof consumed training or simulation budget')
    if sha(source)!=source_sha:raise ValueError('Recovery source changed since proof')
