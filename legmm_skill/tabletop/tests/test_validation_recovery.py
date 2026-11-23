import json,sys
from types import SimpleNamespace
import pytest
from tabletop.validation_recovery import classify_result
from tabletop.dagger_recovery import restore_latest


def outcome(seed,reason='timeout'):
    return dict(seed=seed,reason=reason,success=reason=='success')


def test_interrupted_evaluation_is_never_complete_failure():
    result=dict(driver='student',num_envs=2,safety_stop=None,requested_safe_stop=True,
        episodes=[outcome(10192,None),outcome(10193,'collision:link6:object')])
    assert classify_result(result,[10192,10193])=='interrupted'
    result['requested_safe_stop']=False
    with pytest.raises(ValueError,match='Unresolved'):classify_result(result,[10192,10193])
    result['episodes'][0]['reason']='timeout'
    assert classify_result(result,[10192,10193])=='complete'
    result['episodes'][0]['success']=True
    with pytest.raises(ValueError,match='Success and terminal'):classify_result(result,[10192,10193])


def test_safety_failure_and_wrong_seed_are_not_resource_recovery():
    result=dict(driver='student',num_envs=1,safety_stop='standing_height',episodes=[outcome(10000)])
    with pytest.raises(ValueError,match='Standing safety'):classify_result(result,[10000])
    result['safety_stop']=None
    with pytest.raises(ValueError,match='seed identity'):classify_result(result,[10001])


def test_loader_requires_explicit_sha_formal_identity_and_exclusive_stage(tmp_path):
    source=tmp_path/'absent.pt';manifest=tmp_path/'manifest.json'
    with pytest.raises(ValueError,match='Explicit SHA'):restore_latest(None,source,validation_manifest=manifest)
    with pytest.raises(ValueError,match='mutually exclusive'):restore_latest(None,source,validation_manifest=manifest,allow_partial=True,expected_sha256='x')
    with pytest.raises(ValueError,match='formal64'):restore_latest(None,source,validation_manifest=manifest,expected_sha256='x',expected_formal=False)


@pytest.mark.parametrize('reuse',[False,True])
def test_runner_reuses_completed_prefix_and_runs_only_missing_fixed_cohorts(tmp_path,monkeypatch,reuse):
    from tabletop import validation_recovery
    from tabletop.train_dagger import evaluate_fixed
    sources=[];reuse_rows={};original={}
    for seed in range(10000,10256,32):
        parent=tmp_path/f'teacher_{seed}';parent.mkdir();path=parent/'result.json'
        path.write_text(json.dumps(dict(seed=seed,states=[{'seed':x} for x in range(seed,seed+32)])))
        (parent/'sampling_source.json').write_text(json.dumps(dict(path=str(tmp_path/f'sampling_{seed}.json'))))
        sources.append(dict(result_path=str(path)))
        if seed<10192:
            prior=tmp_path/f'complete_{seed}.json';prior.write_text(json.dumps(dict(episodes=[outcome(x) for x in range(seed,seed+32)])))
            reuse_rows[seed]=dict(seed=seed,result_path=str(prior));original[prior]=prior.read_bytes()
    qualification=tmp_path/'qualification.json';qualification.write_text(json.dumps(dict(episodes=sources)))
    record=tmp_path/'record';record.mkdir();(record/'COORDINATOR_UPDATES.md').write_text('unit stage test')
    out=tmp_path/'out';out.mkdir();calls=[]
    def run(args,app,evaluation):
        calls.append(args.seed)
        assert args.residual_velocity is False
        return dict(safety_stop=None,episodes=[outcome(x,'success' if args.seed==10192 else 'timeout') for x in range(args.seed,args.seed+32)])
    monkeypatch.setitem(sys.modules,'tabletop.learning',SimpleNamespace(EvaluationSession=lambda *a:None))
    monkeypatch.setitem(sys.modules,'tabletop.learning_episode',SimpleNamespace(run_episode=run))
    monkeypatch.setitem(sys.modules,'isaacsim.core.utils.stage',SimpleNamespace(create_new_stage=lambda:None))
    monkeypatch.setattr(validation_recovery,'reusable_results',lambda *a:reuse_rows)
    summary=evaluate_fixed(SimpleNamespace(student=None,visual=None),None,out,qualification,record,reuse_manifest=tmp_path/'manifest.json' if reuse else None)
    assert calls==list(range(10192 if reuse else 10000,10256,32))
    assert len(summary['episodes'])==256 and summary['success_count']==32 and summary['student_gate_passed'] is False
    assert sorted(x['seed'] for x in summary['episodes'])==list(range(10000,10256))
    assert all(path.read_bytes()==raw for path,raw in original.items())
    assert (out/'reused_batches.json').exists()==reuse


def test_formal_launch_requires_successful_cuda_resource_supervisor(tmp_path):
    from tabletop.validation_recovery import require_cuda_proof,sha
    source=tmp_path/'source.pt';source.write_bytes(b'unit source identity');digest=sha(source)
    proof_root=tmp_path/'proof';proof_root.mkdir()
    cpu=dict(completed=True,source=str(source),source_sha256=digest)
    gpu=dict(completed=True,source_path=str(source),source_sha256=digest,proof_output_root=str(proof_root),state_checks={'model':True},next_random_checks={'cuda':True},next_sample64_exact=True,formal_env_decisions_added=0,optimizer_updates_added=0,physics_steps=0)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(source=str(source),source_sha256=digest)))
    (tmp_path/'source_cpu_audit.json').write_text(json.dumps(cpu));(tmp_path/'validation_recovery_GPU_proof.json').write_text(json.dumps(gpu))
    parent=proof_root/'status.json';parent.write_text(json.dumps(dict(completed=False,exit_code=0,reason='resource stop')))
    with pytest.raises(ValueError,match='supervisor'):require_cuda_proof(tmp_path,source,digest,manifest)
    parent.write_text(json.dumps(dict(completed=True,exit_code=0,reason=None)))
    require_cuda_proof(tmp_path,source,digest,manifest)
    source.write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='source changed'):require_cuda_proof(tmp_path,source,digest,manifest)
