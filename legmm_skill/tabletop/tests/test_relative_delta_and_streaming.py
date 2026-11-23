import gzip
import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from airbot_depth.model import AirbotPolicyModel
from tabletop.dagger_recovery import save_latest, restore_latest
from tabletop.data import Reservoir
from tabletop.learning import arm_step4_statistics
from tabletop.spec import LOWER, STEP_LENGTH, UPPER
from tabletop.state_fingerprints import digest, session_fingerprints
from tabletop.storage import StorageGuard
from tabletop.student import Student
from tabletop.train_dagger import planned_rounds, write_round1_fast_gate


def make_session(history, delta, seed=23):
    torch.set_num_threads(2)
    reservoir = Reservoir(4, seed, state_dim=14 if history else 7)
    rng = np.random.default_rng(12)
    for _ in range(4):
        q = rng.uniform(LOWER, UPPER).astype(np.float32)
        state = np.r_[q, q + .024] if history else q
        reservoir.add(rng.random((2, 2, 128, 128)), state,
                      q + rng.normal(0, .03, (8, 7)),
                      dict(shape='box', outcome='success', round=0, seed=seed))
    return SimpleNamespace(student=Student('cpu', seed, command_history=history, relative_delta=delta),
        reservoir=reservoir, visual=SimpleNamespace(manifest=lambda: {'fixture': True}),
        seen=96000, budget=96000, round=2, teacher_drives=False, episode_serial=7,
        physical_env_decisions=96064,collected_samples=90000)


@pytest.mark.parametrize('history',[False,True])
def test_absolute_path_bitwise_legacy(history):
    session=make_session(history,False)
    student=session.student
    with torch.random.fork_rng():
        torch.manual_seed(23)
        old=AirbotPolicyModel(vocabulary_size=2,state_dim=14 if history else 7)
    assert digest(student.model.state_dict())==digest(old.state_dict())
    optimizer=torch.optim.AdamW(old.parameters(),lr=1e-4)
    mean=torch.tensor((LOWER+UPPER)/2,dtype=torch.float32)
    scale=torch.tensor((UPPER-LOWER)/2,dtype=torch.float32)

    def old_prediction(images,q):
        q=torch.as_tensor(q)
        in_mean=torch.cat((mean,mean)) if history else mean
        in_scale=torch.cat((scale,scale)) if history else scale
        return old((q-in_mean)/in_scale,torch.ones((len(q),1),dtype=torch.long),torch.as_tensor(images,dtype=torch.float32))*scale+mean

    reservoir=session.reservoir
    ref=make_session(history,False).reservoir
    example=ref.sample(2)
    ref.rng.setstate(reservoir.rng.getstate())
    old.eval();student.model.eval()
    assert torch.equal(old_prediction(example['images'],example['proprio']),student.prediction(example['images'],example['proprio']))
    init_rng=torch.get_rng_state()
    samples=ref.sample(4)
    old.train();optimizer.zero_grad(set_to_none=True)
    expected=0.
    for i in (0,2):
        pred=old_prediction(samples['images'][i:i+2],samples['proprio'][i:i+2])
        loss=(((pred-torch.as_tensor(samples['label'][i:i+2]))/scale)**2).mean()*(2/4)
        expected+=float(loss.detach())
        loss.backward()
    torch.nn.utils.clip_grad_norm_(old.parameters(),1.)
    optimizer.step()
    final_rng=torch.get_rng_state()
    torch.set_rng_state(init_rng)
    assert student.update(reservoir,batch=4,microbatch=2)==expected
    assert digest(student.model.state_dict())==digest(old.state_dict())
    assert digest(student.optimizer.state_dict())==digest(optimizer.state_dict())
    assert torch.equal(torch.get_rng_state(),final_rng)
    assert reservoir.rng.getstate()==ref.rng.getstate()


@pytest.mark.parametrize('history',[False,True])
def test_measured_anchor_output_scale_and_loss(history):
    session=make_session(history,True)
    student=session.student
    batch=session.reservoir.sample(2)
    q=torch.from_numpy(batch['proprio'])
    anchor=q[:,None,:7].expand(-1,8,-1)
    with torch.no_grad():
        for parameter in student.model.action_head.parameters():parameter.zero_()
    assert torch.equal(student.prediction(batch['images'],q),anchor)
    increments=torch.arange(56,dtype=torch.float32).reshape(8,7)/10
    with torch.no_grad():student.model.action_head.bias.copy_(increments.flatten())
    scale=torch.tensor(4*STEP_LENGTH,dtype=torch.float32)
    expected=anchor+increments*scale
    assert torch.equal(student.prediction(batch['images'],q),expected)
    batch['label']=(anchor-2*scale).numpy()
    target=torch.as_tensor(batch['label'])
    reference=(((expected-target)/scale)**2).mean().item()
    assert student.update(SimpleNamespace(sample=lambda _:batch),batch=2,microbatch=2)==pytest.approx(reference)


@pytest.mark.parametrize('history',[False,True])
@pytest.mark.parametrize('delta',[False,True])
def test_model_checkpoint_contracts(history,delta,tmp_path):
    session=make_session(history,delta)
    student=session.student
    student.update(session.reservoir,batch=2,microbatch=2)
    path=tmp_path/'student.pt'
    meta={'test': 'relative delta'}
    student.save(path,meta)
    saved=torch.load(path,weights_only=False)
    assert ('output_contract' in saved)==delta
    assert ('relative_delta' in saved)==delta
    assert ('delta_scale' in saved['normalization'])==delta
    same=Student('cpu',3,command_history=history,relative_delta=delta)
    assert same.restore(path)==meta
    automatic=Student.from_checkpoint(path)
    assert automatic.command_history==history and automatic.relative_delta==delta
    for restored in (same,automatic):
        assert restored.updates==1
        assert digest(restored.model.state_dict())==digest(student.model.state_dict())
        assert digest(restored.optimizer.state_dict())==digest(student.optimizer.state_dict())
    with pytest.raises(ValueError):Student('cpu',3,command_history=history,relative_delta=not delta).restore(path)
    if delta:
        for key,value in [('relative_delta',False),('output_contract',None),
                          ('normalization',saved['normalization'] | {'delta_scale':torch.ones(7)})]:
            bad=tmp_path/f'{key}.pt'
            torch.save(saved | {key:value},bad)
            with pytest.raises(ValueError):Student.from_checkpoint(bad)


@pytest.mark.parametrize('delta',[False,True])
@pytest.mark.parametrize('compress',[False,True])
def test_streamed_session_contract_and_completed_milestone_resume(delta,compress,tmp_path):
    source=make_session(True,delta)
    source.student.update(source.reservoir,batch=2,microbatch=2)
    expected=session_fingerprints(source)
    guard=StorageGuard(tmp_path,tmp_path/'code')
    status=dict(completed=True,phase='round_03_milestone_complete',formal_training=True)
    result=save_latest(source,status,3,704,7,tmp_path,guard,compress=compress)
    path=tmp_path/'latest.pt'
    encoded=path.read_bytes()
    raw=gzip.decompress(encoded) if compress else encoded
    assert result['sha256']==hashlib.sha256(encoded).hexdigest()
    assert result['uncompressed_sha256']==hashlib.sha256(raw).hexdigest()
    assert result['bytes']==len(encoded) and result['uncompressed_bytes']==len(raw)
    target=make_session(True,delta,seed=4)
    restored,cursor=restore_latest(target,path,expected_formal=True,expected_sha256=result['sha256'])
    assert restored==status
    assert cursor['next_seed']==704 and cursor['episode']==7 and cursor['validation_pending'] is None
    assert list(planned_rounds(cursor['next_round'],6))==[3,4,5]
    assert list(planned_rounds(0,6,3))==[0,1,2]
    assert session_fingerprints(target)==expected
    assert target.student.updates==1 and target.student.output_contract==source.student.output_contract
    assert torch.equal(target.student.delta_scale,source.student.delta_scale)
    with pytest.raises(ValueError,match='Relative-delta'):restore_latest(make_session(True,not delta),path)
    payload=torch.load(io.BytesIO(raw),weights_only=False)
    if delta:
        payload['student']['delta_scale'].zero_()
        corrupt=tmp_path/'corrupt.pt'
        torch.save(payload,corrupt)
        with pytest.raises(ValueError,match='normalization differs'):restore_latest(target,corrupt)
    else:
        # ZIP and whole-buffer gzip archives written by pre-change versions remain readable.
        old=io.BytesIO()
        torch.save(payload,old,pickle_protocol=5)
        old_file=tmp_path/'old.pt'
        old_file.write_bytes(gzip.compress(old.getvalue()) if compress else old.getvalue())
        restore_latest(target,old_file)
        assert session_fingerprints(target)==expected


def test_streaming_guard_failure_keeps_prior_atomic_file(tmp_path):
    session=make_session(True,True)
    guard=StorageGuard(tmp_path,tmp_path/'code')
    saved=save_latest(session,{},1,256,1,tmp_path,guard,compress=True)
    class StopGuard:
        def check(self,write_bytes=0):
            if write_bytes:raise OSError('disk guard fixture')
    with pytest.raises(OSError,match='disk guard fixture'):
        save_latest(session,{},2,320,2,tmp_path,StopGuard(),compress=True)
    assert hashlib.sha256((tmp_path/'latest.pt').read_bytes()).hexdigest()==saved['sha256']
    assert not (tmp_path/'latest.pt.pending').exists()


def test_relative_delta_frozen_features():
    pytest.importorskip('rsl_rl')
    from tabletop.residual_ppo import FrozenStudentFeatures
    session=make_session(True,True)
    student=session.student
    features=FrozenStudentFeatures(student)
    batch=session.reservoir.sample(2)
    q,last=np.split(batch['proprio'],2,axis=1)
    policy,base=features.observe(batch['images'],q,last)
    assert policy['policy'].shape==(2,547)
    assert torch.equal(base,student.act(batch['images'],q,last)[:,:4])
    assert torch.equal(policy['policy'][:,512:519],(torch.from_numpy(q)-student.mean)/student.scale)
    features.close()


def test_step4_and_terminal_gate(tmp_path):
    previous=np.zeros((4,7),dtype=np.float32)
    student=np.zeros((4,8,7),dtype=np.float32)
    teacher=np.zeros_like(student)
    teacher[:,3,0]=[.01,.01,.00001,.01]
    student[:,3,0]=[.02,-.01,.01,.01]
    info=arm_step4_statistics(student,teacher,previous,[0,1,2])
    assert info==pytest.approx(dict(arm_step4_eligible=2,
        arm_step4_amp_ratio_median=1.5,arm_step4_cos_median=0.))
    cohort=tmp_path/'round_01_episode_0007'
    cohort.mkdir()
    path=cohort/'result.json'
    path.write_text(json.dumps(dict(episodes=[dict(success=False,reason='timeout'),
                                            dict(success=False,reason=None)])))
    (cohort/'learning_decisions.jsonl').write_text(json.dumps(info)+'\n')
    stop=write_round1_fast_gate(tmp_path,tmp_path)
    assert not stop['passed'] and stop['completed_episodes']==1 and stop['unresolved_episodes']==1
    path.write_text(json.dumps(dict(episodes=[dict(success=True,reason='success'),
                                            dict(success=False,reason=None)])))
    go=write_round1_fast_gate(tmp_path,tmp_path)
    assert go['passed'] and go['success_count']==1 and go['arm_step4_amp_ratio_median']==1.5


def test_exact_source_admission_and_proof_binding(tmp_path,monkeypatch):
    from tabletop import reldelta_admission as admission

    def write(path,value):
        path.write_text(json.dumps(value))
        return path

    files=tmp_path/'runtime'
    files.mkdir()
    for name in set(admission.CAPACITY_SOURCES+admission.RUNTIME_SOURCES):
        (files/name).write_text(f'exact source {name}')
    monkeypatch.setattr(admission,'__file__',str(files/'reldelta_admission.py'))
    original={name:admission.sha256(files/name) for name in admission.CAPACITY_SOURCES}
    for name in admission.LEARNED_SOURCES:
        original[name]='parent learning source'
    parent=tmp_path/'parent'
    parent.mkdir()
    parent_status=write(parent/'status.json',{})
    def inventory(dest,hashes):
        write(dest/'source_inventory.json',[dict(path=name,sha256=value) for name,value in hashes.items()])
    inventory(parent,original)
    short=tmp_path/'short'
    short.mkdir()
    training=write(short/'training_status.json',dict(command_history=True,relative_delta=True,
        actor_state_dim=14,optimizer_updates=3,full_session_restore_passed=True,
        latest_save=dict(encoding='gzip',before=dict(rss_bytes=500),after_serialization=dict(rss_bytes=510),sha256='proof')))
    short_status=write(short/'status.json',dict(completed=True,CUDA_VISIBLE_DEVICES='0',num_envs=64,
        training_status=str(training),peak_gpu_mib=12800))
    runtime=admission.runtime_hashes()
    inventory(short,runtime)
    update=tmp_path/'COORDINATOR_UPDATES.md'
    update.write_text('fixture authorization')
    write(tmp_path/'coordinator_update_ack.json',dict(version=10,coordinator_updates_sha256=admission.sha256(update)))
    write(tmp_path/'cpu_paired_400updates.json',dict(completed=True,gate_passed=True,completed_updates_each=400))
    proof=dict(completed=True,CUDA_VISIBLE_DEVICES='0',peak_gpu_mib=512,source_sha256='proof',
        next_batch_exact=True,next_random_exact=True,checks=dict.fromkeys(admission.RESTORE_CHECKS,True),runtime_sha256=runtime)
    proof_file=write(tmp_path/'real_GPU_reldelta_restore.json',proof)
    write(tmp_path/'reldelta_admission.json',dict(parent_capacity_status=str(parent_status),short_capacity_status=str(short_status),
        runtime_sha256=runtime,files={name:dict(parent_capacity_sha256=original[name],after_sha256=runtime[name])
                                    for name in admission.LEARNED_SOURCES}))
    admission.require_reldelta_admission(tmp_path,parent_status)
    observer=files/'visual.py'
    previous=observer.read_text()
    observer.write_text('changed')
    with pytest.raises(ValueError,match='visual.py'):
        admission.require_reldelta_admission(tmp_path,parent_status)
    observer.write_text(previous)
    write(proof_file, proof | {'source_sha256':'other'})
    with pytest.raises(ValueError,match='restoration proof failed'):
        admission.require_reldelta_admission(tmp_path,parent_status)
    write(proof_file, proof)
    (files/'student.py').write_text('changed after probe')
    with pytest.raises(ValueError,match='runtime sources changed'):
        admission.require_reldelta_admission(tmp_path,parent_status)


def test_only_exact_audited_observer_and_foot_changes_are_admitted(tmp_path,monkeypatch):
    import shutil
    from pathlib import Path
    from tabletop import foot_admission, reldelta_admission as admission, segment_timing

    source=tmp_path/'source'
    source.mkdir()
    for name in admission.CAPACITY_SOURCES:
        (source/name).write_text(name)
    monkeypatch.setattr(admission,'__file__',str(source/'reldelta_admission.py'))
    hashes={name:admission.sha256(source/name) for name in admission.CAPACITY_SOURCES}
    hashes.update({'visual.py':'visual-parent','learning_episode.py':'episode-parent'})
    (tmp_path/'source_inventory.json').write_text(json.dumps([
        dict(path=name,sha256=digest) for name,digest in hashes.items()]))
    status=tmp_path/'status.json'
    updates=tmp_path/'COORDINATOR_UPDATES.md'
    updates.write_text('v10 evidence fixture')
    ack=tmp_path/'coordinator_update_ack.json'
    ack.write_text(json.dumps(dict(version=10,coordinator_updates_sha256=admission.sha256(updates))))

    timing=Path(segment_timing.__file__)
    shutil.copyfile(timing,source/timing.name)
    monkeypatch.setattr(segment_timing,'__file__',str(source/timing.name))
    diagnostics=source/'diagnostics'
    diagnostics.mkdir()
    visual=source/'visual.py'
    (diagnostics/'v7_segment_timing_proof.json').write_text(json.dumps(dict(
        helper_sha256=admission.sha256(source/timing.name),
        files={'visual.py':dict(before_sha256=hashes['visual.py'],prepared_sha256=admission.sha256(visual))},
        qualified_physics_step_AST_unchanged_after_timing_removal=True,
        functional_observation_and_policy_AST_unchanged=True)))
    foot_helper=Path(foot_admission.__file__).with_name('foot_readback_candidate.py')
    shutil.copyfile(foot_helper,source/foot_helper.name)
    monkeypatch.setattr(foot_admission,'__file__',str(source/'foot_admission.py'))
    foot_proof=tmp_path/'v9_foot_adoption.json'
    foot_proof.write_text(json.dumps(dict(adopted=True,actual_GPU_all_pairs_exact=True,
        same_state_timing_meaningful_improvement=True,resource_guards_satisfied=True,
        capacity_baseline_sha256=hashes['learning_episode.py'],
        adopted_runtime_sha256=admission.sha256(source/'learning_episode.py'),
        helper_sha256=admission.sha256(source/foot_helper.name))))
    assert admission.require_parent_sources(status,tmp_path)==hashes

    for name in ('visual.py','learning_episode.py','physics_config.py'):
        path=source/name
        before=path.read_bytes()
        path.write_bytes(before+b' edited')
        with pytest.raises(ValueError,match=name):
            admission.require_parent_sources(status,tmp_path)
        path.write_bytes(before)
    updates.write_text('different authorization')
    with pytest.raises(ValueError,match='visual.py'):
        admission.require_parent_sources(status,tmp_path)


def test_timing_writer_uses_only_the_run_scoped_storage_guard(tmp_path,monkeypatch):
    from tabletop import spec, storage
    from tabletop.segment_timing import SegmentTimings

    legacy=tmp_path/'legacy'
    legacy.mkdir()
    code=tmp_path/'legmm_skill/tabletop'
    monkeypatch.setattr(spec,'OUTPUT',legacy)
    monkeypatch.setattr(spec,'WORKTREE',tmp_path)
    monkeypatch.setattr(storage,'BUDGET_BYTES',0)
    timings=SegmentTimings(enabled=True)
    output=tmp_path/'timings.json'
    with pytest.raises(OSError,match='W8 budget stop'):
        timings.write(output)
    requested=[]
    guard=SimpleNamespace(check=requested.append)
    with storage.scoped_storage_guard(legacy,code,guard):
        assert timings.write(output)
        assert requested==[output.stat().st_size]
        with pytest.raises(OSError,match='W8 budget stop'):
            storage.StorageGuard(legacy,tmp_path/'different_code').check(1)
    with pytest.raises(OSError,match='W8 budget stop'):
        timings.write(tmp_path/'after.json')
