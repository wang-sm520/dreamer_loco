"""Full-round validation resumes before more collection; default stays raw."""
import gzip,hashlib
import pytest
import torch
from tabletop.dagger_recovery import save_latest,restore_latest
from tabletop.state_fingerprints import expected_fingerprints,session_fingerprints,digest
from tabletop.storage import StorageGuard
from tabletop.tests.test_dagger_recovery import fixture_session

@pytest.mark.parametrize('compressed',[False,True])
def test_full_round_preserves_pending_validation_and_exact_next_batch(tmp_path,compressed):
    torch.set_num_threads(2)
    session=fixture_session(42)
    session.student.update(session.reservoir,batch=8,microbatch=8)
    session.student.updates=6000
    session.seen=session.budget=session.physical_env_decisions=96000
    session.round=2
    state=dict(formal_training=True,diagnostic_only=False,num_envs=64,
        current_round_update_count=2000,total_env_decision_budget=192000)
    record=save_latest(session,state,3,1408,18,tmp_path,StorageGuard(tmp_path,tmp_path/'code'),
        validation_pending=3,compress=compressed)
    path=tmp_path/'latest.pt'
    with (gzip.open(path,'rb') if compressed else path.open('rb')) as f:
        payload=torch.load(f,map_location='cpu',weights_only=False)
    expected=expected_fingerprints(payload)
    sample_rng=session.reservoir.rng.getstate()
    expected_batch=digest(session.reservoir.sample(64));session.reservoir.rng.setstate(sample_rng)
    recovered=fixture_session(91)
    status,cursor=restore_latest(recovered,path,expected_sha256=record['sha256'],expected_formal=True)
    assert session_fingerprints(recovered)==expected
    assert cursor['next_round']==3 and cursor['next_seed']==1408 and cursor['episode']==18
    assert cursor['validation_pending']==3 and cursor['partial_phase'] is None
    assert recovered.seen==96000 and recovered.student.updates==6000
    assert status['total_env_decision_budget']-recovered.seen==96000
    assert digest(recovered.reservoir.sample(64))==expected_batch
    with pytest.raises(ValueError,match='Resource recovery requires interrupted'):
        restore_latest(recovered,path,expected_sha256=record['sha256'],expected_formal=True,allow_resource_abort=True)

def test_default_raw_checkpoint_remains_torch_compatible(tmp_path):
    session=fixture_session(7)
    record=save_latest(session,dict(formal_training=False),1,320,1,tmp_path,StorageGuard(tmp_path,tmp_path/'code'))
    path=tmp_path/'latest.pt'
    assert record['encoding']=='torch' and path.read_bytes()[:2]!=b'\x1f\x8b'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['student']['updates']==session.student.updates
    assert payload['cursor']['next_round']==1
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
