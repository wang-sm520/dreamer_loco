"""Compressed full payload and raw compatibility, including Adam/reservoir/RNG."""
import gzip,hashlib,random
import numpy as np
import pytest
import torch
from tabletop.dagger_recovery import save_latest,restore_latest
from tabletop.state_fingerprints import digest
from tabletop.tests.test_closed_collection_recovery import fixture,continuation
from tabletop.storage import StorageGuard



def test_exact_full_payload_and_next_update_match_raw(tmp_path):
    torch.set_num_threads(2);session,raw,_=fixture(tmp_path)
    payload=torch.load(tmp_path/'latest.pt',weights_only=False)
    packed=tmp_path/'packed';packed.mkdir()
    record=save_latest(session,payload['status'],1,768,8,packed,StorageGuard(tmp_path,tmp_path/'code'),complete_round_boundary=False,compress=True)
    assert record['encoding']=='gzip' and record['bytes']<record['uncompressed_bytes']
    assert hashlib.sha256(gzip.decompress((packed/'latest.pt').read_bytes())).hexdigest()==record['uncompressed_sha256']
    with gzip.open(packed/'latest.pt','rb') as file:assert digest(torch.load(file,map_location='cpu',weights_only=False))==digest(payload)
    continuation(session);expected=[p.detach().clone() for p in session.student.model.parameters()];batch=session.reservoir.sample(3);rng=(random.random(),np.random.random(),torch.rand(3))
    other=tmp_path/'other';other.mkdir();recovered,_,_=fixture(other,91)
    state,cursor=restore_latest(recovered,packed/'latest.pt',allow_closed_collection=True,expected_sha256=record['sha256'],expected_formal=False,updates_per_round=2)
    assert recovered.seen==44032 and cursor['next_seed']==768 and recovered.student.updates==2
    continuation(recovered)
    assert all(torch.equal(a,b) for a,b in zip(expected,recovered.student.model.parameters()))
    actual_batch=recovered.reservoir.sample(3)
    assert all(np.array_equal(batch[k],actual_batch[k]) for k in batch)
    assert rng[0]==random.random() and rng[1]==np.random.random() and torch.equal(rng[2],torch.rand(3))
    with pytest.raises(ValueError,match='SHA differs'):restore_latest(recovered,packed/'latest.pt',allow_closed_collection=True,expected_sha256='bad')
    restore_latest(recovered,tmp_path/'latest.pt',allow_closed_collection=True,expected_sha256=raw['sha256'],expected_formal=False,updates_per_round=2)
    assert recovered.seen==44032


def test_compressed_identity_and_corrupt_SHA_are_rejected(tmp_path):
    session,_,_=fixture(tmp_path);payload=torch.load(tmp_path/'latest.pt',weights_only=False);packed=tmp_path/'packed';packed.mkdir()
    record=save_latest(session,payload['status'],1,768,8,packed,StorageGuard(tmp_path,tmp_path/'code'),complete_round_boundary=False,compress=True)
    with pytest.raises(ValueError,match='formal/diagnostic'):
        restore_latest(session,packed/'latest.pt',allow_closed_collection=True,expected_sha256=record['sha256'],expected_formal=True,updates_per_round=2)
    before=[p.detach().clone() for p in session.student.model.parameters()]
    path=packed/'latest.pt';data=bytearray(path.read_bytes());data[-8]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match='SHA differs'):
        restore_latest(session,path,allow_closed_collection=True,expected_sha256=record['sha256'],expected_formal=False,updates_per_round=2)
    assert all(torch.equal(a,b) for a,b in zip(before,session.student.model.parameters()))


def test_compressed_atomic_failure_keeps_old_source_and_pending_peak(tmp_path,monkeypatch):
    import hashlib
    import tabletop.storage as storage
    session,_,_=fixture(tmp_path);payload=torch.load(tmp_path/'latest.pt',weights_only=False);packed=tmp_path/'packed';packed.mkdir();guard=StorageGuard(tmp_path,tmp_path/'code')
    record=save_latest(session,payload['status'],1,768,8,packed,guard,complete_round_boundary=False,compress=True)
    old=(packed/'latest.pt').read_bytes();usage=storage.allocated_bytes(tmp_path);rounded=(record['bytes']+4095)//4096*4096
    with monkeypatch.context() as m:
        m.setattr(storage,'BUDGET_BYTES',usage+rounded-1)
        with pytest.raises(OSError,match='budget stop'):
            save_latest(session,payload['status'],1,768,8,packed,guard,complete_round_boundary=False,compress=True)
    assert (packed/'latest.pt').read_bytes()==old and not (packed/'latest.pt.pending').exists()
    def fail_replace(*args):raise OSError('injected atomic replace failure')
    with monkeypatch.context() as m:
        m.setattr('tabletop.dagger_recovery.os.replace',fail_replace)
        with pytest.raises(OSError,match='injected atomic'):
            save_latest(session,payload['status'],1,768,8,packed,guard,complete_round_boundary=False,compress=True)
    pending=packed/'latest.pt.pending';assert pending.exists() and (packed/'latest.pt').read_bytes()==old
    assert hashlib.sha256(old).hexdigest()==record['sha256']
    assert guard.check()['allocated_bytes']==usage+((pending.stat().st_size+4095)//4096)*4096
    with pytest.raises(FileExistsError):
        save_latest(session,payload['status'],1,768,8,packed,guard,complete_round_boundary=False,compress=True)
