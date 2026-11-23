import hashlib
import torch
from tabletop.round_end_observer_v2 import read_owned_source,identity


def test_pending_rename_during_load_keeps_same_opened_atomic_source(tmp_path,monkeypatch):
    pending=tmp_path/'latest.pt.pending';target=tmp_path/'latest.pt';payload={'complete_round_boundary':True,'student':{'updates':4000},'value':torch.arange(16)};torch.save(payload,pending)
    original=torch.load
    def load_and_rename(file,**kwargs):
        result=original(file,**kwargs);pending.rename(target);return result
    monkeypatch.setattr(torch,'load',load_and_rename)
    restored,digest,key=read_owned_source(pending)
    assert not pending.exists() and torch.equal(restored['value'],payload['value'])
    assert key==identity(target) and digest==hashlib.sha256(target.read_bytes()).hexdigest()
