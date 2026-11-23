"""Regression for a stopped attempt using a separate rolling checkpoint root."""
import copy, hashlib, json
from pathlib import Path
import pytest
import torch
from tabletop.tests.test_resource_abort_recovery import fixture, RESTORE
from tabletop.resource_abort_recovery import resource_abort_plan
from tabletop.dagger_recovery import save_latest
from tabletop.storage import StorageGuard
from tabletop.state_fingerprints import expected_fingerprints, session_fingerprints, expected_next_batch, digest


def split_fixture(tmp_path):
    session,_,episode=fixture(tmp_path)
    payload=torch.load(tmp_path/'latest.pt',weights_only=False)
    rolling=tmp_path/'rolling';rolling.mkdir()
    previous=payload['status']['latest_save'];previous['path']=str(rolling/'latest.pt')
    (rolling/'latest_save_memory.jsonl').write_text(json.dumps(previous)+'\n')
    saved=save_latest(session,payload['status'],2,1088,13,rolling,StorageGuard(tmp_path,tmp_path/'code'),complete_round_boundary=False,compress=True)
    (tmp_path/'training_status.json').write_text(json.dumps(dict(PID=payload['status']['PID'],latest_save=saved)))
    (tmp_path/'latest_save_memory.jsonl').unlink()
    import gzip
    with gzip.open(rolling/'latest.pt','rb') as f:payload=torch.load(f,weights_only=False)
    return session,payload,saved,rolling


def test_split_rolling_root_restores_exact_full_state_and_prefix(tmp_path):
    original,payload,saved,rolling=split_fixture(tmp_path)
    plan=resource_abort_plan(payload,2)
    assert plan['consumed_prefix_decisions']==3776 and plan['next_seed']==1088
    assert str(rolling/'latest_save_memory.jsonl') in plan['evidence_sha256']
    assert str(tmp_path/'training_status.json') in plan['evidence_sha256']
    expected=expected_fingerprints(payload);next_batch=expected_next_batch(payload)
    other=tmp_path/'other';other.mkdir();restored,_,_=fixture(other,47)
    _,cursor=RESTORE(restored,rolling/'latest.pt',allow_resource_abort=True,expected_sha256=saved['sha256'],expected_formal=False,updates_per_round=2)
    assert session_fingerprints(restored)==expected
    assert digest(restored.reservoir.sample(64))==next_batch
    assert restored.seen==67776 and restored.student.updates==4
    assert cursor['episode']==13 and not cursor['resource_abort_plan']['aborted_initial_states_replayed']


@pytest.mark.parametrize('mutation',['missing_receipt','wrong_pid','wrong_sha','wrong_saved_path','unowned_path','missing_full_record','symlink_log'])
def test_split_root_requires_owned_bound_save_receipts(tmp_path,mutation):
    _,payload,_,rolling=split_fixture(tmp_path)
    receipt=tmp_path/'training_status.json';log=rolling/'latest_save_memory.jsonl'
    if mutation=='missing_receipt':receipt.unlink()
    elif mutation in ('wrong_pid','wrong_sha'):
        x=json.loads(receipt.read_text())
        if mutation=='wrong_pid':x['PID']+=1
        else:x['latest_save']['sha256']='0'*64
        receipt.write_text(json.dumps(x))
    elif mutation=='wrong_saved_path':
        rows=[json.loads(x) for x in log.read_text().splitlines()];rows[-1]['path']=str(tmp_path/'latest.pt');log.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    elif mutation=='unowned_path':payload['status']['latest_save']['path']='/tmp/unowned-w8/latest.pt'
    elif mutation=='missing_full_record':log.write_text(log.read_text().splitlines()[-1]+'\n')
    elif mutation=='symlink_log':
        real=rolling/'real.jsonl';log.rename(real);log.symlink_to(real)
    with pytest.raises(ValueError):resource_abort_plan(payload,2)
