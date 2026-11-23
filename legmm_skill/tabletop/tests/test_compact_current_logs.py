import gzip,hashlib,json
import pytest
from tabletop.compact_current_logs import finalize_closed_label_log


def test_closed_current_log_is_lossless_and_other_attempt_is_refused(tmp_path):
    current=tmp_path/'current';episode=current/'episode';episode.mkdir(parents=True)
    raw=(b'{"seed":3,"label":[1,2,3]}\n')*1000
    source=episode/'closed_loop_labels.jsonl';source.write_bytes(raw)
    assert finalize_closed_label_log(episode,current) is None and source.read_bytes()==raw
    (episode/'result.json').write_text('{}');(episode/'collection.json').write_text('{}')
    assert finalize_closed_label_log(episode,current) is None and source.read_bytes()==raw
    (episode/'post_stage_memory_release.json').write_text('{}')
    record=finalize_closed_label_log(episode,current)
    assert gzip.open(episode/record['stored_name'],'rb').read()==raw
    assert record['original_sha256']==hashlib.sha256(raw).hexdigest()
    assert not source.exists() and record['stored_bytes']<len(raw)
    assert finalize_closed_label_log(episode,current) is None
    old=tmp_path/'old'/'episode';old.mkdir(parents=True)
    with pytest.raises(ValueError,match='current-attempt'):finalize_closed_label_log(old,current)


def test_preexisting_destination_never_removes_original(tmp_path):
    episode=tmp_path/'episode';episode.mkdir()
    for name in ('result.json','collection.json','post_stage_memory_release.json'):(episode/name).write_text('{}')
    source=episode/'closed_loop_labels.jsonl';source.write_bytes(b'original\n')
    (episode/'closed_loop_labels.jsonl.gz').write_bytes(b'not-a-gzip-stream')
    with pytest.raises(FileExistsError):finalize_closed_label_log(episode,tmp_path)
    assert source.read_bytes()==b'original\n'


def test_budget_stop_retains_original_before_temporary_write(tmp_path,monkeypatch):
    episode=tmp_path/'episode';episode.mkdir()
    for name in ('result.json','collection.json','post_stage_memory_release.json'):(episode/name).write_text('{}')
    source=episode/'closed_loop_labels.jsonl';source.write_bytes(b'original\n')
    def refused(self,write_bytes):raise OSError('budget stop')
    monkeypatch.setattr('tabletop.compact_current_logs.StorageGuard.check',refused)
    with pytest.raises(OSError,match='budget stop'):finalize_closed_label_log(episode,tmp_path)
    assert source.read_bytes()==b'original\n'
    assert not list(episode.glob('*.pending')) and not list(episode.glob('*.gz'))
