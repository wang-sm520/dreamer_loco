"""Lossless finalization of closed label logs from the current owned attempt.

This never touches prior attempts, checkpoints, physics results, or open cohorts.
The raw SHA and byte count remain inspectable beside the verified gzip stream.
"""
import gzip,hashlib,json,os
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard

CURRENT_ATTEMPT=OUTPUT/'dagger_true4_visual64_retry1_resume_01'


def finalize_closed_label_log(episode,allowed_root=CURRENT_ATTEMPT,guard=None):
    episode=Path(episode);allowed_root=Path(allowed_root)
    if episode.is_symlink() or episode.parent.resolve()!=allowed_root.resolve():
        raise ValueError('Only direct current-attempt cohort directories may be finalized')
    if not all((episode/name).is_file() for name in ('result.json','collection.json','post_stage_memory_release.json')):
        return None # post-stage marker follows run_episode finally/closed writers
    source=episode/'closed_loop_labels.jsonl';target=episode/'closed_loop_labels.jsonl.gz';manifest=episode/'label_log_storage.json'
    if not source.exists():return None
    if source.is_symlink() or target.exists() or manifest.exists():raise FileExistsError('Refuse ambiguous or pre-existing output')
    # Account for the complete simultaneous raw + gzip temporary allocation.
    size=source.stat().st_size
    if guard is None:guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    guard.check(size+(size//16384+1)*16+1024)
    temporary=episode/'closed_loop_labels.jsonl.gz.pending';digest=hashlib.sha256();count=0
    with source.open('rb') as raw,temporary.open('xb') as sink:
        with gzip.GzipFile(filename='',mode='wb',fileobj=sink,compresslevel=6,mtime=0) as zipped:
            while block:=raw.read(1024**2):digest.update(block);count+=len(block);zipped.write(block)
        sink.flush();os.fsync(sink.fileno())
    actual=hashlib.sha256();expanded=0
    with gzip.open(temporary,'rb') as restored:
        while block:=restored.read(1024**2):actual.update(block);expanded+=len(block)
    if actual.hexdigest()!=digest.hexdigest() or expanded!=count:raise RuntimeError('Lossless log verification failed; original retained')
    os.rename(temporary,target)
    record=dict(format='lossless_gzip_label_log_v1',original_name=source.name,stored_name=target.name,
        original_bytes=count,original_sha256=digest.hexdigest(),stored_bytes=target.stat().st_size,
        complete_result=str(episode/'result.json'),closed_stage_marker=str(episode/'post_stage_memory_release.json'),logical_content_unchanged=True)
    with manifest.open('x') as file:json.dump(record,file,indent=2);file.write('\n');file.flush();os.fsync(file.fileno())
    # Only this current-attempt raw file is removed, after exact roundtrip proof.
    source.unlink()
    descriptor=os.open(episode,os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return record
