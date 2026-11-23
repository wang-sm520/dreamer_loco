"""Preserve every byte of one superseded capacity checkpoint before removing raw copy."""
import gzip,hashlib,json,os
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard

def main():
    root=OUTPUT/'visual_capacity_cleanup_true4_retry1_n64_02';source=root/'latest.pt';dest=root/'latest.pt.gz';temp=root/'latest.pt.gz.pending'
    # Explicit coordinator v9 permits nonunique diagnostic checkpoint cleanup;
    # keep all bytes here, rather than discard the diagnostic buffer.
    status=json.loads((root/'training_status.json').read_text());assert status['diagnostic_only'] and not status['formal_training']
    proof=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_true4_contact_retry1_20261115/v9_real_GPU_restore_proof.json');assert json.loads(proof.read_text())['completed']
    size=source.stat().st_size;StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop').check(size+size//100+4096)
    assert not dest.exists() and not temp.exists();h=hashlib.sha256()
    with source.open('rb') as file,temp.open('xb') as sink:
        with gzip.GzipFile(filename='',mode='wb',fileobj=sink,compresslevel=3,mtime=0) as zipped:
            for block in iter(lambda:file.read(1024**2),b''):h.update(block);zipped.write(block)
        sink.flush();os.fsync(sink.fileno())
    verify=hashlib.sha256();count=0
    with gzip.open(temp,'rb') as file:
        for block in iter(lambda:file.read(1024**2),b''):verify.update(block);count+=len(block)
    assert h.hexdigest()==verify.hexdigest() and count==size
    os.rename(temp,dest)
    record=dict(original_path=str(source),original_bytes=size,original_sha256=h.hexdigest(),stored_path=str(dest),stored_bytes=dest.stat().st_size,lossless_roundtrip_verified=True,diagnostic_only=True,formal_sources_untouched=True,reason='coordinator_v9_bounded_storage; exact compressed alternative before removing redundant raw representation')
    (root/'latest_archive.json').write_text(json.dumps(record,indent=2)+'\n');source.unlink()
    proof.with_name('v9_diagnostic_checkpoint_lossless_archive.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
