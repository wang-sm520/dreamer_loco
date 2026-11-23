"""Exact lossless alternatives for four old diagnostic model-only checkpoint copies."""
import gzip,hashlib,json,os
from pathlib import Path
from .storage import StorageGuard
from .spec import OUTPUT,WORKTREE

def main():
    root=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_true4_contact_retry1_20261115')
    assert json.loads((root/'coordinator_update_ack.json').read_text())['version']>=9
    reports=[]
    for folder,name in [('visual_capacity_cleanup_true4_retry1_n64_02','capacity_restore_test_02.pt'),('visual_capacity_true4_retry1_n64_01','capacity_restore_test.pt'),('visual_capacity_true4_retry1_n128_01','capacity_restore_test.pt'),('visual_capacity_true4_retry1_n32_02','capacity_restore_test.pt')]:
        parent=OUTPUT/folder;status=json.loads((parent/'training_status.json').read_text());assert status['diagnostic_only'] and not status['formal_training']
        source=parent/name;target=source.with_suffix('.pt.gz');temp=source.with_suffix('.pt.gz.pending');assert not target.exists() and not temp.exists()
        size=source.stat().st_size;StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop').check(size+size//100+4096);h=hashlib.sha256()
        with source.open('rb') as f,temp.open('xb') as sink:
            with gzip.GzipFile(filename='',mode='wb',fileobj=sink,compresslevel=3,mtime=0) as z:
                for b in iter(lambda:f.read(1024**2),b''):h.update(b);z.write(b)
            sink.flush();os.fsync(sink.fileno())
        actual=hashlib.sha256();count=0
        with gzip.open(temp,'rb') as f:
            for b in iter(lambda:f.read(1024**2),b''):actual.update(b);count+=len(b)
        assert count==size and actual.hexdigest()==h.hexdigest();os.rename(temp,target)
        report=dict(original=str(source),original_bytes=size,original_sha256=h.hexdigest(),stored=str(target),stored_bytes=target.stat().st_size,exact_roundtrip=True,diagnostic_only=True,logical_data_preserved=True,authority='coordinator_v9_lossless_alternative_before_redundant_raw_removal')
        source.with_suffix('.pt.archive.json').write_text(json.dumps(report,indent=2)+'\n');source.unlink();reports.append(report)
    (root/'v9_small_diagnostic_archives.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
if __name__=='__main__':main()
