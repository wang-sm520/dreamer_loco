"""Preserve an exact compressed archive and audit an actual full-round source."""
import argparse, datetime, gzip, hashlib, json, os, resource, time, zlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--record-root',type=Path,required=True)
    a=p.parse_args()
    from .spec import OUTPUT,WORKTREE
    from .storage import StorageGuard
    assert a.out.resolve().is_relative_to(OUTPUT) and a.source.resolve().is_relative_to(OUTPUT)
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    guard.check()
    a.out.mkdir(exist_ok=False)
    status=dict(PID=os.getpid(),completed=False,diagnostic_only=True,phase='compression_measurement',
        CUDA_VISIBLE_DEVICES='',seed=256,steps=96000,optimizer_updates=6000,checkpoint=str(a.source))
    def persist():
        (a.out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    persist()
    try:
        available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
        if available<20*1024**3:raise MemoryError('CPU source audit requires 20GiB available')
        started=time.monotonic()
        compressor=zlib.compressobj(3,zlib.DEFLATED,31)
        decoder=zlib.decompressobj(31)
        raw_sha=hashlib.sha256();decoded_sha=hashlib.sha256()
        raw_bytes=packed_bytes=decoded_bytes=0
        with a.source.open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):
                raw_sha.update(block);raw_bytes+=len(block)
                packed=compressor.compress(block);packed_bytes+=len(packed)
                decoded=decoder.decompress(packed);decoded_sha.update(decoded);decoded_bytes+=len(decoded)
            packed=compressor.flush();packed_bytes+=len(packed)
            decoded=decoder.decompress(packed)+decoder.flush();decoded_sha.update(decoded);decoded_bytes+=len(decoded)
        assert decoder.eof and decoded_bytes==raw_bytes and raw_sha.hexdigest()==decoded_sha.hexdigest()==a.sha256
        # Exact measured compressed payload, including the simultaneous original.
        disk_before=guard.check(packed_bytes+4096)
        target=a.out/'source.pt.gz';pending=a.out/'source.pt.gz.pending'
        with a.source.open('rb') as raw,pending.open('xb') as sink:
            with gzip.GzipFile(filename='',mode='wb',fileobj=sink,compresslevel=3,mtime=0) as z:
                for block in iter(lambda:raw.read(1024**2),b''):z.write(block)
            sink.flush();os.fsync(sink.fileno())
        verified=hashlib.sha256();expanded=0
        with gzip.open(pending,'rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):verified.update(block);expanded+=len(block)
        assert expanded==raw_bytes and verified.hexdigest()==a.sha256
        os.replace(pending,target)
        import torch
        torch.set_num_threads(2)
        from .state_fingerprints import expected_fingerprints,expected_next_batch
        with a.source.open('rb') as f:payload=torch.load(f,map_location='cpu',weights_only=False)
        assert payload['complete_round_boundary'] and payload['format']=='w8_dagger_full_round_v1'
        assert payload['status']['formal_training'] and not payload['status']['diagnostic_only']
        assert payload['status']['num_envs']==64 and payload['status']['total_env_decision_budget']==192000
        assert payload['collection']['seen']==96000 and payload['student']['updates']==6000
        assert payload['cursor']['next_round']==3 and payload['cursor']['validation_pending']==3
        assert len(payload['reservoir']['items'])==payload['reservoir']['capacity']==10000
        assert payload['reservoir']['seen']==payload['collection']['collected_samples']
        assert sum(payload['reservoir']['composition'].values())==payload['reservoir']['seen']
        archive_sha=hashlib.sha256()
        with target.open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):archive_sha.update(block)
        status.update(completed=True,phase='CPU_source_and_exact_archive_verified',
            source_path=str(a.source),source_sha256=a.sha256,source_bytes=raw_bytes,
            archive_path=str(target),archive_sha256=archive_sha.hexdigest(),archive_bytes=target.stat().st_size,
            lossless_roundtrip_verified=True,original_source_unmodified=True,
            source_fingerprints=expected_fingerprints(payload),expected_next_batch64_sha256=expected_next_batch(payload),
            cursor=payload['cursor'],collection=payload['collection'],model_updates=payload['student']['updates'],
            buffer_frames=len(payload['reservoir']['items']),global_rng_keys=list(payload['rng']),
            remaining_formal_interactions=192000-payload['collection']['seen'],remaining_optimizer_updates=12000-payload['student']['updates'],
            disk_before_archive=disk_before,disk_after=guard.check(),
            CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_s=time.monotonic()-started,GPU_restore_verified=False,
            formal_env_decisions_added=0,optimizer_updates_added=0,
            observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        persist()
        (a.record_root/'source_cpu_audit.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
    except BaseException as e:
        status.update(phase='audit_failed',error=repr(e));persist();raise

if __name__=='__main__':main()
