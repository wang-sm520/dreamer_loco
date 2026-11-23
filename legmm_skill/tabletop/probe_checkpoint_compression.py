"""Read-only bounded streaming compression/roundtrip probe; no saved model copy."""
import hashlib,json,time,zlib,resource
from pathlib import Path

def main():
    root=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_true4_contact_retry1_20261115');source=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_1_random_reachable_20261114/dagger_true4_visual64_retry1_resume_01/latest.pt')
    compressor=zlib.compressobj(3,zlib.DEFLATED,31);restore=zlib.decompressobj(31);rawhash=hashlib.sha256();decoded=hashlib.sha256();packedhash=hashlib.sha256();raw_size=packed_size=decoded_size=0;start=time.monotonic()
    def take(part):
        nonlocal packed_size,decoded_size
        packedhash.update(part);packed_size+=len(part);data=restore.decompress(part);decoded.update(data);decoded_size+=len(data)
    with source.open('rb') as file:
        for block in iter(lambda:file.read(1024**2),b''):
            rawhash.update(block);raw_size+=len(block);take(compressor.compress(block))
        take(compressor.flush());tail=restore.flush();decoded.update(tail);decoded_size+=len(tail)
    assert restore.eof and decoded_size==raw_size and rawhash.hexdigest()==decoded.hexdigest()=='2a09afa4dd8c1036ab0a8611e1a690b56860f84c76409b3b1a68900eb7ecaf98'
    report=dict(source=str(source),raw_bytes=raw_size,raw_sha256=rawhash.hexdigest(),gzip_bytes=packed_size,gzip_sha256=packedhash.hexdigest(),roundtrip_exact=True,source_unmodified=True,new_checkpoint_file_written=False,elapsed_s=time.monotonic()-start,CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,formal_model_updates_added=0,formal_env_decisions_added=0,actual_GPU_compressed_restore_verified=False)
    (root/'v9_actual_source_compression_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
