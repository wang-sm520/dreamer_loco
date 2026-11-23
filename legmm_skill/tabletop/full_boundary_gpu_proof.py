"""Real CUDA raw/gzip full-boundary equivalence, without scenes or learning."""
import argparse,gzip,hashlib,json,os,random,resource,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--record-root',type=Path,required=True)
    p.add_argument('--resume-state',type=Path,required=True)
    p.add_argument('--resume-sha256',required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    a,_=p.parse_known_args();a.headless=True;a.enable_cameras=True
    expected=json.loads((a.record_root/'source_cpu_audit.json').read_text())
    assert expected['completed'] and expected['source_path']==str(a.resume_state)
    assert expected['source_sha256']==a.resume_sha256
    assert expected['model_updates']==6000 and expected['collection']['seen']==96000
    app=AppLauncher(a).app
    try:
        import numpy as np
        import torch
        from .learning import DaggerSession
        from .compressed_recovery_candidate import recovery_source
        from .state_fingerprints import session_fingerprints,digest,capture_rng
        from .lifecycle_cleanup import install_stop_handler
        install_stop_handler()
        source=Path(__file__).with_name('dagger_recovery.py').read_text()
        ns={'__name__':'tabletop._full_boundary_gzip_proof','__package__':'tabletop'}
        exec(compile(source if 'if compress:data=gzip.compress' in source else recovery_source(source),'full_boundary_gzip_proof','exec'),ns)
        session=DaggerSession('cuda:0',256)
        results={}
        for name,path,sha in [('raw',a.resume_state,a.resume_sha256),('gzip',Path(expected['archive_path']),expected['archive_sha256'])]:
            started=time.monotonic()
            _,cursor=ns['restore_latest'](session,path,expected_sha256=sha,expected_formal=True,expected_num_envs=64)
            actual=session_fingerprints(session)
            checks={key:actual[key]==value for key,value in expected['source_fingerprints'].items()}
            assert all(checks.values()),checks
            rng=session.reservoir.rng.getstate();batch=session.reservoir.sample(64);session.reservoir.rng.setstate(rng)
            assert digest(batch)==expected['expected_next_batch64_sha256']
            states=capture_rng(True)
            pr=random.Random();pr.setstate(states['python']);nr=np.random.RandomState();nr.set_state(states['numpy'])
            tc=torch.Generator(device='cpu');tc.set_state(states['torch'])
            tg=torch.Generator(device='cuda:0');tg.set_state(states['cuda'][0])
            random_checks=dict(python=pr.random()==random.random(),numpy=float(nr.random_sample())==float(np.random.random()),
                torch_cpu=torch.equal(torch.rand(8,generator=tc),torch.rand(8)),
                torch_cuda=torch.equal(torch.rand(8,device='cuda:0',generator=tg),torch.rand(8,device='cuda:0')))
            random.setstate(states['python']);np.random.set_state(states['numpy']);torch.set_rng_state(states['torch']);torch.cuda.set_rng_state_all(states['cuda'])
            assert all(random_checks.values())
            assert session.student.updates==6000 and session.seen==96000 and cursor==expected['cursor']
            results[name]=dict(state_checks=checks,next_sample64_exact=True,next_random_checks=random_checks,
                cursor=cursor,restore_elapsed_s=time.monotonic()-started)
        decoded=hashlib.sha256()
        with gzip.open(expected['archive_path'],'rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):decoded.update(block)
        assert decoded.hexdigest()==a.resume_sha256
        report=dict(PID=os.getpid(),completed=True,diagnostic_only=True,formal_training=False,
            formal_env_decisions_added=0,optimizer_updates_added=0,scene_created=False,physics_steps=0,
            source_sha256=a.resume_sha256,source_path=str(a.resume_state),original_raw_source_preserved=True,
            archive_sha256=expected['archive_sha256'],decoded_original_sha256=decoded.hexdigest(),
            comparisons=results,model_updates=session.student.updates,chain_seen=session.seen,
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (a.out/'training_status.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.record_root/'full_boundary_GPU_proof.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:app.close()

if __name__=='__main__':main()
