"""Real CUDA validation of the explicit partial-validation recovery protocol."""
import argparse,json,os,random,resource,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--record-root',type=Path,required=True)
    p.add_argument('--resume-state',type=Path,required=True);p.add_argument('--resume-sha256',required=True)
    p.add_argument('--validation-recovery-manifest',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a,_=p.parse_known_args();a.headless=True;a.enable_cameras=True
    expected=json.loads((a.record_root/'source_cpu_audit.json').read_text())
    assert expected['completed'] and expected['source']==str(a.resume_state) and expected['source_sha256']==a.resume_sha256
    app=AppLauncher(a).app
    try:
        import numpy as np
        import torch
        from .learning import DaggerSession
        from .dagger_recovery import restore_latest
        from .state_fingerprints import session_fingerprints,digest,capture_rng
        from .lifecycle_cleanup import install_stop_handler,check_stop
        install_stop_handler();session=DaggerSession('cuda:0',256);started=time.monotonic()
        status,cursor=restore_latest(session,a.resume_state,expected_sha256=a.resume_sha256,expected_formal=True,validation_manifest=a.validation_recovery_manifest)
        actual=session_fingerprints(session);checks={key:actual[key]==value for key,value in expected['source_fingerprints'].items()}
        assert all(checks.values()),checks
        state=session.reservoir.rng.getstate();assert digest(session.reservoir.sample(64))==expected['expected_next_batch64_sha256'];session.reservoir.rng.setstate(state)
        rng=capture_rng(True);pr=random.Random();pr.setstate(rng['python']);nr=np.random.RandomState();nr.set_state(rng['numpy'])
        tc=torch.Generator(device='cpu');tc.set_state(rng['torch']);tg=torch.Generator(device='cuda:0');tg.set_state(rng['cuda'][0])
        random_checks=dict(python=pr.random()==random.random(),numpy=float(nr.random_sample())==float(np.random.random()),torch_cpu=torch.equal(torch.rand(8,generator=tc),torch.rand(8)),torch_cuda=torch.equal(torch.rand(8,device='cuda:0',generator=tg),torch.rand(8,device='cuda:0')))
        assert all(random_checks.values())
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
        assert cursor==expected['cursor'] and session.seen==96000 and session.student.updates==6000
        check_stop('validation recovery proof complete')
        report=dict(proof_output_root=str(a.out),PID=os.getpid(),completed=True,phase='validation_recovery_CUDA_verified',formal_training=False,diagnostic_only=True,formal_env_decisions_added=0,optimizer_updates_added=0,scene_created=False,physics_steps=0,source_path=str(a.resume_state),source_sha256=a.resume_sha256,state_checks=checks,next_sample64_exact=True,next_random_checks=random_checks,cursor=cursor,model_updates=session.student.updates,chain_seen=session.seen,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_s=time.monotonic()-started)
        (a.out/'training_status.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.record_root/'validation_recovery_GPU_proof.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:app.close()
if __name__=='__main__':main()
