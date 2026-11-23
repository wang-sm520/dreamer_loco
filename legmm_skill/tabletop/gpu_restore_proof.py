"""Diagnostic-only real CUDA restore, with no scene, interaction or updates."""
import argparse,hashlib,json,os,random
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--record-root',type=Path,required=True);p.add_argument('--resume-state',type=Path,required=True);p.add_argument('--resume-sha256',required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a,unused=p.parse_known_args();a.headless=True;a.enable_cameras=True
    app=AppLauncher(a).app
    import numpy as np
    import torch
    from .learning import DaggerSession
    from .dagger_recovery import restore_latest
    from .state_fingerprints import session_fingerprints,digest,capture_rng
    from .lifecycle_cleanup import install_stop_handler
    install_stop_handler()
    expected=json.loads((a.record_root/'v11_actual_interrupted_source_CPU_audit.json').read_text())
    assert expected['source_sha256']==a.resume_sha256
    session=DaggerSession('cuda:0',256)
    state,cursor=restore_latest(session,a.resume_state,allow_resource_abort=True,expected_sha256=a.resume_sha256,expected_formal=True,expected_num_envs=64)
    actual=session_fingerprints(session)
    checks={key:actual[key]==value for key,value in expected['source_fingerprints'].items()}
    assert all(checks.values()),checks
    rng=session.reservoir.rng.getstate();batch=session.reservoir.sample(64);session.reservoir.rng.setstate(rng)
    assert digest(batch)==expected['expected_next_batch64_sha256']
    states=capture_rng(True)
    pr=random.Random();pr.setstate(states['python']);nr=np.random.RandomState();nr.set_state(states['numpy'])
    tc=torch.Generator(device='cpu');tc.set_state(states['torch']);tg=torch.Generator(device='cuda:0');tg.set_state(states['cuda'][0])
    random_checks=dict(python=pr.random()==random.random(),numpy=float(nr.random_sample())==float(np.random.random()),torch_cpu=torch.equal(torch.rand(8,generator=tc),torch.rand(8)),torch_cuda=torch.equal(torch.rand(8,device='cuda:0',generator=tg),torch.rand(8,device='cuda:0')))
    random.setstate(states['python']);np.random.set_state(states['numpy']);torch.set_rng_state(states['torch']);torch.cuda.set_rng_state_all(states['cuda'])
    assert all(random_checks.values())
    assert session.student.updates==4000 and session.seen==67776 and cursor['next_seed']==1088 and cursor['episode']==13
    h=hashlib.sha256()
    with a.resume_state.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==a.resume_sha256
    result=dict(PID=os.getpid(),completed=True,diagnostic_only=True,formal_training=False,formal_env_decisions_added=0,optimizer_updates_added=0,scene_created=False,physics_steps=0,source_sha256=a.resume_sha256,source_unmodified=True,state_checks=checks,next_sample64_exact=True,next_random_checks=random_checks,cursor=cursor,model_updates=session.student.updates,chain_seen=session.seen)
    (a.out/'training_status.json').write_text(json.dumps(result,indent=2)+'\n');(a.record_root/'v11_real_GPU_restore_proof.json').write_text(json.dumps(result,indent=2)+'\n')
    app.close()
if __name__=='__main__':main()
