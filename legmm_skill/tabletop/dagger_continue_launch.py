"""Single-owned-Isaac visual learning launcher with the W8 resource guards."""
import argparse,fcntl,json,os,signal,subprocess,time,hashlib
from pathlib import Path
from .launch import gpu_snapshot
from .spec import OUTPUT,W8,WORKTREE
from .storage import StorageGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,choices=[0,2],default=0)
    p.add_argument('--num-envs',type=int,choices=[8,16,32,64,128,256,512,1024,2048,4096],required=True)
    p.add_argument('--resume-student',type=Path,required=True);p.add_argument('--resume-status',type=Path,required=True);p.add_argument('--start-round',type=int,default=6);p.add_argument('--capacity-only',action='store_true');p.add_argument('--visual-diagnostic',action='store_true');p.add_argument('--standing-audit',choices=['none','no-render','render'],default='none');p.add_argument('--seed',type=int,default=256);p.add_argument('--sampling',type=Path);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--record-root',type=Path,required=True);p.add_argument('--capacity-evidence',type=Path);a=p.parse_args()
    if Path(a.name).name!=a.name:raise ValueError('New simple attempt name required')
    if not a.capacity_only and not a.visual_diagnostic and a.standing_audit=='none':
        if a.capacity_evidence is None:raise ValueError('Real visual/update capacity evidence required')
        capacity=json.loads(a.capacity_evidence.read_text())
        training=json.loads(Path(capacity['training_status']).read_text())
        if not capacity.get('completed') or capacity.get('num_envs')!=a.num_envs or training.get('optimizer_updates',0)<1:
            raise ValueError('Capacity evidence is incomplete or uses another actual environment count')
        inventory={Path(x['path']).name:x['sha256'] for x in json.loads((a.capacity_evidence.parent/'source_inventory.json').read_text())}
        for name in ['student.py','visual.py','learning.py','learning_episode.py','train_dagger.py',
                     'isaac_scene.py','geometry.py','closed_loop.py','audit_all_rigid.py','learning_geometry.py']:
            current=hashlib.sha256((WORKTREE/'legmm_skill/tabletop'/name).read_bytes()).hexdigest()
            if inventory.get(name)!=current:raise ValueError('Visual capacity source changed: '+name)
    lock=(OUTPUT/'.isaac_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');guard.check()
    card=gpu_snapshot(a.gpu)
    if card['used_mib']>4096 or card['free_mib']<18*1024:raise RuntimeError('Shared GPU resource gate')
    out=OUTPUT/a.name;out.mkdir(exist_ok=False)
    import shutil
    snapshot=out/'source_snapshot';snapshot.mkdir()
    inventory=[]
    for source in sorted((WORKTREE/'legmm_skill/tabletop').glob('*.py')):
        shutil.copy2(source,snapshot/source.name)
        inventory.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (out/'source_inventory.json').write_text(json.dumps(inventory,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        HF_HUB_OFFLINE='1',HF_HOME=str(W8/'huggingface'),XDG_CACHE_HOME=str(OUTPUT/'cache'),
        XDG_CONFIG_HOME=str(OUTPUT/'config'),XDG_DATA_HOME=str(OUTPUT/'data'),TMPDIR=str(OUTPUT/'tmp'),
        CUDA_CACHE_PATH=str(OUTPUT/'cache/cuda'),__GL_SHADER_DISK_CACHE_PATH=str(OUTPUT/'cache/gl'),
        OMNI_KIT_ACCEPT_EULA='YES',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    kit=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0 --portable-root {OUTPUT}/kit --/plugins/carb.tasking.plugin/threadCount=2 --/plugins/omni.tbb.globalcontrol/maxThreadCount=2'
    command=['nice','-n','10','/home/bdi/miniconda3/envs/legmm_isaac/bin/python','-s','-m','tabletop.train_dagger_continue',
        '--out',str(out),'--qualification',str(a.qualification),'--record-root',str(a.record_root),
        '--num-envs',str(a.num_envs),'--device','cuda:0','--kit_args',kit]
    command.extend(['--resume-student',str(a.resume_student),'--resume-status',str(a.resume_status),'--start-round',str(a.start_round)])
    if a.capacity_only:command.append('--capacity-only')
    if a.visual_diagnostic:command.append('--visual-diagnostic')
    if a.standing_audit!='none':command.extend(['--standing-audit',a.standing_audit])
    command.extend(['--seed',str(a.seed)])
    if a.sampling:command.extend(['--sampling',str(a.sampling)])
    status=dict(PID=None,supervisor_pid=os.getpid(),completed=False,phase='starting_visual_pipeline',
        formal_training=not (a.capacity_only or a.visual_diagnostic or a.standing_audit!='none'),num_envs=a.num_envs,CUDA_VISIBLE_DEVICES=str(a.gpu),config=command,
        peak_gpu_mib=0,peak_own_rss_bytes=0,elapsed_s=0.)
    def save():(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    with (out/'run.log').open('x') as log,(out/'resources.jsonl').open('x') as resources:
        child=subprocess.Popen(command,cwd=WORKTREE/'legmm_skill',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status['PID']=child.pid;save();start=time.monotonic();error=None
        try:
            while child.poll() is None:
                card_now=gpu_snapshot(a.gpu);owned=[]
                for process in card_now['processes']:
                    try:
                        if os.getpgid(process['pid'])==child.pid:owned.append(process)
                    except ProcessLookupError:pass
                own=sum(x['memory_mib'] for x in owned)
                if any(x['uuid']!=card['uuid'] for x in owned):raise RuntimeError('Wrong GPU')
                if own>18*1024:raise RuntimeError('Own GPU exceeds 18 GiB')
                if card_now['used_mib']-own>4096:raise RuntimeError('Shared GPU exceeds external gate')
                meminfo=Path('/proc/meminfo').read_text().splitlines()
                available=int(next(x for x in meminfo if x.startswith('MemAvailable:')).split()[1])*1024
                if available<8*1024**3:raise MemoryError('Host available memory below 8 GiB')
                disk=guard.check()
                process_rows=subprocess.check_output(['ps','-eo','pid,pgid,rss','--no-headers'],text=True).splitlines()
                own_rss=sum(int(parts[2])*1024 for row in process_rows if len(parts:=row.split())==3 and int(parts[1])==child.pid)
                status.update(peak_gpu_mib=max(status['peak_gpu_mib'],own),peak_own_rss_bytes=max(status['peak_own_rss_bytes'],own_rss),elapsed_s=time.monotonic()-start)
                resources.write(json.dumps(dict(elapsed_s=status['elapsed_s'],own_gpu_mib=own,available_ram_bytes=available,own_rss_bytes=own_rss,**disk))+'\n');resources.flush();save();time.sleep(2)
        except BaseException as e:
            error=repr(e)
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        code=child.wait()
    training=out/'training_status.json';state=json.loads(training.read_text()) if training.exists() else {}
    status.update(exit_code=code,reason=error or state.get('reason'),completed=bool(code==0 and not error and state.get('completed')),
        training_status=str(training),elapsed_s=time.monotonic()-start)
    save();print(json.dumps(status),flush=True)
    raise SystemExit(0 if status['completed'] else 1)


if __name__=='__main__':main()
