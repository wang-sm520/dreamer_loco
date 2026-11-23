"""Bounded diagnostic-only GPU restore launcher with the W8 resource guards."""
import argparse,fcntl,json,os,signal,subprocess,time,hashlib
from pathlib import Path
from .gpu_guard import gpu_snapshot
from .spec import OUTPUT,W8,WORKTREE
from .storage import StorageGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,choices=[0,2],default=0)
    p.add_argument('--num-envs',type=int,choices=[8,16,32,64,128,256,512,1024,2048,4096],required=True)
    p.add_argument('--capacity-only',action='store_true');p.add_argument('--visual-diagnostic',action='store_true');p.add_argument('--standing-audit',choices=['none','no-render','render'],default='none');p.add_argument('--seed',type=int,default=256);p.add_argument('--sampling',type=Path);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--record-root',type=Path,required=True);p.add_argument('--capacity-evidence',type=Path)
    p.add_argument('--capacity-cohorts',type=int,default=1);p.add_argument('--replay-sampling-dir',type=Path)
    p.add_argument('--resume-student',type=Path);p.add_argument('--skip-teacher-first',action='store_true')
    p.add_argument('--resume-state',type=Path);p.add_argument('--recovery-root',type=Path);p.add_argument('--allow-partial-recovery',action='store_true')
    p.add_argument('--resume-sha256')
    p.add_argument('--allow-closed-collection-recovery',action='store_true')
    p.add_argument('--capacity-decisions-per-env',type=int,default=2);p.add_argument('--capacity-teacher-control',action='store_true');a=p.parse_args();a.capacity_only=True
    capacity=None;training=None
    if Path(a.name).name!=a.name:raise ValueError('New simple attempt name required')
    if not a.capacity_only and not a.visual_diagnostic and a.standing_audit=='none':
        if a.capacity_evidence is None:raise ValueError('Real visual/update capacity evidence required')
        capacity=json.loads(a.capacity_evidence.read_text())
        training=json.loads(Path(capacity['training_status']).read_text())
        if not capacity.get('completed') or capacity.get('num_envs')!=a.num_envs or training.get('optimizer_updates',0)<1:
            raise ValueError('Capacity evidence is incomplete or uses another actual environment count')
        if len(training.get('rounds',[]))<3:raise ValueError('Three consecutive visual/update cohorts required after host-memory failure')
        if training.get('capacity_decisions_per_env',0)<75 or any(row.get('last_cohort_physical_steps',0)<1500 for row in training['rounds']):
            raise ValueError('Short lifecycle tests do not establish long-cohort memory stability')
        inventory={Path(x['path']).name:x['sha256'] for x in json.loads((a.capacity_evidence.parent/'source_inventory.json').read_text())}
        # Runner/recovery-only fixes are CPU validated under coordinator v6;
        # the measured physical, visual and optimization path stays identical.
        for name in ['student.py','visual.py','learning.py','learning_episode.py',
                     'isaac_scene.py','geometry.py','closed_loop.py','audit_all_rigid.py','learning_geometry.py','lifecycle_cleanup.py']:
            current=hashlib.sha256((WORKTREE/'legmm_skill/tabletop'/name).read_bytes()).hexdigest()
            if inventory.get(name)!=current:raise ValueError('Visual capacity source changed: '+name)
    lock=(OUTPUT/'.isaac_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');guard.check()
    card=gpu_snapshot(a.gpu)
    if card['used_mib']>4096 or card['free_mib']<18*1024:raise RuntimeError('Shared GPU resource gate')
    available_start=int(next(row for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:')).split()[1])*1024
    if capacity:
        peak_rss=max(capacity['peak_own_rss_bytes'],training.get('latest_save',{}).get('process_peak_rss_bytes',0),training.get('full_restore_peak_rss_bytes',0))
    else:
        old=OUTPUT/f'visual_capacity_true4_retry1_n{a.num_envs}_01/status.json'
        peak_rss=json.loads(old.read_text()).get('peak_own_rss_bytes',0) if old.exists() else 8*1024**3+a.num_envs*100_000_000
        peak_rss+=4*1024**3 # bounded buffer + atomic serialization diagnostic allowance
    required_start=peak_rss+12*1024**3
    if available_start<required_start:raise MemoryError(f'RAM startup margin: available={available_start}, expected_peak={peak_rss}, required={required_start}')
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
    command=['nice','-n','10','/home/bdi/miniconda3/envs/legmm_isaac/bin/python','-s','-m','tabletop.foot_gpu_proof',
        '--out',str(out),'--qualification',str(a.qualification),'--record-root',str(a.record_root),
        '--num-envs',str(a.num_envs),'--device','cuda:0','--kit_args',kit]
    if a.capacity_only:command.append('--capacity-only')
    command.extend(['--capacity-cohorts',str(a.capacity_cohorts)])
    command.extend(['--capacity-decisions-per-env',str(a.capacity_decisions_per_env)])
    if a.capacity_teacher_control:command.append('--capacity-teacher-control')
    if a.resume_state:command.extend(['--resume-state',str(a.resume_state)])
    if a.resume_sha256:command.extend(['--resume-sha256',a.resume_sha256])
    if a.recovery_root:command.extend(['--recovery-root',str(a.recovery_root)])
    if a.allow_partial_recovery:command.append('--allow-partial-recovery')
    if a.allow_closed_collection_recovery:command.append('--allow-closed-collection-recovery')
    if a.replay_sampling_dir:command.extend(['--replay-sampling-dir',str(a.replay_sampling_dir)])
    if a.resume_student:command.extend(['--resume-student',str(a.resume_student)])
    if a.skip_teacher_first:command.append('--skip-teacher-first')
    if a.visual_diagnostic:command.append('--visual-diagnostic')
    if a.standing_audit!='none':command.extend(['--standing-audit',a.standing_audit])
    command.extend(['--seed',str(a.seed)])
    if a.sampling:command.extend(['--sampling',str(a.sampling)])
    status=dict(PID=None,supervisor_pid=os.getpid(),completed=False,phase='starting_visual_pipeline',
        formal_training=not (a.capacity_only or a.visual_diagnostic or a.standing_audit!='none'),num_envs=a.num_envs,CUDA_VISIBLE_DEVICES=str(a.gpu),config=command,
        peak_gpu_mib=0,peak_own_rss_bytes=0,elapsed_s=0.,startup_available_ram_bytes=available_start,
        expected_peak_rss_bytes=peak_rss,required_startup_available_bytes=required_start,soft_stop_ram_bytes=12*1024**3,hard_stop_ram_bytes=8*1024**3)
    def save():(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    with (out/'run.log').open('x') as log,(out/'resources.jsonl').open('x') as resources:
        child=subprocess.Popen(command,cwd=WORKTREE/'legmm_skill',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status['PID']=child.pid;save();start=time.monotonic();error=None;stop_requested_at=None;hard_stop=False
        try:
            while child.poll() is None:
                card_now=gpu_snapshot(a.gpu,child.pid,card['baseline_overhead_mib']);owned=card_now['owned_processes']
                own=card_now['owned_upper_bound_mib']
                meminfo=Path('/proc/meminfo').read_text().splitlines()
                available=int(next(x for x in meminfo if x.startswith('MemAvailable:')).split()[1])*1024
                disk=guard.check()
                process_rows=subprocess.check_output(['ps','-eo','pid,pgid,rss','--no-headers'],text=True).splitlines()
                own_rss=sum(int(parts[2])*1024 for row in process_rows if len(parts:=row.split())==3 and int(parts[1])==child.pid)
                status.update(peak_gpu_mib=max(status['peak_gpu_mib'],own),peak_own_rss_bytes=max(status['peak_own_rss_bytes'],own_rss),
                    minimum_available_ram_bytes=min(status.get('minimum_available_ram_bytes',available),available),elapsed_s=time.monotonic()-start)
                resources.write(json.dumps(dict(elapsed_s=status['elapsed_s'],own_gpu_mib=own,
                    known_own_gpu_mib=card_now['known_own_mib'],external_gpu_mib=card_now['external_upper_bound_mib'],
                    unattributed_gpu_mib=card_now['unattributed_mib'],gpu_total_before_mib=card_now['total_before_mib'],
                    gpu_total_after_mib=card_now['total_after_mib'],gpu_processes=card_now['processes'],
                    available_ram_bytes=available,own_rss_bytes=own_rss,**disk))+'\n');resources.flush();save()
                if card_now['wrong_gpu_allocations']:hard_stop=True;raise RuntimeError('Wrong GPU allocation: '+str(card_now['wrong_gpu_allocations']))
                if own>18*1024:hard_stop=True;raise RuntimeError(f'Own GPU exceeds18GiB: {own}MiB')
                if card_now['external_upper_bound_mib']>4096:raise RuntimeError('Shared GPU external occupancy exceeds4GiB: '+str(card_now['external_processes']))
                if available<8*1024**3:hard_stop=True;raise MemoryError(f'Host available memory below8GiB: available={available}, own_rss={own_rss}')
                if available<12*1024**3 and stop_requested_at is None:
                    stop_requested_at=time.monotonic();os.killpg(child.pid,signal.SIGTERM)
                    status['safe_stop_requested']=dict(available_ram_bytes=available,own_rss_bytes=own_rss,elapsed_s=status['elapsed_s']);save()
                if stop_requested_at is not None and time.monotonic()-stop_requested_at>60:hard_stop=True;raise RuntimeError('Safe stop boundary timeout60s')
                time.sleep(2)
        except BaseException as e:
            error=repr(e)
            if child.poll() is None:
                if hard_stop:os.killpg(child.pid,signal.SIGKILL)
                else:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        code=child.wait()
    training=out/'training_status.json';state=json.loads(training.read_text()) if training.exists() else {}
    status.update(exit_code=code,reason=error or state.get('reason'),completed=bool(code==0 and not error and state.get('completed')),
        training_status=str(training),elapsed_s=time.monotonic()-start)
    save();print(json.dumps(status),flush=True)
    raise SystemExit(0 if status['completed'] else 1)


if __name__=='__main__':main()
