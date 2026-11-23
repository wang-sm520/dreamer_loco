"""Single-owned-Isaac paired independent test launcher with the W8 resource guards."""
import argparse,fcntl,json,os,signal,subprocess,time,hashlib
from pathlib import Path
from .gpu_guard import gpu_snapshot
from .spec import OUTPUT,W8,WORKTREE
from .storage import StorageGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,choices=[0,2],default=0)
    p.add_argument('--num-envs',type=int,choices=[32],default=32)
    p.add_argument('--student-status',type=Path,required=True);p.add_argument('--student-checkpoint',type=Path,required=True);p.add_argument('--ppo-status',type=Path,required=True);p.add_argument('--sampling-dir',type=Path,required=True);p.add_argument('--capacity-only',action='store_true');p.add_argument('--visual-diagnostic',action='store_true');p.add_argument('--standing-audit',choices=['none','no-render','render'],default='none');p.add_argument('--seed',type=int,default=256);p.add_argument('--sampling',type=Path);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--record-root',type=Path,required=True);p.add_argument('--capacity-evidence',type=Path);a=p.parse_args()
    if Path(a.name).name!=a.name:raise ValueError('New simple attempt name required')
    from .train_ppo import student_qualification,teacher_qualification
    student_qualification(a.student_status,a.student_checkpoint);teacher_qualification(a.qualification)
    state=json.loads(a.ppo_status.read_text())
    if not state.get('completed') or state.get('phase')!='ppo_budget_complete':raise ValueError('Complete actual PPO before final test')
    if a.capacity_only or a.visual_diagnostic or a.standing_audit!='none' or a.sampling:raise ValueError('Use dedicated diagnostics; this is final test')
    lock=(OUTPUT/'.isaac_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');guard.check()
    card=gpu_snapshot(a.gpu)
    if card['used_mib']>4096 or card['free_mib']<18*1024:raise RuntimeError('Shared GPU resource gate')
    # Conservative measured long-visual cohort admission also covers final-test N32.
    memory_evidence=a.capacity_evidence or OUTPUT/'visual_capacity_cleanup_true4_retry1_n64_02/status.json'
    measured=json.loads(memory_evidence.read_text())
    if not measured.get('completed') or measured.get('num_envs',0)<a.num_envs:
        raise ValueError('Completed measured RAM evidence at least as large as requested cohort required')
    measured_training=json.loads(Path(measured['training_status']).read_text())
    peak_rss=max(measured['peak_own_rss_bytes'],measured_training.get('latest_save',{}).get('process_peak_rss_bytes',0),measured_training.get('full_restore_peak_rss_bytes',0))
    available_start=int(next(row for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:')).split()[1])*1024
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
    command=['nice','-n','10','/home/bdi/miniconda3/envs/legmm_isaac/bin/python','-s','-m','tabletop.final_test',
        '--out',str(out),'--qualification',str(a.qualification),'--record-root',str(a.record_root),
        '--num-envs',str(a.num_envs),'--device','cuda:0','--kit_args',kit]
    command.extend(['--student-status',str(a.student_status),'--student-checkpoint',str(a.student_checkpoint),'--ppo-status',str(a.ppo_status),'--sampling-dir',str(a.sampling_dir)])
    if a.capacity_only:command.append('--capacity-only')
    if a.visual_diagnostic:command.append('--visual-diagnostic')
    if a.standing_audit!='none':command.extend(['--standing-audit',a.standing_audit])
    command.extend(['--seed',str(a.seed)])
    if a.sampling:command.extend(['--sampling',str(a.sampling)])
    status=dict(PID=None,supervisor_pid=os.getpid(),completed=False,phase='starting_independent_test',
        formal_training=False,num_envs=a.num_envs,CUDA_VISIBLE_DEVICES=str(a.gpu),config=command,
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
