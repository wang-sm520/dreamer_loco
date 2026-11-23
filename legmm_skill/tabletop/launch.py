"""Own-process GPU 0/2 guard, new output paths, detached-safe scalar logging."""
import argparse,csv,json,os,signal,subprocess,time,hashlib,shutil,fcntl
from datetime import datetime,timezone
from pathlib import Path
from .spec import OUTPUT,W8,WORKTREE
from .storage import StorageGuard,FLOOR_BYTES,BUDGET_BYTES


def gpu_snapshot(gpu):
    rows=list(csv.reader(subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
    r=rows[0]
    processes=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True).splitlines()))
    return dict(index=int(r[0]),uuid=r[1].strip(),used_mib=int(r[2]),free_mib=int(r[3]),
                processes=[dict(uuid=p[0].strip(),pid=int(p[1]),memory_mib=int(p[2])) for p in processes if len(p)==3 and p[2].strip().isdigit()])


def completed_probe(exit_code,result,expected_envs,error_exists=False):
    # Kit's fast shutdown can turn a Python exception into exit code zero.
    # Require physical evidence, not just a successful process exit.
    return bool(exit_code==0 and not error_exists and isinstance(result,dict)
        and result.get('physical_steps',0)>0 and result.get('num_envs')==expected_envs
        and len(result.get('episodes',[]))==expected_envs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,choices=[0,2],default=0)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--num-envs',type=int,choices=[16,32],default=32)
    p.add_argument('--mode',choices=['stand','teacher','contact','self-contact','w2-reference'],default='stand')
    p.add_argument('--audit-no-gravity-compensation',action='store_true')
    p.add_argument('--audit-w2-model',action='store_true')
    p.add_argument('--audit-yaw-zero',action='store_true')
    p.add_argument('--audit-payload',type=float)
    p.add_argument('--audit-w2-self-collision-off',action='store_true')
    p.add_argument('--audit-reset-noise',action='store_true')
    p.add_argument('--audit-w2-robot-material',action='store_true')
    p.add_argument('--audit-scene-solver-4-0',action='store_true')
    p.add_argument('--audit-scene-solver-unclamped',action='store_true')
    p.add_argument('--residual-velocity',action='store_true')
    p.add_argument('--sampling',type=Path)
    p.add_argument('--trace-teacher',action='store_true')
    p.add_argument('--closed-loop-teacher',action='store_true')
    a=p.parse_args()
    if Path(a.name).name!=a.name: raise ValueError('Simple new attempt name required')
    owned_lock=(OUTPUT/'.isaac_job.lock').open('a')
    fcntl.flock(owned_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');guard.check()
    card=gpu_snapshot(a.gpu)
    if card['used_mib']>4*1024 or card['free_mib']<18*1024:
        raise RuntimeError(f'GPU resource gate: {card}')
    out=OUTPUT/a.name;out.mkdir(exist_ok=False)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    inventory=[]
    for source in sorted((WORKTREE/'legmm_skill/tabletop').glob('*.py')):
        shutil.copy2(source,snapshot/source.name)
        inventory.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (out/'source_inventory.json').write_text(json.dumps(inventory,indent=2))
    for name in ['cache','config','data','tmp','kit']:(OUTPUT/name).mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        HF_HUB_OFFLINE='1',HF_HOME=str(W8/'huggingface'),XDG_CACHE_HOME=str(OUTPUT/'cache'),
        XDG_CONFIG_HOME=str(OUTPUT/'config'),XDG_DATA_HOME=str(OUTPUT/'data'),TMPDIR=str(OUTPUT/'tmp'),
        CUDA_CACHE_PATH=str(OUTPUT/'cache/cuda'),__GL_SHADER_DISK_CACHE_PATH=str(OUTPUT/'cache/gl'),
        OMNI_KIT_ACCEPT_EULA='YES',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    kit=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0 --portable-root {OUTPUT}/kit --/plugins/carb.tasking.plugin/threadCount=2 --/plugins/omni.tbb.globalcontrol/maxThreadCount=2'
    command=['nice','-n','10','/home/bdi/miniconda3/envs/legmm_isaac/bin/python','-s','-m','tabletop.isaac_probe',
        '--out',str(out),'--num-envs',str(a.num_envs),'--seed',str(a.seed),'--mode',a.mode,'--device','cuda:0','--kit_args',kit]
    if a.trace_teacher:command.append('--trace-teacher')
    if a.closed_loop_teacher:command.append('--closed-loop-teacher')
    if a.audit_no_gravity_compensation:command.append('--audit-no-gravity-compensation')
    if a.audit_w2_model:command.append('--audit-w2-model')
    if a.audit_yaw_zero:command.append('--audit-yaw-zero')
    if a.audit_payload is not None:command.extend(['--audit-payload',str(a.audit_payload)])
    if a.audit_w2_self_collision_off:command.append('--audit-w2-self-collision-off')
    if a.audit_reset_noise:command.append('--audit-reset-noise')
    if a.audit_w2_robot_material:command.append('--audit-w2-robot-material')
    if a.audit_scene_solver_4_0:command.append('--audit-scene-solver-4-0')
    if a.audit_scene_solver_unclamped:command.append('--audit-scene-solver-unclamped')
    if a.residual_velocity:command.append('--residual-velocity')
    if a.sampling:command.extend(['--sampling',str(a.sampling)])
    if a.mode=='w2-reference':
        command=['nice','-n','10','/home/bdi/miniconda3/envs/legmm_isaac/bin/python','-s','-m','tabletop.w2_reference',
            '--out',str(out),'--num-envs',str(a.num_envs),'--device','cuda:0','--kit_args',kit]
    status=dict(config=command,seed=a.seed,steps=0,checkpoint=None,CUDA_VISIBLE_DEVICES=str(a.gpu),PID=None,
                completed=False,phase='cpu_sampling_before_isaac',gpu=card,start_utc=datetime.now(timezone.utc).isoformat(),
                disk_floor_bytes=FLOOR_BYTES,task_budget_bytes=BUDGET_BYTES,checkpoint_count=0)
    def write_status():
        guard.check(16384)
        (out/'status.json').write_text(json.dumps(status,indent=2))
    write_status();start=time.monotonic();peak=0;reason=None
    with (out/'run.log').open('x') as log,(out/'resources.jsonl').open('x') as resources:
        job=subprocess.Popen(command,cwd=WORKTREE/'legmm_skill',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status['PID']=job.pid;write_status()
        def stop_owned():
            if job.poll() is None:
                os.killpg(job.pid,signal.SIGTERM)
                try:job.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL)
        try:
            while job.poll() is None:
                current=gpu_snapshot(a.gpu);owned=[]
                for process in current['processes']:
                    try:
                        if os.getpgid(process['pid'])==job.pid:owned.append(process)
                    except ProcessLookupError:pass
                own=sum(x['memory_mib'] for x in owned);peak=max(peak,own)
                if own and status['phase']=='cpu_sampling_before_isaac':
                    status['phase']='isaac_gpu_running';status['first_gpu_elapsed_s']=time.monotonic()-start;write_status()
                measurement=guard.check()
                resources.write(json.dumps(dict(elapsed_s=time.monotonic()-start,own_gpu_mib=own,
                    gpu_used_mib=current['used_mib'],**measurement))+'\n');resources.flush()
                if any(x['uuid']!=card['uuid'] for x in owned):raise RuntimeError('Own task appeared on unselected GPU')
                if own>18*1024:raise RuntimeError('Own GPU memory exceeds 18 GiB')
                if current['used_mib']-own>4*1024:raise RuntimeError('External GPU occupancy exceeds 4 GiB')
                time.sleep(2)
        except BaseException as error:
            reason=repr(error);stop_owned()
        code=job.wait()
    result=None
    if (out/'result.json').exists():
        result=json.loads((out/'result.json').read_text());status['steps']=result['physical_steps']
        status['success_count']=sum(x['success'] for x in result['episodes'])
    valid=completed_probe(code,result,a.num_envs,(out/'error.json').exists())
    if not valid and reason is None:reason='missing_or_failed_physical_result'
    status.update(completed=valid and reason is None,exit_code=code,reason=reason,peak_gpu_mib=peak,elapsed_s=time.monotonic()-start)
    try:write_status()
    except OSError: print('Cannot safely persist final status: '+json.dumps(status),flush=True)
    print(json.dumps(status),flush=True)
    raise SystemExit(0 if status['completed'] else 1)

if __name__=='__main__':main()
