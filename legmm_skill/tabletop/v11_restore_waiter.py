"""Resource wait followed only by the raw CUDA recovery diagnostic."""
import datetime,json,subprocess,time
from pathlib import Path
from .spec import OUTPUT,WORKTREE
ROOT=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_resource_abort_continue_20261115')
def main():
    source=json.loads((ROOT/'v11_actual_interrupted_source_CPU_audit.json').read_text())
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
        admitted=[]
        for row in rows.splitlines():
            index,used,free,util=map(int,row.split(','))
            if index in (0,2) and used<=4096 and free>=18432:admitted.append(index)
        status=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='waiting_for_GPU_raw_restore_proof',GPU_rows=rows,available_ram_bytes=available,formal_training_running=False,W8_completed=False,source=source['source_path'],source_sha256=source['source_sha256'])
        if admitted and available>=30*1024**3:
            gpu=0 if 0 in admitted else 2
            command=['/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.gpu_restore_launch','--name','v11_resource_abort_raw_GPU_restore_01','--gpu',str(gpu),'--num-envs','64','--qualification',str(OUTPUT/'teacher_true4_pregrasp_fixed256_retry1_01/summary.json'),'--record-root',str(ROOT),'--resume-state',source['source_path'],'--resume-sha256',source['source_sha256']]
            status.update(phase='launching_raw_GPU_restore_proof',command=command)
            (ROOT/'v11_GPU_admission_wait.json').write_text(json.dumps(status,indent=2)+'\n')
            code=subprocess.call(command,cwd=WORKTREE/'legmm_skill')
            status.update(phase='raw_GPU_supervisor_exited_requires_evidence_review',exit_code=code)
            (ROOT/'v11_GPU_admission_wait.json').write_text(json.dumps(status,indent=2)+'\n');return
        (ROOT/'v11_GPU_admission_wait.json').write_text(json.dumps(status,indent=2)+'\n')
        time.sleep(30)
if __name__=='__main__':main()
