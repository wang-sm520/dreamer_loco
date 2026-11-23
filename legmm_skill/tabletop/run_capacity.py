"""Sequential true-vision/update capacity ladder, stopping before predicted exhaustion."""
import argparse,json,subprocess,time,re,datetime,hashlib
from pathlib import Path
from .spec import OUTPUT,WORKTREE
from .storage import StorageGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--record-root',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--prefix',required=True)
    p.add_argument('--expected-coordinator-version',type=int,default=2);p.add_argument('--attempt',type=int,default=1);p.add_argument('--start-envs',type=int,default=32)
    a=p.parse_args();r=a.record_root;results=[];guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop')
    state=dict(phase='visual_capacity_ladder',capacity_target=4096,completed=False,formal_training=False,
        tested=[],actual_max_passed=None,formal_config=None)
    def save():(r/'visual_capacity_status.json').write_text(json.dumps(state,indent=2)+'\n')
    for n in [32,64,128,256,512,1024,2048,4096]:
        if n<a.start_envs:continue
        instructions=(r/'COORDINATOR_UPDATES.md').read_text();print(instructions,flush=True)
        match=re.search(r'当前有效版本：\s*(\d+)',instructions)
        version=int(match[1]) if match else 1
        if version!=a.expected_coordinator_version:raise RuntimeError('New coordinator version requires active executor review')
        ack=json.loads((r/'coordinator_update_ack.json').read_text());ack.update(version=version,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='visual_capacity',
            coordinator_updates_sha256=hashlib.sha256(instructions.encode()).hexdigest())
        (r/'coordinator_update_ack.json').write_text(json.dumps(ack,indent=2))
        if results:
            last=results[-1];peak=last['peak_gpu_mib'];baseline=last['startup_min_nonzero_gpu_mib']
            prediction=baseline+(peak-baseline)*n/last['num_envs']
            # GPU allocation can jump at a tiled atlas boundary. Keep a 10%
            # measured-growth margin plus 512 MiB rather than provoke OOM.
            upper=prediction*1.10+512
            state['next_preflight']=dict(num_envs=n,predicted_mib=prediction,conservative_mib=upper)
            if upper>18*1024:
                state.update(stop_reason='next_level_conservative_gpu_prediction_exceeds_18GiB',untested_from=n);save();break
        name=f'{a.prefix}_n{n}_{a.attempt:02d}'
        command=['/home/bdi/miniconda3/envs/dreamer_loco/bin/python','-m','tabletop.learning_launch',
            '--name',name,'--num-envs',str(n),'--gpu','0','--capacity-only','--qualification',str(a.qualification),
            '--record-root',str(r)]
        with (r/(name+'_launcher.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=WORKTREE/'legmm_skill',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(current_supervisor_pid=child.pid,current_envs=n,current_output=str(OUTPUT/name));save()
            while child.poll() is None:guard.check();time.sleep(5)
        status=json.loads((OUTPUT/name/'status.json').read_text())
        if child.returncode or not status['completed']:
            state.update(stop_reason='capacity_stage_failed',failure_status=status);save();raise RuntimeError(name)
        training=json.loads((OUTPUT/name/'training_status.json').read_text())
        records=[json.loads(line) for line in (OUTPUT/name/'resources.jsonl').open()]
        gpu=[x['own_gpu_mib'] for x in records if x['own_gpu_mib']>0]
        row=dict(num_envs=n,status_path=str(OUTPUT/name/'status.json'),peak_gpu_mib=status['peak_gpu_mib'],
            peak_own_rss_bytes=status['peak_own_rss_bytes'],startup_min_nonzero_gpu_mib=min(gpu),
            elapsed_s=status['elapsed_s'],env_decisions=training['env_decisions'],
            optimizer_updates=training['optimizer_updates'],rounds=training['rounds'],
            torch_peak_allocated_bytes=training['torch_peak_allocated_bytes'],
            torch_peak_reserved_bytes=training['torch_peak_reserved_bytes'])
        assert training['actual_envs']==n and training['optimizer_updates']>=1
        results.append(row);state.update(tested=results,actual_max_passed=n);save()
    state.update(completed=True,current_supervisor_pid=None,formal_config=None);save()
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
