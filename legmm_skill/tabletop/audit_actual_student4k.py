"""Bounded CPU inference from real round2 reservoir rows; zero training/Isaac.

Only same-weight round2 rows with recorded previous commands are replayed.
CPU reconstructed actions are distinguished from recorded online target3.
"""
import argparse,collections,gzip,hashlib,json,resource,time,datetime
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    if available<20*1024**3:raise MemoryError('Bounded CPU audit startup20GiB margin')
    import torch
    from .student import Student
    from .state_fingerprints import digest
    from .storage import StorageGuard
    from .spec import OUTPUT,WORKTREE
    torch.set_num_threads(2);a.out.mkdir(exist_ok=False);started=time.monotonic()
    source=OUTPUT/'dagger_true4_visual64_retry1_resume_02/latest.pt';expected='d0927058632044f05b65d1e19dc9b5f356062c27cfcc28d864aa8a4c483cca2e';h=hashlib.sha256()
    with source.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
        assert h.hexdigest()==expected;f.seek(0);payload=torch.load(f,map_location='cpu',weights_only=False)
    assert payload['student']['updates']==4000 and payload['collection']['round']==2 and payload['cursor']['updates_completed_in_round']==0
    e=Path(payload['status']['current_episode']);result=json.loads((e/'result.json').read_text());label=e/'closed_loop_labels.jsonl.gz';opener=gzip.open if label.exists() else open
    if not label.exists():label=e/'closed_loop_labels.jsonl'
    with opener(label,'rt') as f:logs={(x['seed'],x['frame']):x for x in map(json.loads,f)}
    first4=set(range(result['seed'],result['seed']+4));items=payload['reservoir']['items'];eligible=[];tracking=[]
    for index,row in enumerate(items):
        meta=row[3];key=(meta['seed'],meta['frame']);log=logs.get(key)
        if meta['round']!=2 or log is None:continue
        tick=300+meta['frame'];age=(tick-1-10)%11+1;queue=log['metadata'].get('pending_queue_last')
        if age==11 and queue is not None:tracking.append(np.asarray(row[1])-np.asarray(queue))
        if meta['seed'] in first4 and 'actual_previous_command' in log:eligible.append((key,index,log))
    eligible.sort();assert eligible,'No same-weight actual rows with recorded command'
    pick=np.linspace(0,len(eligible)-1,min(32,len(eligible)),dtype=int);selected=[eligible[i] for i in pick]
    images=np.stack([items[index][0] for _,index,_ in selected]);q=np.stack([items[index][1] for _,index,_ in selected]);teacher=np.stack([items[index][2] for _,index,_ in selected]);previous=np.asarray([log['actual_previous_command'] for _,_,log in selected],np.float32)
    for i,(key,index,log) in enumerate(selected):
        assert np.array_equal(q[i,:6],np.asarray(log['measured_arm'],np.float32));assert np.array_equal(teacher[i],np.asarray(log['label'],np.float32));assert log['queue_tick_next']==300+key[1]
    student=Student('cpu',256);student.model.load_state_dict(payload['student']['model']);student.model.eval();student.updates=4000
    assert torch.equal(student.mean,payload['student']['mean']) and torch.equal(student.scale,payload['student']['scale']);before=digest(student.model.state_dict())
    raw=[];governed=[]
    with torch.no_grad():
        for start in range(0,len(selected),8):
            free=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
            if free<12*1024**3:raise MemoryError('CPU inference audit12GiB floor')
            raw.append(student.prediction(images[start:start+8],q[start:start+8]).numpy());governed.append(student.act(images[start:start+8],q[start:start+8],previous[start:start+8],microbatch=8).numpy())
    raw=np.concatenate(raw);governed=np.concatenate(governed);assert digest(student.model.state_dict())==before==digest(payload['student']['model'])
    aligned=[];queue_matches=[]
    for i,((seed,frame),_,log) in enumerate(selected):
        nxt=logs.get((seed,frame+16));queue=log['metadata'].get('pending_queue_last');age=(300+frame-1-10)%11+1
        if age<=4 and queue is not None:queue_matches.append(float(abs(previous[i]-np.asarray(queue)).max()))
        if nxt and 'actual_previous_command' in nxt:aligned.append(governed[i,3]-np.asarray(nxt['actual_previous_command']))
    metric=lambda error:dict(arm_RMSE_rad=np.sqrt(np.mean(error[...,:6]**2,axis=tuple(range(error.ndim-1)))).tolist(),native_opening_RMSE_mm=float(np.sqrt(np.mean(error[...,6]**2))*1000))
    alignment=np.asarray(aligned);maximum=abs(alignment).max(0).tolist() if aligned else None
    save=a.out/'cpu_reconstructed_actions.npz';StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop').check(200_000)
    np.savez_compressed(save,cpu_raw=raw,cpu_governed=governed,teacher=teacher,proprio=q,actual_previous_requested=previous,seeds=np.asarray([k[0] for k,_,_ in selected]),frames=np.asarray([k[1] for k,_,_ in selected]),buffer_indices=np.asarray([i for _,i,_ in selected]),image_sha256=np.asarray([hashlib.sha256(items[i][0].tobytes()).hexdigest() for _,i,_ in selected]))
    record=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,diagnostic_only=True,formal_training=False,formal_interactions_added=0,optimizer_updates_added=0,source_sha256=expected,source_model_updates=4000,same_weight_round2_only=True,real_reservoir_images_not_approximated=True,source_model_hash=before,selected_rows=len(selected),eligible_rows=len(eligible),raw_vs_teacher=metric(raw-teacher),governed_first4_vs_teacher=metric(governed[:,:4]-teacher[:,:4]),observed_vs_last_requested_contains_delay_not_tracking=metric(q-previous),applied_reference_age11_rows=len(tracking),applied_reference_age11_tracking=metric(np.asarray(tracking)) if tracking else None,actual_next_block_target3_comparisons=len(aligned),max_CPU_replay_vs_actual_target3_error_native=maximum,CPU_command_alignment_within_tolerance=bool(maximum is not None and max(maximum[:6])<=1e-5 and maximum[6]<=1e-6),latest_queue_slot_crosscheck_rows=len(queue_matches),latest_queue_slot_max_error=max(queue_matches) if queue_matches else None,teacher_phase_tags=dict(collections.Counter(log['metadata']['phase'] for _,_,log in selected)),artifact=str(save),artifact_sha256=hashlib.sha256(save.read_bytes()).hexdigest(),source_unmodified=True,CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_s=time.monotonic()-started,limits=['CPU reconstruction is not a saved raw online GPU prediction; backend numerical differences apply','Only first4 training seeds have exact recorded previous commands; reservoir is sampled subset','Only target3 is aligned against next-block online command; full4-target online history absent','Age11 applied-reference subset uses original fixed-slot queue semantics; all other requested-command gaps include delay','Native opening is endleft-endright joint coordinate, not literal pad-face gap','No training or physical success qualification from this diagnostic'])
    (a.out/'status.json').write_text(json.dumps(record,indent=2)+'\n');Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_true4_contact_retry1_20261115/v10_actual_student4k_action_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
