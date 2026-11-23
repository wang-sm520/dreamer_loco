"""Bounded CPU and exact logged-command audit; never touches an active process."""
import argparse,gzip,hashlib,json,gc,datetime
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from .spec import STEP_LENGTH,LOWER,UPPER,OUTPUT,STOW
from .student import Student
from .lifecycle_cleanup import memory_snapshot,trim_unused


def metric(error):
    error=np.asarray(error,dtype=float).reshape(-1,7)
    if not len(error):return dict(targets=0)
    rmse=np.sqrt(np.mean(error**2,axis=0));mae=np.mean(abs(error),axis=0)
    return dict(targets=len(error),arm_RMSE_rad=rmse[:6].tolist(),opening_RMSE_m=float(rmse[6]),
        opening_RMSE_mm=float(rmse[6]*1000),arm_MAE_rad=mae[:6].tolist(),opening_MAE_mm=float(mae[6]*1000),
        per_axis_RMSE_native_steps=(rmse/STEP_LENGTH).tolist())


def slot_sample(frame,warmup=300):
    """Actual runtime writes queue[tick%11], metadata logs queue[10]."""
    tick=warmup+frame
    written=tick-1-(tick-1-10)%11
    action_frame=written-warmup;block=(action_frame//16)*16
    return dict(observation_frame=frame,queue_tick=tick,slot_write_action_frame=action_frame,
        source_block_frame=block,target_index=(action_frame-block)//4,age_physics_steps=tick-written)


def read_labels(folder):
    raw=folder/'closed_loop_labels.jsonl';packed=folder/'closed_loop_labels.jsonl.gz'
    opener=raw.open if raw.exists() else lambda mode:gzip.open(packed,mode)
    with opener('rt') as file:
        for line in file:yield json.loads(line)


def execution_audit(result,rows):
    groups=defaultdict(list)
    for row in rows:groups[row['seed']].append(row)
    outcomes={r['seed']:r for r in result['episodes']};errors=[];fully=[];hold=[];latest=[];applied=[]
    phases=defaultdict(list);indices=Counter();skipped=Counter();cases=[];calibration=[]
    for seed,seq in groups.items():
        outcome=outcomes[seed];gate=(seq[0]['metadata'].get('reason') or '').startswith('initial_gate:')
        for previous,following in zip(seq,seq[1:]):
            frame=previous['frame'];sample=slot_sample(following['frame'])
            if sample['source_block_frame']!=frame:raise ValueError('Nonadjacent logged action blocks')
            if gate:skipped['initial_gate_external_hold']+=1;continue
            if frame>=outcome['steps']:skipped['post_terminal_external_hold']+=1;continue
            index=sample['target_index'];actual=np.asarray(following['metadata']['pending_queue_last'])
            target=np.asarray(previous['label'])[index];error=actual-target
            errors.append(error);indices[index]+=1;phases[previous['metadata']['phase']].append(error)
            if following['frame']<outcome['steps']:fully.append(error)
            observed_q=np.r_[following['measured_arm'],following['metadata']['opening']]
            if following['frame']<outcome['steps']:
                if index==3:latest.append(actual-observed_q)
                if sample['age_physics_steps']==11:applied.append(actual-observed_q)
        if (outcome['reason'] or '').startswith(('collision','drop')):
            contact_frame=outcome['steps']-1
            before=max((x for x in seq if x['frame']<=contact_frame),key=lambda x:x['frame'])
            after=next((x for x in seq if x['frame']==before['frame']+16),None)
            sampled=None
            if after is not None:
                sampled=slot_sample(after['frame']);sampled['actual_native_target']=after['metadata']['pending_queue_last']
                sampled['target_was_active_at_failure']=sampled['slot_write_action_frame']//4==contact_frame//4
                sampled['teacher_same_target']=before['label'][sampled['target_index']]
            cases.append(dict(seed=seed,outcome=outcome['reason'],terminal_step=outcome['steps'],terminal_time_s=outcome['steps']*.01,
                last_observation_frame=before['frame'],observation_before_failure_s=(outcome['steps']-before['frame'])*.01,
                shadow_teacher_phase=before['metadata']['phase'],shadow_teacher_reason=before['metadata']['reason'],
                measured_arm_rad=before['measured_arm'],measured_opening_m=before['metadata']['opening'],
                measured_object_pose=before['measured_object_pose'],sampled_actual_command=sampled,
                contact_points_and_full_four_target_history_available=False))
    return dict(scope='Actual governed native targets sampled by fixed ring slot; not raw network predictions',
        selected_target_indices=dict(indices),skipped_blocks=dict(skipped),generated_preterminal_target_error=metric(errors),
        fully_preterminal_block_error=metric(fully),by_shadow_teacher_phase={k:metric(v) for k,v in phases.items()},
        latest_command_tracking_subset=metric(latest),applied_command_tracking_subset=metric(applied),
        tracking_scope='Latest subset uses target_index3; applied subset uses slot age11. Both include real PD/load response, latest additionally100ms delay.',
        failure_order=cases,unavailable=['All raw online student predictions','First executed target index0 and unsampled targets1..3','Per-frame camera tensors for this student cohort','Per-frame base/body poses and contact points'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--teacher-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--samples',type=int,default=128);p.add_argument('--save-tensors',action='store_true');a=p.parse_args()
    if not 1<=a.samples<=128 or not a.out.resolve().is_relative_to(OUTPUT):raise ValueError('Bounded W8-only inference')
    if memory_snapshot()['available_ram_bytes']<24*1024**3:raise MemoryError('CPU audit admission24GiB')
    a.out.mkdir(exist_ok=False);result=json.loads((a.episode/'result.json').read_text());rows=list(read_labels(a.episode))
    report=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),formal_training=False,Isaac_App_started=False,
        optimizer_updates=0,new_environment_interactions=0,actual_cohort=execution_audit(result,rows),
        correction='Earlier pending_queue_last-minus-measured aggregate was fixed-slot error, not exact latest-command tracking.',
        control_implementation_changed=False,live_process_state_accessed=False)
    (a.out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    with a.checkpoint.open('rb') as file:
        digest=hashlib.sha256()
        while block:=file.read(8*1024**2):digest.update(block)
        if digest.hexdigest()!=a.sha256:raise ValueError('Formal snapshot SHA differs')
        file.seek(0);payload=torch.load(file,map_location='cpu',weights_only=False)
    if payload['format']!='w8_dagger_full_round_v1' or not payload['status']['formal_training']:raise ValueError('Formal complete snapshot required')
    student=Student('cpu',0);student.model.load_state_dict(payload['student']['model']);student.model.eval()
    if not torch.equal(student.mean,payload['student']['mean']) or not torch.equal(student.scale,payload['student']['scale']):raise ValueError('Normalization differs')
    updates=payload['student']['updates'];items=payload['reservoir']['items'];indices=np.random.default_rng(0).choice(len(items),min(a.samples,len(items)),replace=False)
    selected=[items[i] for i in indices]
    if any(not 0<=row[3]['seed']<10000 for row in selected):raise ValueError('Train-only audit')
    del items,payload;gc.collect();trim_unused()
    required=set()
    for row in selected:
        seed,frame=row[3]['seed'],row[3]['frame'];required.update(((seed,frame),(seed,frame-16),(seed,frame+16)))
    found={}
    for folder in sorted(a.teacher_root.glob('round_00_episode_*')):
        for row in read_labels(folder):
            key=(row['seed'],row['frame'])
            if key in required:found[key]=row
    images=np.stack([x[0] for x in selected]);q=np.stack([x[1] for x in selected]);labels=np.stack([x[2] for x in selected]);previous=[];phases=[];calibration=[]
    for item in selected:
        key=(item[3]['seed'],item[3]['frame']);row=found[key];frame=key[1]
        if not np.array_equal(np.asarray(row['label'],np.float32),item[2]):raise ValueError('Stored labels and raw teacher log differ')
        if not np.array_equal(np.asarray(row['measured_arm']+[row['metadata']['opening']],np.float32),item[1]):raise ValueError('Stored proprio and raw log differ')
        previous.append(row['metadata']['pending_queue_last'] if frame==0 else found[(key[0],frame-16)]['label'][3]);phases.append(row['metadata']['phase'])
        following=found.get((key[0],frame+16))
        if following:
            sample=slot_sample(frame+16);expected=np.asarray(row['label'][sample['target_index']],np.float32)
            actual=np.asarray(following['metadata']['pending_queue_last'],np.float32);calibration.append(float(np.max(abs(expected-actual))))
    previous=np.asarray(previous,np.float32);raw=[];projected=[];processed=0
    with torch.inference_mode():
        for i in range(0,len(selected),8):
            if memory_snapshot()['available_ram_bytes']<20*1024**3:break
            raw.append(student.prediction(images[i:i+8],q[i:i+8]).numpy())
            projected.append(student.act(images[i:i+8],q[i:i+8],previous[i:i+8],microbatch=8).numpy());processed+=len(q[i:i+8])
    if not processed:raise MemoryError('CPU audit yields before first batch')
    raw=np.concatenate(raw);projected=np.concatenate(projected);labels=labels[:processed];previous=previous[:processed]
    if not all(np.isfinite(x).all() for x in (raw,projected,labels)):raise FloatingPointError('Nonfinite action audit')
    delta=np.diff(np.concatenate((previous[:,None],projected),axis=1),axis=1)
    if np.max(abs(delta)/STEP_LENGTH)>1.0001 or np.any(projected<LOWER-1e-6) or np.any(projected>UPPER+1e-6):raise ValueError('Native projection bound violated')
    report['stored_teacher_state_CPU_inference']=dict(source_checkpoint=str(a.checkpoint),source_sha256=a.sha256,model_updates=updates,
        samples=processed,requested=a.samples,phase_counts=dict(Counter(phases[:processed])),raw_first4=metric(raw[:,:4]-labels[:,:4]),
        governed_first4=metric(projected[:,:4]-labels[:,:4]),raw_all8=metric(raw-labels),governed_all8=metric(projected-labels),
        raw_outside_joint_limits_fraction=np.mean((raw<LOWER)|(raw>UPPER),axis=(0,1)).tolist(),max_native_step_ratio=float(np.max(abs(delta)/STEP_LENGTH)),
        teacher_ring_slot_calibration_samples=len(calibration),teacher_ring_slot_max_abs_error=max(calibration,default=None),
        actor_inputs=['stored_depth_float16_then_float32','measured_q7'],input_rounding='AirbotPolicyModel rounds online inputs through float16 too',
        last_command_source='Previous teacher-driven block label index3; frame0 settled STOW queue',
        limitation='These are real stored teacher-visited states, not historical student640-703 images. CPU forward is not recorded CUDA output; raw online student history cannot be reconstructed.')
    details=report['stored_teacher_state_CPU_inference']
    details['by_phase']={phase:dict(samples=int(sum(x==phase for x in phases[:processed])),
        raw_first4=metric(raw[np.asarray(phases[:processed])==phase,:4]-labels[np.asarray(phases[:processed])==phase,:4]),
        governed_first4=metric(projected[np.asarray(phases[:processed])==phase,:4]-labels[np.asarray(phases[:processed])==phase,:4]))
        for phase in sorted(set(phases[:processed]))}
    details['opening_raw_range_m']=[float(raw[...,6].min()),float(raw[...,6].max())]
    details['native_opening_definition']='measured endleft-endright joint displacement, m; physical pad-face gap also depends on fixed asset geometry'
    if a.save_tensors:
        import io
        from .storage import StorageGuard
        stream=io.BytesIO()
        np.savez_compressed(stream,raw=raw,governed=projected,labels=labels,proprio=q[:processed],previous_command=previous,
            phase=np.asarray(phases[:processed]),buffer_index=indices[:processed],
            seed=np.asarray([x[3]['seed'] for x in selected[:processed]]),frame=np.asarray([x[3]['frame'] for x in selected[:processed]]),
            images_sha256=np.asarray([hashlib.sha256(x[0].tobytes()).hexdigest() for x in selected[:processed]]))
        data=stream.getvalue();StorageGuard(OUTPUT,Path(__file__).parent).check(len(data));path=a.out/'teacher_state_prediction_audit.npz'
        with path.open('xb') as file:file.write(data)
        details['inspectable_prediction_arrays']=dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),images_copied=False)
    report.update(completed=processed==len(selected),CPU_inference_only=True)
    (a.out/'status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['stored_teacher_state_CPU_inference']))


if __name__=='__main__':main()
