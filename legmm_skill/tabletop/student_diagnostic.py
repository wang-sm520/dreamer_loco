"""Read-only train-seed action diagnosis; no val/test model tuning or updates."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .spec import OUTPUT,WORKTREE,STEP_LENGTH


def prepare(source,out):
    import torch
    from PIL import Image
    from airbot_depth.depth import DepthAnythingTransform,DEFAULT_DEPTH_CONFIG
    from airbot_rl.policy import head_depth
    from airbot_rl.depth import encode_wrist_depth
    from .storage import StorageGuard
    if not out.resolve().is_relative_to(OUTPUT):raise ValueError('W8 diagnostic output required')
    guard=StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop');guard.check();out.mkdir(exist_ok=False)
    result=json.loads((source/'result.json').read_text())
    if any(not 0<=s['seed']<10000 for s in result['states']):raise ValueError('Only train-seed causal diagnosis')
    states={s['seed']:i for i,s in enumerate(result['states'])}
    labels={}
    for line in (source/'closed_loop_labels.jsonl').open():
        row=json.loads(line);labels[row['seed'],row['frame']]=row
    transform=DepthAnythingTransform(DEFAULT_DEPTH_CONFIG,device='cpu',local_files_only=True)
    rows=[];images=[];proprio=[];commands=[];targets=[];skipped=[]
    for entry in json.loads((source/'visual_diagnostics/index.json').read_text()):
        folder=Path(entry['path']);meta=json.loads((folder/'metadata.json').read_text());seed=meta['seed'];frame=meta['frame']
        row=labels[seed,frame];i=states[seed];joints=result['initial_joint_positions'][i];names=result['joint_names']
        opening=row['metadata'].get('opening')
        if opening is None:
            if frame!=0:
                skipped.append(dict(seed=seed,frame=frame,reason='Exact measured finger opening unavailable in this diagnostic log'));continue
            opening=joints[names.index('endleft')]-joints[names.index('endright')]
        q=np.asarray([*row['measured_arm'],opening],dtype=np.float32)
        previous=q.copy() if frame==0 else np.asarray(labels[seed,frame-16]['label'][3],dtype=np.float32)
        rgb=torch.from_numpy(np.array(Image.open(folder/'head_processed_rgb.png')))[None]
        wrist=torch.from_numpy(np.load(folder/'wrist_processed_metres.npy'))[None]
        with torch.no_grad():image=torch.stack((head_depth(transform,rgb,batch=1),encode_wrist_depth(wrist)),1)[0].numpy()
        consistency={}
        for view,name in [(0,'head_da2'),(1,'wrist_encoded')]:
            stored=np.array(Image.open(folder/(name+'.png'))).astype(np.float32)/255
            consistency[name]=dict(mean_abs_error_to_quantized_preview=float(np.abs(image[view,0]-stored).mean()),
                max_abs_error_to_quantized_preview=float(np.abs(image[view,0]-stored).max()))
        images.append(image.astype(np.float16));proprio.append(q);commands.append(previous);targets.append(row['label'])
        rows.append(dict(seed=seed,frame=frame,phase=meta['phase'],source=str(folder),consistency=consistency))
    guard.check(20_000_000)
    np.savez(out/'observations.npz',images=np.stack(images),proprio=np.stack(proprio),last_command=np.stack(commands),labels=np.asarray(targets,dtype=np.float32))
    (out/'status.json').write_text(json.dumps(dict(completed=True,formal_training=False,geometry_or_success_evaluation=False,
        phase='CPU_reprocessed_train_diagnostic',source=str(source),samples=len(rows),skipped=skipped,
        cpu_preprocessing_not_claimed_bitwise_equal_to_GPU=True,depth_provenance=transform.provenance(),rows=rows),indent=2))


def inspect(checkpoint,cache,out):
    import torch
    from .student import Student
    if out.exists():raise FileExistsError(out)
    data=np.load(cache/'observations.npz');metadata=json.loads((cache/'status.json').read_text())
    student=Student('cpu',0);student.restore(checkpoint)
    with torch.no_grad():
        raw=torch.cat([student.prediction(data['images'][i:i+8],data['proprio'][i:i+8]) for i in range(0,len(data['images']),8)]).numpy()
        actions=student.act(data['images'],data['proprio'],data['last_command'],microbatch=8).numpy()
    target=data['labels'];hold=data['last_command'][:,None,:]
    step=STEP_LENGTH[None,None,:]
    error=(actions[:,:4]-target[:,:4])/step
    motion=(target[:,:4]-hold)/step;predicted=(actions[:,:4]-hold)/step
    norms=np.linalg.norm(motion,axis=2)*np.linalg.norm(predicted,axis=2)
    cosine=np.sum(motion*predicted,axis=2)/np.maximum(norms,1e-12)
    moving=norms>1e-8
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        train_diagnostic_only=True,success_rate_not_measured=True,samples=len(target),cache=str(cache),
        raw_joint_rmse=np.sqrt(np.mean((raw-target)**2,axis=(0,1))).tolist(),
        governed_first4_step_unit_rmse=float(np.sqrt(np.mean(error**2))),
        hold_last_command_step_unit_rmse=float(np.sqrt(np.mean(motion**2))),
        moving_direction_cosine=float(cosine[moving].mean()) if moving.any() else None,
        rows=[dict(meta,step_unit_rmse=float(np.sqrt(np.mean(e**2)))) for meta,e in zip(metadata['rows'],error)])
    out.write_text(json.dumps(report,indent=2)+'\n')
    return report


def inspect_reservoir(checkpoint,out,count=256):
    """Exact saved actor inputs; never reconstruct observations from truth."""
    import torch,gc
    from .student import Student
    if out.exists():raise FileExistsError(out)
    with checkpoint.open('rb') as file:
        digest=hashlib.sha256()
        while chunk:=file.read(8*1024**2):digest.update(chunk)
        file.seek(0);payload=torch.load(file,map_location='cpu',weights_only=False)
    if payload['format'] not in ('w8_dagger_full_round_v1','w8_dagger_partial_v1'):raise ValueError('Full saved DAgger actor-input buffer required')
    student=Student('cpu',0);student.model.load_state_dict(payload['student']['model']);student.model.eval()
    if not torch.equal(student.mean,payload['student']['mean']) or not torch.equal(student.scale,payload['student']['scale']):raise ValueError('Diagnostic normalization differs')
    indices=np.random.default_rng(0).choice(len(payload['reservoir']['items']),size=min(count,len(payload['reservoir']['items'])),replace=False)
    rows=[payload['reservoir']['items'][i] for i in indices]
    model_updates=payload['student']['updates'];formal=payload['status']['formal_training']
    images=np.stack([row[0] for row in rows]);q=np.stack([row[1] for row in rows]);target=np.stack([row[2] for row in rows])
    metadata=[row[3] for row in rows];del payload,rows;gc.collect()
    with torch.no_grad():prediction=torch.cat([student.prediction(images[i:i+8],q[i:i+8]) for i in range(0,len(q),8)]).numpy()
    errors=(prediction-target)/STEP_LENGTH[None,None,:]
    motion=(target[:,:4]-q[:,None,:])/STEP_LENGTH[None,None,:]
    proposed=(prediction[:,:4]-q[:,None,:])/STEP_LENGTH[None,None,:]
    norms=np.linalg.norm(motion,axis=2)*np.linalg.norm(proposed,axis=2)
    cosine=np.sum(motion*proposed,axis=2)/np.maximum(norms,1e-12)
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest.hexdigest(),formal_model=formal,
        model_updates=model_updates,train_diagnostic_only=True,success_rate_not_measured=True,
        exact_stored_actor_observations=True,samples=len(q),sample_indices=indices.tolist(),
        native_joint_rmse=np.sqrt(np.mean((prediction-target)**2,axis=(0,1))).tolist(),
        first4_step_unit_rmse=float(np.sqrt(np.mean(errors[:,:4]**2))),
        copy_proprio_first4_step_unit_rmse=float(np.sqrt(np.mean(motion**2))),
        raw_delta_direction_cosine=float(cosine[norms>1e-8].mean()),
        last_command_not_reconstructed=True,raw_prediction_not_governed_rollout=True,
        rows=[dict(meta,first4_step_unit_rmse=float(np.sqrt(np.mean(error[:4]**2)))) for meta,error in zip(metadata,errors)])
    out.write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--prepare',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--saved-reservoir',action='store_true')
    a=p.parse_args()
    if a.prepare:prepare(a.source,a.prepare)
    elif a.saved_reservoir:print(json.dumps(inspect_reservoir(a.checkpoint,a.out),indent=2))
    else:print(json.dumps(inspect(a.checkpoint,a.cache,a.out),indent=2))


if __name__=='__main__':main()
