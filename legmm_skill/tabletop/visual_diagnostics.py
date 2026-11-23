"""Sparse raw camera and ground-truth segmentation evidence, NEVER actor inputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def camera_optical_point(world_point,world_position,quat_wxyz):
    """Metres in, metres out; world camera +X forward/+Z up -> optical +Z/+Y down."""
    from scipy.spatial.transform import Rotation
    q=np.asarray(quat_wxyz)
    local=Rotation.from_quat(q[[1,2,3,0]]).inv().apply(np.asarray(world_point)-world_position)
    return np.array([-local[1],-local[2],local[0]])


def require_fresh_frame(previous,current):
    if current<0 or (previous is not None and current<=previous):
        raise RuntimeError('Camera frame did not advance')


def label_scene(stage,count):
    from isaacsim.core.utils.semantics import add_update_semantics
    for i in range(count):
        for name,label in [('Robot','w8_robot'),('Object','w8_target'),('Table','w8_table')]:
            add_update_semantics(stage.GetPrimAtPath(f'/World/envs/env_{i}/{name}'),label+f'_{i}')
    add_update_semantics(stage.GetPrimAtPath('/World/floor'),'w8_floor')


class VisualDiagnostics:
    def __init__(self,out,states):
        self.out=Path(out)/'visual_diagnostics';self.out.mkdir(exist_ok=False)
        self.states=states;self.saved=set();self.previous_frame={};self.records=[]

    def record(self,scene,images,rgb,wrist,phases,frame):
        from pxr import Usd,UsdGeom
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        for i,state in enumerate(self.states):
            actual_phase=phases[i]
            phase='standing' if frame==0 else actual_phase
            if phase=='grasp':phase='insert'
            key=(state.seed,phase)
            if key in self.saved or phase not in ['standing','clear','pre','insert','close','lift','hold','rejected']:continue
            self.saved.add(key);folder=self.out/f'seed_{state.seed}_t{frame*.01:05.2f}_{phase}';folder.mkdir(exist_ok=False)
            record=dict(seed=state.seed,env_index=i,frame=frame,time_s=frame*.01,phase=phase,teacher_phase=actual_phase,diagnostic_only=True,cameras={})
            for name in ['head_cam','wrist_cam']:
                sensor=scene[name];data=sensor.data
                sensor_frame=int(sensor.frame[i]);previous=self.previous_frame.get((state.seed,name))
                require_fresh_frame(previous,sensor_frame)
                self.previous_frame[(state.seed,name)]=sensor_frame
                prim_path=f'/World/envs/env_{i}/'+name
                prim=stage.GetPrimAtPath(prim_path);camera=UsdGeom.Camera(prim)
                row=dict(prim_path=prim_path,latest_pose_update_enabled=bool(sensor.cfg.update_latest_camera_pose),frame=sensor_frame,sensor_timestamp_s=float(sensor._timestamp[i]),
                    live_world_position=data.pos_w[i].cpu().tolist(),live_quat_w_world=data.quat_w_world[i].cpu().tolist(),
                    intrinsic_matrix=data.intrinsic_matrices[i].cpu().tolist(),image_shape=list(data.image_shape),
                    offset=dict(pos=list(sensor.cfg.offset.pos),rot=list(sensor.cfg.offset.rot),convention=sensor.cfg.offset.convention),
                    usd_local_to_world=[list(v) for v in UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())],
                    clipping_range=list(camera.GetClippingRangeAttr().Get()),focal_length=camera.GetFocalLengthAttr().Get(),
                    horizontal_aperture=camera.GetHorizontalApertureAttr().Get(),vertical_aperture=camera.GetVerticalApertureAttr().Get(),
                    render_outputs={})
                for kind,value in data.output.items():
                    array=value[i].cpu().numpy()
                    if kind=='rgb':
                        Image.fromarray(array[...,:3]).save(folder/(name+'_raw_rgb.png'))
                        row['render_outputs'][kind]=dict(shape=list(array.shape),min=int(array.min()),max=int(array.max()),black_fraction=float((array[...,:3].max(-1)==0).mean()))
                    elif kind=='distance_to_image_plane':
                        np.save(folder/(name+'_raw_depth_metres.npy'),array)
                        valid=np.isfinite(array)&(array>0)
                        row['render_outputs'][kind]=dict(shape=list(array.shape),finite_fraction=float(np.isfinite(array).mean()),valid_fraction=float(valid.mean()),min_valid=float(array[valid].min()) if valid.any() else None,max_valid=float(array[valid].max()) if valid.any() else None)
                    elif kind=='semantic_segmentation':
                        np.save(folder/(name+'_semantic_ids.npy'),array)
                        info=data.info.get(kind,{}) if isinstance(data.info,dict) else data.info[i].get(kind,{}) if data.info else {}
                        mapping=info.get('idToLabels',{})
                        counts={str(int(v)):int(c) for v,c in zip(*np.unique(array,return_counts=True))}
                        row['render_outputs'][kind]=dict(pixel_counts=counts,id_to_labels=mapping)
                point=scene['target'].data.root_pos_w[i].cpu().numpy()
                optical=camera_optical_point(point,data.pos_w[i].cpu().numpy(),data.quat_w_world[i].cpu().numpy())
                row['diagnostic_target_optical_metres']=optical.tolist()
                row['camera_exposure_attributes']={attr.GetName():str(attr.Get()) for attr in prim.GetAttributes() if 'exposure' in attr.GetName().lower()}
                record['cameras'][name]=row
            Image.fromarray(rgb[i].cpu().numpy()).save(folder/'head_processed_rgb.png')
            for view,name in [(0,'head_da2'),(1,'wrist_encoded')]:
                Image.fromarray((images[i,view,0].cpu().numpy().clip(0,1)*255).astype(np.uint8)).save(folder/(name+'.png'))
                Image.fromarray((images[i,view,1].cpu().numpy().clip(0,1)*255).astype(np.uint8)).save(folder/(name+'_valid_mask.png'))
            np.save(folder/'wrist_processed_metres.npy',wrist[i].cpu().numpy())
            record['lighting']={str(p.GetPath()):{attr.GetName():str(attr.Get()) for attr in p.GetAttributes() if 'intensity' in attr.GetName().lower() or 'exposure' in attr.GetName().lower()} for p in stage.Traverse() if p.GetTypeName().endswith('Light')}
            (folder/'metadata.json').write_text(json.dumps(record,indent=2))
            self.records.append(dict(seed=state.seed,phase=phase,time_s=frame*.01,path=str(folder)))
        (self.out/'index.json').write_text(json.dumps(self.records,indent=2))
