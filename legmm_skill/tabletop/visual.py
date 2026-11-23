"""Two real camera streams, preserving source resolution and depth transforms.

Import attachment helpers only after AppLauncher. Camera poses add no physical
mass, following the task contract. All transforms are frozen in the manifest.
"""
import torch
import hashlib
from pathlib import Path
from airbot_depth.depth import DepthAnythingTransform,DEFAULT_DEPTH_CONFIG
from airbot_rl.policy import head_depth
from airbot_rl.depth import encode_wrist_depth
from .segment_timing import timer_for

# A2 base-local head mount is a task design choice, not calibrated hardware.
# Wrist mount/intrinsics and both render sizes match the existing source.
HEAD_BASE_POSITION=(.05,.18,.32)
HEAD_BASE_QUATERNION=(0.981060262190407, -0.015134435901338622, 0.0858316511774313, -0.17298739392508947) # fixed side mount: yaw -20 deg, 10 deg down


def attach_cameras(cfg,diagnostic=False):
    from isaaclab.sensors import TiledCameraCfg,TiledCamera
    import carb
    class NoPhysicsInitTiledCamera(TiledCamera):
        def _initialize_impl(self):
            settings=carb.settings.get_settings();previous=settings.get("/app/player/playSimulations")
            settings.set_bool("/app/player/playSimulations",False)
            try:super()._initialize_impl()
            finally:settings.set_bool("/app/player/playSimulations",bool(previous))
    from airbot_rl.scene import HEAD_CANVAS,HEAD_F,WRIST_CANVAS,WRIST_F,WRIST_CAM_POS,WRIST_CAM_ROT,_camera
    cfg.head_cam=TiledCameraCfg(prim_path='{ENV_REGEX_NS}/head_cam',
        data_types=['rgb'],width=HEAD_CANVAS[0],height=HEAD_CANVAS[1],spawn=_camera(HEAD_CANVAS,HEAD_F,.02),
        offset=TiledCameraCfg.OffsetCfg(pos=HEAD_BASE_POSITION,rot=HEAD_BASE_QUATERNION,convention='world'))
    cfg.wrist_cam=TiledCameraCfg(prim_path='{ENV_REGEX_NS}/wrist_cam',
        data_types=['distance_to_image_plane'],width=WRIST_CANVAS[0],height=WRIST_CANVAS[1],
        spawn=_camera(WRIST_CANVAS,WRIST_F,.005),
        offset=TiledCameraCfg.OffsetCfg(pos=WRIST_CAM_POS,rot=WRIST_CAM_ROT,convention='opengl'))
    cfg.head_cam.class_type=NoPhysicsInitTiledCamera
    cfg.wrist_cam.class_type=NoPhysicsInitTiledCamera
    cfg.head_cam.update_latest_camera_pose=True
    cfg.wrist_cam.update_latest_camera_pose=True
    if diagnostic:
        cfg.head_cam.data_types=['rgb','distance_to_image_plane','semantic_segmentation']
        cfg.wrist_cam.data_types=['distance_to_image_plane','semantic_segmentation']
        cfg.head_cam.colorize_semantic_segmentation=False
        cfg.wrist_cam.colorize_semantic_segmentation=False
    return cfg


def sync_camera_mounts(scene,render_callback):
    """Follow actual PhysX body frames, not stale authored USD link transforms.

    These world-root camera prims are rigidly mounted by a fixed local pose.
    Only robot body pose and immutable extrinsics enter this composition; no
    object pose or tracking target is read.
    """
    from isaaclab.utils.math import quat_apply,quat_mul,convert_camera_frame_orientation_convention
    from airbot_rl.scene import WRIST_CAM_POS,WRIST_CAM_ROT
    robot=scene['robot'];n=len(robot.data.root_pos_w);device=robot.device;records=[]
    for name,body,position,rotation,convention in [
        ('head_cam','base_link',HEAD_BASE_POSITION,HEAD_BASE_QUATERNION,'world'),
        ('wrist_cam','link6',WRIST_CAM_POS,WRIST_CAM_ROT,'opengl')]:
        index=robot.find_bodies(body)[0][0]
        parent_pos=robot.data.body_link_pos_w[:,index]
        parent_quat=robot.data.body_link_quat_w[:,index]
        offset=torch.tensor(position,device=device,dtype=torch.float32).expand(n,3)
        relative=torch.tensor(rotation,device=device,dtype=torch.float32)[None].expand(n,4)
        relative=convert_camera_frame_orientation_convention(relative,origin=convention,target='world')
        requested_pos=parent_pos+quat_apply(parent_quat,offset)
        requested_quat=quat_mul(parent_quat,relative)
        # Installed TiledCamera creates XformPrimView without the renderer USD
        # mirror used by Camera. Enable it on this W8-owned instance only.
        # Otherwise SimulationContext.render() overwrites the Fabric write.
        if not hasattr(scene[name]._view,'_sync_usd_on_fabric_write'):raise RuntimeError('Camera view lacks renderer sync support')
        scene[name]._view._sync_usd_on_fabric_write=True
        gl_quat=convert_camera_frame_orientation_convention(requested_quat,origin='world',target='opengl')
        scene[name]._view._set_world_poses_usd(requested_pos,gl_quat)
        # Flush authored transforms through the renderer without stepping physics.
        render_callback()
        scene[name].set_world_poses(requested_pos,requested_quat,convention='world')
        scene[name]._view._fabric_usd_sync_done=True
        read_pos,read_quat=scene[name]._view.get_world_poses()
        usd_pos,_=scene[name]._view._get_world_poses_usd()
        records.append(dict(usd_world=usd_pos.cpu().tolist(),name=name,body=body,parent_world=parent_pos.cpu().tolist(),parent_quat=parent_quat.cpu().tolist(),requested_world=requested_pos.cpu().tolist(),requested_quat=requested_quat.cpu().tolist(),immediate_view_world=read_pos.cpu().tolist()))
    return records


class DualDepth:
    def __init__(self,device):
        self.device=device
        self.depth=DepthAnythingTransform(DEFAULT_DEPTH_CONFIG,device=device,local_files_only=True)
        self.provenance=self.depth.provenance()

    @torch.no_grad()
    def observe(self,scene):
        from airbot_rl.scene import head_image,wrist_depth
        rgb=scene['head_cam'].data.output['rgb'][...,:3]
        rgb=head_image(rgb,torch.ones(len(rgb),device=rgb.device))
        wrist=wrist_depth(scene['wrist_cam'].data.output['distance_to_image_plane'])
        timings=timer_for(self)
        images=torch.stack((timings.call('da2_head',head_depth,self.depth,rgb,batch=8).to(self.device),timings.call('wrist_preprocess',encode_wrist_depth,wrist)),1)
        if not torch.isfinite(images).all():raise FloatingPointError('Nonfinite dual-depth observation')
        return images,rgb,wrist

    def manifest(self):
        from airbot_rl.scene import HEAD_CANVAS,WRIST_CANVAS,WRIST_CAM_POS,WRIST_CAM_ROT
        import airbot_rl.policy,airbot_rl.depth,airbot_rl.scene
        dependencies={Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (airbot_rl.policy,airbot_rl.depth,airbot_rl.scene)}
        return dict(depth_config=self.depth.config,depth_provenance=self.provenance,source_sha256=dependencies,da2_microbatch=8,
            head=dict(base_position=HEAD_BASE_POSITION,quaternion=HEAD_BASE_QUATERNION,canvas=HEAD_CANVAS,
                representation='RGB -> pinned DA2 relative inverse depth',logical_parent='actual PhysX base_link',pose_update='fixed extrinsic composed every rendered decision'),
            wrist=dict(link6_position=WRIST_CAM_POS,quaternion=WRIST_CAM_ROT,canvas=WRIST_CANVAS,
                representation='metric distance_to_image_plane',logical_parent='actual PhysX link6',pose_update='fixed extrinsic composed every rendered decision'),actor_image_shape=[2,2,128,128])
