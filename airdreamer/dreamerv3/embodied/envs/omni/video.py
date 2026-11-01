"""Shared video recorder with nav-aligned defaults in nav arenas."""

import os
import numpy as np
import torch


class _Mp4Writer:
    """MP4 写出器（两端实现一致，直接合并到此处）。"""

    def __init__(self, path, fps, width, height):
        import av
        self._av = av
        self._container = av.open(str(path), mode='w')
        self._stream = self._container.add_stream('h264', rate=int(fps))
        self._stream.width = int(width)
        self._stream.height = int(height)
        self._stream.pix_fmt = 'yuv420p'

    def write(self, frame_rgb):
        frame = self._av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self):
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


class _VideoRecorder:
    """训练录像器。nav 模式默认跟随无人机，multi 模式默认固定相机。"""

    def __init__(self, cfg, logdir, arena, log_fn=None):
        self._log = log_fn or (lambda msg: print(msg))
        self._enabled = False
        if not cfg or not bool(cfg.get('enabled', False)):
            self._log('[VideoRecorder] Skipped: video.enabled is False or no config')
            return

        self._arena = arena
        self._logdir = str(logdir) if logdir else None
        self._camera = None
        self._camera_needs_refresh = False
        self._segment_active = False
        self._segment_start_train_step = None

        try:
            from omni.isaac.core.utils.extensions import enable_extension
            enable_extension('omni.replicator.isaac')
        except Exception:
            pass

        self._enabled = True
        nav_mode = str(getattr(arena, '_interaction_mode', 'multi')) == 'nav'
        self._focus_env = int(cfg.get('focus_env', 0) or 0)
        default_drone_following = bool(nav_mode)
        default_follow_all_envs = bool(nav_mode)
        default_pos = (5.0, 0.0, 3.0) if nav_mode else (0.0, 0.0, 5.0)
        default_lookat = (0.0, 0.0, 0.0) if nav_mode else (0.0, 0.0, 2.0)
        default_prim_path = '/World/nav_video_camera' if nav_mode else '/World/train_video_camera'
        default_focal_length = 35.0 if nav_mode else 24.0

        self._drone_following = bool(cfg.get('drone_following', default_drone_following))
        default_attach_to_drone = self._drone_following if nav_mode else False
        self._attach_to_drone = bool(cfg.get('attach_to_drone', default_attach_to_drone))
        self._follow_all_envs = bool(cfg.get('follow_all_envs', default_follow_all_envs))
        if self._drone_following and bool(getattr(arena, 'headless', True)) and (not self._attach_to_drone):
            self._attach_to_drone = True
            self._log('[VideoRecorder] headless=True: forcing attach_to_drone=True to avoid PhysX pose-update errors.')

        self._pos = tuple(float(x) for x in cfg.get('pos', default_pos))
        self._lookat = tuple(float(x) for x in cfg.get('lookat', default_lookat))
        self._resolution = tuple(int(x) for x in cfg.get('resolution', (1280, 720)))
        self._camera_prim_path = str(cfg.get('prim_path', default_prim_path))
        self._camera_name_fallback = 'nav_video_camera' if nav_mode else 'train_video_camera'
        self._camera_focal_length = float(cfg.get('focal_length', default_focal_length))

        physics_dt = float(arena.physics_dt)
        self._physics_hz = (1.0 / physics_dt) if physics_dt > 0 else 60.0
        self._desired_fps = float(cfg.get('fps', 30) or 30)

        desired, phz = self._desired_fps, self._physics_hz
        if desired <= 0:
            self._stride = 1
            self._actual_fps = float(phz)
        else:
            self._stride = max(1, int(round(float(phz) / float(desired))))
            self._actual_fps = float(phz) / float(self._stride)

        self._segment_every = int(cfg.get('segment_every', 300))
        segment_seconds = cfg.get('segment_length_seconds', None)
        try:
            segment_seconds = float(segment_seconds)
        except Exception:
            segment_seconds = 0.0
        self._segment_length_physics = (
            int(round(segment_seconds * float(self._physics_hz)))
            if segment_seconds > 0 else 0
        )
        if self._segment_every <= 0 or self._segment_length_physics <= 0:
            self._enabled = False
            self._log(
                f'[VideoRecorder] Skipped: segment_every={self._segment_every}, '
                f'segment_length_physics={self._segment_length_physics} (need both > 0)'
            )
            return
        self._next_trigger = int(self._segment_every)

        outdir = str(cfg.get('outdir', 'video') or 'video')
        if self._logdir:
            logdir_expanded = os.path.expanduser(str(self._logdir))
            self._outdir = os.path.join(logdir_expanded, outdir)
        else:
            self._outdir = os.path.join(os.getcwd(), outdir)
        self._outdir = os.path.abspath(self._outdir)
        os.makedirs(self._outdir, exist_ok=True)
        self._log(f'[VideoRecorder] Videos will be saved to: {self._outdir}')

    def _resolve_pose(self):
        pos, lookat = self._pos, self._lookat
        base = None
        if self._drone_following and getattr(self._arena, '_drone_pool', None):
            focus = int(np.clip(int(self._focus_env), 0, len(self._arena._drone_pool) - 1))
            drone = self._arena._drone_pool[focus]
            if getattr(drone, 'initialized', False) and hasattr(drone, '_view'):
                try:
                    state = drone.get_state(env_frame=False)
                    pos_flat = state[..., :3].reshape(-1, 3)
                    base = pos_flat[0].detach().cpu().numpy() if hasattr(pos_flat[0], 'cpu') else pos_flat[0]
                except Exception:
                    base = None
            if base is None:
                drone_paths = getattr(self._arena, '_drone_prim_paths', None)
                if drone_paths:
                    try:
                        import omni.isaac.core.utils.prims as prim_utils
                        from pxr import Usd, UsdGeom

                        prim = prim_utils.get_prim_at_path(f"{drone_paths[focus]}/base_link")
                        if prim is not None and prim.IsValid():
                            mat = np.array(
                                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                                    Usd.TimeCode.Default()),
                                dtype=np.float64,
                            )
                            base = mat[:3, 3]
                    except Exception:
                        base = None
        if base is None:
            env_positions = getattr(self._arena, '_env_positions', None)
            if env_positions is not None:
                focus = int(np.clip(int(self._focus_env), 0, int(env_positions.shape[0]) - 1))
                base = env_positions[int(focus)]
                if isinstance(base, torch.Tensor):
                    base = base.detach().cpu().numpy()
        if base is not None:
            base = np.asarray(base, np.float32).reshape(3).tolist()
            pos = (float(base[0] + pos[0]), float(base[1] + pos[1]), float(base[2] + pos[2]))
            lookat = (float(base[0] + lookat[0]), float(base[1] + lookat[1]), float(base[2] + lookat[2]))
        return pos, lookat

    def _ensure_camera(self):
        if (not self._enabled) or (self._camera is not None):
            return
        import omni.isaac.core.utils.prims as prim_utils
        from omni_drones.sensors.camera import Camera
        from omni_drones.sensors.config import PinholeCameraCfg

        res = self._resolution
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0.0,
            resolution=(int(res[0]), int(res[1])),
            data_types=['rgb'],
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=self._camera_focal_length,
                focus_distance=400.0,
                horizontal_aperture=28.995,  # HFOV 45° with focal_length=35
                f_stop=0.0,
                clipping_range=(0.1, 1.0e5),
            ),
        )
        camera = Camera(camera_cfg)
        spawned = False
        drone_paths = getattr(self._arena, '_drone_prim_paths', None)
        if self._drone_following and self._attach_to_drone and drone_paths:
            focus = int(np.clip(int(self._focus_env), 0, len(drone_paths) - 1))
            env_ids = list(range(len(drone_paths))) if self._follow_all_envs else [focus]
            cam_name = self._camera_prim_path.strip('/').split('/')[-1] or self._camera_name_fallback
            cam_paths = [f"{drone_paths[i]}/base_link/{cam_name}" for i in env_ids]
            spawn_paths = [p for p in cam_paths if not prim_utils.is_prim_path_valid(p)]
            if spawn_paths:
                camera.spawn(spawn_paths, translations=[self._pos] * len(spawn_paths), targets=None)
                spawned = True
            if len(cam_paths) > 1:
                camera.initialize(prim_paths_expr=f"/World/drones/drone_.*/base_link/{cam_name}")
            else:
                camera.initialize(prim_paths_expr=cam_paths[0])
            self._camera = camera
            self._camera_needs_refresh = True
            return
        pos, lookat = self._resolve_pose()
        if (not spawned) and (not prim_utils.is_prim_path_valid(self._camera_prim_path)):
            camera.spawn([self._camera_prim_path], translations=[pos], targets=[lookat])
        camera.initialize(prim_paths_expr=self._camera_prim_path)
        self._camera = camera
        self._camera_needs_refresh = True

    def _update_camera_pose(self, pos, lookat):
        """更新相机 prim 变换，用于跟机录像。"""
        try:
            from pxr import Gf
            import omni.isaac.core.utils.prims as prim_utils
            from omni_drones.sensors.camera import orientation_from_view

            prim = prim_utils.get_prim_at_path(self._camera_prim_path)
            if not prim or not prim.IsValid():
                return
            orient = orientation_from_view(pos, lookat)
            quat = Gf.Quatd(orient[0], orient[1], orient[2], orient[3])
            prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*pos))
            prim.GetAttribute('xformOp:orient').Set(quat)
        except Exception:
            pass

    def _extract_rgb_uint8(self, images):
        import torch
        if images is None:
            return None
        rgb = None
        if isinstance(images, dict):
            rgb = images.get('rgb', None)
        else:
            try:
                rgb = images['rgb']
            except Exception:
                try:
                    item = images[0]
                    rgb = item.get('rgb', None) if isinstance(item, dict) else item['rgb']
                except Exception:
                    rgb = None
        if rgb is None:
            return None
        if isinstance(rgb, torch.Tensor):
            if rgb.ndim == 4:
                if rgb.shape[1] >= 3:
                    rgb = rgb[:, :3].permute(0, 2, 3, 1).contiguous()
                elif rgb.shape[-1] >= 3:
                    rgb = rgb[..., :3].contiguous()
                else:
                    return None
            elif rgb.ndim == 3:
                if rgb.shape[0] >= 3:
                    rgb = rgb[:3].permute(1, 2, 0).contiguous()
                elif rgb.shape[-1] >= 3:
                    rgb = rgb[..., :3].contiguous()
                else:
                    return None
            elif rgb.ndim == 2:
                rgb = rgb.unsqueeze(-1).repeat(1, 1, 3).contiguous()
            else:
                return None
            if rgb.dtype != torch.uint8:
                f = rgb.float()
                mx = float(f.max().detach().cpu().item()) if f.numel() else 0.0
                if mx <= 1.0:
                    f = f * 255.0
                rgb = f.clamp(0, 255).to(torch.uint8)
            frame = rgb.detach().cpu().numpy()
        else:
            frame = np.asarray(rgb)
            if frame.ndim == 4:
                if frame.shape[1] >= 3:
                    frame = np.transpose(frame[:, :3, ...], (0, 2, 3, 1))
                elif frame.shape[-1] >= 3:
                    frame = frame[..., :3]
                else:
                    return None
            elif frame.ndim == 3:
                if frame.shape[0] >= 3 and frame.shape[-1] < 3:
                    frame = np.transpose(frame[:3, ...], (1, 2, 0))
                elif frame.shape[-1] >= 3:
                    frame = frame[..., :3]
                else:
                    return None
            elif frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=-1)
            else:
                return None
            if frame.dtype != np.uint8:
                mx = float(frame.max()) if frame.size else 0.0
                if mx <= 1.0:
                    frame = frame * 255.0
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _start_segment(self, train_step_after):
        self._segment_active = True
        self._segment_start_train_step = int(train_step_after)
        self._segment_end_train_step = None
        self._segment_physics_done = 0
        self._writer = None  # 流式写盘，首帧时惰性初始化
        self._segment_frame_count = 0

    def _finish_segment(self):
        writer = getattr(self, '_writer', None)
        frame_count = getattr(self, '_segment_frame_count', 0)
        if writer is None or frame_count == 0:
            if not getattr(self, '_warned_empty_capture', False):
                self._warned_empty_capture = True
                print(
                    '[VideoRecorder] Segment finished with 0 frames (capture may fail in headless mode). '
                    'Try headless: False or xvfb-run for headless rendering.'
                )
            self._segment_active = False
            self._segment_start_train_step = None
            self._segment_end_train_step = None
            self._segment_physics_done = 0
            self._writer = None
            return
        try:
            writer.close()
        except Exception:
            pass
        start = int(self._segment_start_train_step or 0)
        end = int(self._segment_end_train_step or start)
        final_path = os.path.join(self._outdir, f'train_{start}_{end}.mp4')
        tmp_path = getattr(self, '_segment_tmp_path', None)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.rename(tmp_path, final_path)
            except OSError:
                pass
        self._segment_active = False
        self._segment_start_train_step = None
        self._segment_end_train_step = None
        self._segment_physics_done = 0
        self._writer = None
        self._segment_frame_count = 0
        self._segment_tmp_path = None

    def close(self):
        if not self._enabled:
            return
        try:
            if self._segment_active:
                if self._segment_end_train_step is None:
                    self._segment_end_train_step = int(
                        getattr(self._arena, '_train_steps_total', 0))
                self._finish_segment()
        except Exception:
            pass

    def will_capture_frame(self, train_step_after, physics_step_index):
        if not self._enabled:
            return False
        train_step_after = int(train_step_after)
        physics_step_index = int(physics_step_index)
        active = self._segment_active or (train_step_after >= int(self._next_trigger or 0))
        if not active:
            return False
        return (physics_step_index % int(self._stride)) == 0

    def on_physics_step(self, train_step_after, physics_step_index, capture_frame):
        if not self._enabled:
            return
        train_step_after = int(train_step_after)
        if not self._segment_active:
            if train_step_after >= int(self._next_trigger or 0):
                self._start_segment(train_step_after)
                while int(self._next_trigger or 0) <= train_step_after:
                    self._next_trigger = int(self._next_trigger or 0) + int(self._segment_every)
        if not self._segment_active:
            return
        if bool(capture_frame):
            self._ensure_camera()
            if self._camera is not None:
                try:
                    if (self._drone_following and (not self._attach_to_drone)
                            and (not bool(getattr(self._arena, 'headless', True)))):
                        pos, lookat = self._resolve_pose()
                        self._update_camera_pose(pos, lookat)
                    if self._camera_needs_refresh:
                        self._arena.render_current()
                        self._camera_needs_refresh = False
                    images = self._camera.get_images()
                    frame = self._extract_rgb_uint8(images)
                    if frame is not None:
                        if frame.ndim == 4:
                            frame = frame[int(np.clip(self._focus_env, 0, frame.shape[0] - 1))]
                        if self._writer is None:
                            start = int(self._segment_start_train_step or 0)
                            tmp_path = os.path.join(self._outdir, f'train_{start}_tmp.mp4')
                            self._segment_tmp_path = tmp_path
                            self._writer = _Mp4Writer(
                                tmp_path, max(1, int(round(self._actual_fps))),
                                int(frame.shape[1]), int(frame.shape[0]))
                        self._writer.write(frame)
                        self._segment_frame_count += 1
                except Exception as e:
                    print(f'[VideoRecorder] Error capturing frame: {e}')
        self._segment_physics_done += int(self._stride)
        self._segment_end_train_step = train_step_after
        if self._segment_physics_done >= int(self._segment_length_physics):
            self._finish_segment()
