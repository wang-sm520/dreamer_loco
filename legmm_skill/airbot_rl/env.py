"""Can-grasp env. One step is one deployed decision: a base policy proposes the next EXECUTE targets (the frozen depth
BC, or the privileged scripted grasp), PPO may add a bounded residual, and every target is expanded into 25 Hz
waypoints by the deployment client's interpolation, then sent through each environment's command delay.
"""

import json
import math

import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from airbot_rl.assets import CAN_POSITIONS
from airbot_rl.policy import FrozenBC
from airbot_rl.reward import staged_reward
from airbot_rl.scene import (ARM_LINKS, CAN_HEIGHT, CAN_RADIUS, COMMAND_DELAY, CONTROL_HZ, DOG_LINKS, EXECUTE, FINGERS,
                             LOWER, PHYSICS_HZ, STEP_LENGTH, SUBSTEPS, UPPER, CanSceneCfg, d405_noise, head_image,
                             wrist_depth)
from airbot_rl.script import ScriptedGrasp

HOME_EEF = 0.01  # recorded episodes start with the gripper closed at 0.007-0.01 m
PAD_CENTER = (0.0, 0.0, 0.005)  # finger pad centre in each finger body; their midpoint is the grasp point
PRIVILEGED = 28  # state, arm velocity, can position and orientation, can - grasp, grasp - home, finger-can contacts

CONDITIONS = {
    "clean": {"noise_strength_range": (0.0, 0.0), "head_gain_range": (1.0, 1.0),
              "command_delay_range": (COMMAND_DELAY, COMMAND_DELAY), "can_jitter_range": (-0.01, 0.01)},
    "noise1": {"noise_strength_range": (1.0, 1.0)},
    "random": {"noise_strength_range": (0.0, 1.0), "head_gain_range": (0.8, 1.2),
               "command_delay_range": (8, 12), "can_jitter_range": (-0.02, 0.02)},  # user 10-14
}


@configclass
class CanEnvCfg(DirectRLEnvCfg):
    base = "bc"  # "bc": frozen depth BC on both cameras; "script": privileged scripted grasp, no cameras
    teacher = False  # with base "bc": also run the scripted grasp, whose targets (teacher_chunk) label distillation
    noise_strength_range = CONDITIONS["clean"]["noise_strength_range"]
    head_gain_range = CONDITIONS["clean"]["head_gain_range"]
    command_delay_range = CONDITIONS["clean"]["command_delay_range"]  # physics steps
    can_jitter_range = CONDITIONS["clean"]["can_jitter_range"]  # m, on each horizontal axis
    checkpoint = "airbot_rl/runs/bc_wrist_v1/step_006000.pt"

    max_targets = 1000  # user-confirmed 10-03
    max_joint_step = 0.1  # user-confirmed 10-03
    success_radius = 0.05  # user-confirmed 10-03
    success_can_bottom = 0.25  # user-confirmed 10-03
    residual_scale = (0.05,) * 6 + (0.01,)  # user-confirmed 10-03
    contact_force = 1.0  # N, user-confirmed 10-03
    limit_penalty = 100.0  # joints only; the gripper is clipped without penalty like deployment (user 10-14)
    contact_penalty = 1.0  # user-confirmed 10-03
    topple_penalty = 0.5  # user-confirmed 10-03
    topple_angle = math.radians(45.0)  # user-confirmed 10-03; counts only a can not in the hand (user 10-11)
    # user-approved 10-11 with the staged reward (was 10): must beat hovering next to success, 0.94 / (1 - 0.99)
    success_reward = 100.0

    decimation = SUBSTEPS  # replaced every step by SUBSTEPS x the number of waypoints to run
    episode_length_s = max_targets / CONTROL_HZ  # nominal; max_episode_length counts decisions instead
    # Cameras are rendered explicitly at decisions. A huge render_interval stalls Isaac's reset (1e5 did,
    # 1000 starts fine), so keep 1000: one stray in-loop render per 5 s of sim time.
    sim = SimulationCfg(dt=1 / PHYSICS_HZ, render_interval=1000,
                        render=RenderCfg(antialiasing_mode="FXAA"))  # no temporal AA across sparse frames
    scene = CanSceneCfg(num_envs=8)
    action_space = EXECUTE * 7
    observation_space = 512 + 7 + EXECUTE * 7
    state_space = PRIVILEGED + EXECUTE * 7


class CanEnv(DirectRLEnv):
    cfg: CanEnvCfg

    def __init__(self, cfg: CanEnvCfg, render_mode=None, **kwargs):
        if cfg.base == "script":
            cfg.scene.head_cam = cfg.scene.wrist_cam = None
            cfg.observation_space = cfg.state_space
        super().__init__(cfg, render_mode, **kwargs)
        self.arm, self.can = self.scene["arm"], self.scene["can"]
        self.contacts = [self.scene[f"contact_{link}"] for link in ARM_LINKS]
        self.arm_ids = self.arm.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0]
        self.finger_ids = self.arm.find_joints(["endleft", "endright"], preserve_order=True)[0]
        self.pad_ids = self.arm.find_bodies(list(FINGERS), preserve_order=True)[0]
        tensor = lambda values: torch.tensor(values, device=self.device)  # noqa: E731
        self.lower, self.upper, self.step_length = tensor(LOWER), tensor(UPPER), tensor(STEP_LENGTH)
        self.residual_scale = tensor(cfg.residual_scale)
        # Can starts: one FK grasp midpoint per retained recording, in the env frame.
        self.can_starts = tensor([[p["x"], p["y"]] for p in json.loads(CAN_POSITIONS.read_text())])
        n = self.num_envs
        self.command = torch.zeros(n, 7, device=self.device)  # last sent target, as deployment's previous_action
        self.base_chunk = torch.zeros(n, EXECUTE, 7, device=self.device)
        history = max(CONDITIONS["random"]["command_delay_range"][1], cfg.command_delay_range[1]) + 1
        self.queue = torch.zeros(history, n, 7, device=self.device)  # targets in flight, kept across decisions
        self.env_ids = torch.arange(n, device=self.device)
        self.noise_strength = torch.zeros(n, device=self.device)
        self.head_gain = torch.ones(n, device=self.device)
        self.delay_steps = torch.full((n,), COMMAND_DELAY, dtype=torch.long, device=self.device)
        self.can_offset = torch.zeros(n, 2, device=self.device)
        self.home = torch.zeros(n, 3, device=self.device)
        self.excess = torch.zeros(n, device=self.device)
        flag = lambda: torch.zeros(n, dtype=torch.bool, device=self.device)  # noqa: E731
        self.touched, self.touched_dog, self.touched_floor = flag(), flag(), flag()
        self.toppled, self.episode_toppled, self.success = flag(), flag(), flag()
        self.held, self.gripped, self.held_candidate, self.fallen = flag(), flag(), flag(), flag()
        self.finger_can = torch.zeros(n, len(FINGERS), dtype=torch.bool, device=self.device)
        self.tick = self.current_ticks = 0
        self.script = ScriptedGrasp(self) if cfg.base == "script" or cfg.teacher else None
        self.base = FrozenBC(cfg.checkpoint, str(self.device)) if cfg.base == "bc" else self.script

    @property
    def max_episode_length(self):
        return self.cfg.max_targets // EXECUTE

    # --- geometry ------------------------------------------------------------------------------------------
    def native_state(self):
        q = self.arm.data.joint_pos
        return torch.cat((q[:, self.arm_ids], q[:, self.finger_ids[:1]] - q[:, self.finger_ids[1:]]), dim=1)

    def native_to_sim(self, native):
        target = self.arm.data.joint_pos_target.clone()
        target[:, self.arm_ids] = native[:, :6]
        target[:, self.finger_ids] = native[:, 6:] * torch.tensor([0.5, -0.5], device=self.device)
        return target

    def grasp_point(self):
        pads = self.arm.data.body_pos_w[:, self.pad_ids] + quat_apply(
            self.arm.data.body_quat_w[:, self.pad_ids], torch.tensor(PAD_CENTER, device=self.device).expand(
                self.num_envs, 2, 3))
        return pads.mean(1) - self.scene.env_origins

    def can_pos(self):
        return self.can.data.root_pos_w - self.scene.env_origins

    def can_axis(self):
        return quat_apply(self.can.data.root_quat_w, torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(
            self.num_envs, 3))

    def can_lowest(self):
        cos = self.can_axis()[:, 2].abs()
        return (self.can.data.root_pos_w[:, 2] - CAN_HEIGHT / 2 * cos
                - CAN_RADIUS * (1 - cos.square()).clamp_min(0).sqrt())

    def camera_inputs(self):
        """(head uint8 RGB, wrist metres), both on the real cameras' pixel grids at 1/SCALE."""
        wrist = wrist_depth(self.scene["wrist_cam"].data.output["distance_to_image_plane"])
        wrist = d405_noise(wrist, self.noise_strength)
        return head_image(self.scene["head_cam"].data.output["rgb"], self.head_gain), wrist

    # --- DirectRLEnv hooks ----------------------------------------------------------------------------------
    def _pre_physics_step(self, actions):
        raw = self.base_chunk + self.residual_scale * torch.tanh(actions.view(-1, EXECUTE, 7))
        targets = torch.clamp(raw, self.lower, self.upper)
        self.excess = (raw - targets)[..., :6].abs().sum((1, 2))
        self.gripper_clip = (raw - targets)[..., 6].abs().sum(1)
        starts, ends, previous = [], [], self.command
        for target in targets.unbind(1):
            step = target - previous
            capped = torch.cat((step[:, :6].clamp(-self.cfg.max_joint_step, self.cfg.max_joint_step), step[:, 6:]), 1)
            self.excess += (step - capped).abs().sum(1)
            starts.append(previous)
            previous = previous + capped
            ends.append(previous)
        self.command = previous
        starts, ends = torch.stack(starts, 1), torch.stack(ends, 1)
        # Deployment: ceil(max |delta| / STEP_LENGTH) waypoints per target, linspace(previous, target)[1:].
        counts = ((ends - starts).abs() / self.step_length).ceil().amax(-1).clamp_min(1)
        done = counts.cumsum(1)
        self.waypoint_count = done[:, -1]  # this env's 25 Hz waypoints this decision; the batch runs the longest
        self.extras["duration"] = self.waypoint_count / EXECUTE  # in nominal decisions, for ppo.TimePPO's discount
        ticks = int(done[:, -1].max())
        tick = torch.arange(1, ticks + 1, device=self.device).view(1, -1, 1)
        segment = (tick > done[:, None]).sum(-1).clamp(max=EXECUTE - 1)  # finished envs hold their last target
        fraction = ((tick[..., 0] - (done - counts).gather(1, segment)) / counts.gather(1, segment)).clamp(max=1)
        index = segment[..., None].expand(-1, -1, 7)
        start = starts.gather(1, index)
        self.waypoints = start + (ends.gather(1, index) - start) * fraction[..., None]
        self.cfg.decimation = SUBSTEPS * ticks
        self.current_ticks = ticks
        self.substep = 0
        for flag in (self.touched, self.touched_dog, self.touched_floor, self.finger_can):
            flag[:] = False

    def _update_touched(self, length):
        """OR the filtered contacts of the last `length` physics steps into this decision's flags."""
        dogs = len(DOG_LINKS)
        touched = {link: sensor.data.force_matrix_w_history[:, :length, 0].norm(dim=-1).amax(1) > self.cfg.contact_force
                   for link, sensor in zip(ARM_LINKS, self.contacts)}
        for hit in touched.values():
            self.touched_dog |= hit[:, :dogs].any(1)
            self.touched_floor |= hit[:, dogs]
        self.finger_can |= torch.stack([touched[link][:, dogs + 1] for link in FINGERS], 1)
        self.touched |= self.touched_dog | self.touched_floor

    def _apply_action(self):
        slot = self.tick % len(self.queue)
        self.queue[slot] = self.waypoints[:, self.substep // SUBSTEPS]
        delayed = self.queue[(self.tick - self.delay_steps) % len(self.queue), self.env_ids]
        self.tick += 1
        self.arm.set_joint_position_target(self.native_to_sim(delayed))
        if self.substep and self.substep % SUBSTEPS == 0:
            self._update_touched(SUBSTEPS)
        self.substep += 1

    def _get_observations(self):
        state = self.native_state()
        can, grasp = self.can_pos(), self.grasp_point()
        privileged = torch.cat((state, self.arm.data.joint_vel[:, self.arm_ids], can, self.can.data.root_quat_w,
                                can - grasp, grasp - self.home, self.finger_can.float()), dim=1)
        if self.script is not None:
            self.teacher_chunk = self.script(self.base.model.horizon if self.cfg.base == "bc" else EXECUTE)
        if self.cfg.base == "bc":
            self.sim.render()
            chunk, features, norm_state = self.base(state, *self.camera_inputs())
            self.base_chunk = chunk[:, :EXECUTE]
            policy = torch.cat((features, norm_state, self.base_chunk.flatten(1)), 1)
        else:
            self.base_chunk = self.teacher_chunk
            policy = torch.cat((privileged, self.base_chunk.flatten(1)), 1)
        return {"policy": policy, "critic": torch.cat((privileged, self.base_chunk.flatten(1)), 1)}

    def _get_dones(self):
        self._update_touched(min(self.current_ticks, SUBSTEPS))
        home = (self.grasp_point() - self.home).norm(dim=1) < self.cfg.success_radius
        self.success = home & (self.can_lowest() > self.cfg.success_can_bottom)
        return self.success, self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self):
        stage, self.held, self.gripped, self.held_candidate = staged_reward(
            self.grasp_point(), self.can_pos(), self.can_axis(), self.native_state()[:, 6], self.finger_can,
            self.home, self.can_lowest(), CAN_RADIUS, CAN_HEIGHT, UPPER[6], self.held_candidate)
        # Preserve terminal decision flags across DirectRLEnv's automatic reset for evaluation.
        self.extras.update(held=self.held.clone(), gripped=self.gripped.clone())
        # Fallen: tilted past topple_angle and not in the hand. A can carried home tilts 53-63 deg, in the demos too.
        self.fallen = (self.can_axis()[:, 2] < math.cos(self.cfg.topple_angle)) & ~self.gripped
        toppled = self.fallen & ~self.toppled
        self.toppled |= toppled
        self.episode_toppled = self.toppled.clone()
        reward = (stage + self.cfg.success_reward * self.success - self.cfg.limit_penalty * self.excess
                  - self.cfg.contact_penalty * self.touched - self.cfg.topple_penalty * toppled)
        logs = {"stage": stage.mean(), "held": self.held.float().mean(), "success": self.success.float().mean(),
                "gripped": self.gripped.float().mean(),
                "clip_excess": self.excess.mean(), "gripper_clip": self.gripper_clip.mean(),
                "contact": self.touched.float().mean(),
                "toppled": self.toppled.float().mean(),
                "noise_strength": self.noise_strength.mean(), "head_gain": self.head_gain.mean(),
                "command_delay_steps": self.delay_steps.float().mean()}
        ended = self.reset_terminated | self.reset_time_outs
        if ended.any():
            logs["episode_success"] = self.success[ended].float().mean()
        self.extras["log"] = logs
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        dog = self.scene["dog"]
        dog.write_joint_state_to_sim(dog.data.default_joint_pos[env_ids], dog.data.default_joint_vel[env_ids],
                                     env_ids=env_ids)
        joints = self.arm.data.default_joint_pos[env_ids].clone()
        joints[:, self.finger_ids] = torch.tensor([HOME_EEF / 2, -HOME_EEF / 2], device=self.device)
        self.arm.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
        self.arm.set_joint_position_target(joints, env_ids=env_ids)
        pose = self.can.data.default_root_state[env_ids].clone()
        pose[:, :2] = self.can_starts[torch.randint(len(self.can_starts), (n,), device=self.device)]
        self.can_offset[env_ids] = sample_uniform(*self.cfg.can_jitter_range, (n, 2), self.device)
        pose[:, :2] += self.can_offset[env_ids]
        pose[:, :3] += self.scene.env_origins[env_ids]
        yaw = sample_uniform(-math.pi, math.pi, (n,), self.device)
        pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        self.can.write_root_state_to_sim(pose, env_ids)
        # Fixed ranges consume no random draws, preserving clean's original reset sequence.
        for values, bounds in ((self.noise_strength, self.cfg.noise_strength_range),
                               (self.head_gain, self.cfg.head_gain_range)):
            values[env_ids] = bounds[0] if bounds[0] == bounds[1] else sample_uniform(*bounds, (n,), self.device)
        low, high = self.cfg.command_delay_range
        self.delay_steps[env_ids] = low if low == high else torch.randint(low, high + 1, (n,), device=self.device)
        self.sim.forward()  # refresh link poses before reading them and before the next render
        self.command[env_ids] = self.native_state()[env_ids]
        self.queue[:, env_ids] = self.command[env_ids]
        self.home[env_ids] = self.grasp_point()[env_ids]
        self.toppled[env_ids] = False
        self.finger_can[env_ids] = False
        self.held[env_ids] = False
        self.gripped[env_ids] = False
        self.held_candidate[env_ids] = False
        self.fallen[env_ids] = False
