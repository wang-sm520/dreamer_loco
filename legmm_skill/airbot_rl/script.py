"""Privileged scripted top-down grasp, the teacher: reads the true can pose and closes the loop every decision.

Wrist straight (j4-j6 = 0, as at the demos' grasp frames): damped least-squares position IK on j1-j3 steers the pad
midpoint, the closing axis stays horizontal and the approach tilts 30-54 deg. Offline, the vertical descent is
reachable for all 97 recorded can starts, at least 0.69 rad inside the joint limits. Targets stay just inside the
deployment client's step length, so each one is a single 25 Hz waypoint.

The rule reads the current state only, no phase memory: DAgger labels the student's states with it, and a memory
the student cannot see gave the same view opposite labels (go down while descending, go up after a restart).
"""

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, skew_symmetric_matrix

from airbot_rl.reward import GRASP_BELOW_TOP, HELD_WIDTH, REACH_ABOVE
from airbot_rl.scene import CAN_HEIGHT, CAN_RADIUS, EXECUTE, STEP_LENGTH, UPPER

SQUEEZE = 0.0466  # m: the demos' median command while holding (their reading 0.0546); the can stops the pads at
# 0.064 here, and squeezing only to the reading (0.056) let 7% of the cans slide out on the way home
LIFT_TO = 0.10  # m of can bottom above the floor before heading home
CLEAR = 0.04  # m over the can top to rise to before realigning
CARROT = 0.03  # m: farthest IK target per decision; one big joint step traces an arc (1.5 cm off over 12 cm)


class ScriptedGrasp:
    def __init__(self, env):
        self.env = env
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", ik_method="dls", ik_params={"lambda_val": 0.05}),
            env.num_envs, env.device)
        self.link6 = env.arm.find_bodies("link6")[0][0]
        self.jacobian_row = self.link6 - 1 if env.arm.is_fixed_base else self.link6  # fixed bases drop the root
        self.travel = torch.tensor((0.0, 1.0, 0.0), device=env.device).expand(env.num_envs, 3)  # fingers, in link6
        # 0.999: the env's ceil(|step| / STEP_LENGTH) must come out as one waypoint despite rounding.
        self.step = 0.999 * torch.tensor(STEP_LENGTH, device=env.device)

    def __call__(self, count=EXECUTE):
        """-> the next `count` targets [N,count,7], native units: full speed towards the goal, then holding it."""
        env, gripped = self.env, self.env.gripped
        grasp, can, axis = env.grasp_point(), env.can_pos(), env.can_axis()
        width = env.native_state()[:, 6]
        top, at = CAN_HEIGHT / 2, CAN_HEIGHT / 2 - GRASP_BELOW_TOP  # along the can axis, from its centre
        along = ((grasp - can) * axis).sum(1)
        lateral = grasp - can - along[:, None] * axis
        off_axis = lateral.norm(dim=1)
        travel = quat_apply(env.arm.data.body_quat_w[:, self.link6], self.travel)
        across = (lateral * travel).sum(1)
        # The fingers clear the can if its chord along their travel fits between them (the teacher's own descents
        # stay within 0.1 mm across, but up to 9 mm off the axis the other way).
        chord = (CAN_RADIUS ** 2 - (lateral - across[:, None] * travel).square().sum(1)).clamp_min(0).sqrt()
        fits = (across.abs() + chord < width / 2) & (off_axis < CAN_RADIUS)
        # At the grasp height: close when the can fits between the pads, and keep closing once they are partway
        # (the measured width is in the student's state too). Shut past the can's width means a miss.
        closing = (~gripped & (width > HELD_WIDTH) & (off_axis < 0.012) & ((along - at).abs() < 0.006)
                   & (fits | (width < UPPER[6] - 0.002)))
        lifting = gripped & (env.can_lowest() < LIFT_TO)
        rise = ~gripped & ~closing & (along < top + 0.03) & ~fits  # beside the can: straight up, never across it
        descend = ~gripped & ~closing & ~rise & fits

        def on_axis(height):
            return can + height[:, None] * axis

        target = on_axis(along.clamp(top + CLEAR, top + REACH_ABOVE))  # otherwise: realign over the can
        target = torch.where(descend[:, None], on_axis((along - CARROT).clamp(min=at)), target)
        target = torch.where(rise[:, None], grasp + (top + CLEAR - along)[:, None] * axis, target)
        target = torch.where(closing[:, None], on_axis(torch.full_like(along, at)), target)
        target = torch.where(lifting[:, None], grasp + torch.tensor((0.0, 0.0, CARROT), device=env.device), target)
        error = target - grasp
        target = grasp + error * (CARROT / error.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1)

        joints = env.arm_ids[:3]
        jacobian = env.arm.root_physx_view.get_jacobians()[:, self.jacobian_row, :, joints]
        # link6's Jacobian moved to the grasp point: v = v_link6 + w x lever.
        lever = grasp + env.scene.env_origins - env.arm.data.body_pos_w[:, self.link6]
        jacobian = jacobian[:, :3] - torch.bmm(skew_symmetric_matrix(lever), jacobian[:, 3:])
        self.ik.set_command(target, ee_quat=env.arm.data.body_quat_w[:, self.link6])
        arm = self.ik.compute(grasp, None, jacobian, env.arm.data.joint_pos[:, joints])
        arm = torch.where((gripped & ~lifting)[:, None], 0.0, arm)  # carry it home in joint space
        grip = torch.where(closing | gripped, SQUEEZE, UPPER[6])
        step = (torch.cat((arm, torch.zeros_like(arm), grip[:, None]), 1) - env.command)[:, None]
        reach = torch.arange(1, count + 1, device=env.device)[None, :, None] * self.step
        arm = step[..., :6] / (step[..., :6].abs() / reach[..., :6]).amax(-1, keepdim=True).clamp_min(1)
        return env.command[:, None] + torch.cat((arm, torch.minimum(step[..., 6:].abs(), reach[..., 6:])
                                                 * step[..., 6:].sign()), -1)
