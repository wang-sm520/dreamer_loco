"""Current-state geometric recovery labels for tabletop DAgger."""
from dataclasses import replace
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .closed_loop import local_pose_step
from .geometry import side_heading
from .spec import LOWER, STEP_LENGTH, UPPER, PlannerConfig, gate_reason


class StatelessDaggerTeacher:
    def __init__(self, planner, state, cfg=PlannerConfig()):
        self.planner, self.state, self.cfg = planner, state, cfg

    def _configure(self, measured):
        m = measured
        p = self.planner
        object_rotation = Rotation.from_quat(np.r_[m.object_pose[4:7], m.object_pose[3]]).as_matrix()
        yaw = math.atan2(object_rotation[1, 0], object_rotation[0, 0])
        state = replace(self.state, object_xy=tuple(m.object_pose[:2]), object_yaw=yaw)
        p.configure(state, m.base_pose, m.object_pose, m.joints)
        base = Rotation.from_quat(np.r_[m.base_pose[4:7], m.base_pose[3]]).as_matrix()
        base_yaw = math.atan2(base[1, 0], base[0, 0])
        shoulder = p.d.xpos[p.m.body('link2').id].copy()
        delta = m.object_pose[:2] - shoulder[:2]
        live = replace(state, base_yaw=base_yaw, bearing=math.atan2(delta[1], delta[0]) - base_yaw)
        return live, object_rotation, shoulder

    def label(self, measured, horizon=8, *, initial_gate_reason=None):
        """Pass the episode's original gate result for decisions after reset."""
        m = measured
        m.validate()
        if not np.any(np.max(abs(m.delay_queue - m.last_command), axis=1) < 1e-5):
            raise ValueError('Last command is absent from the live delay queue')
        p, cfg = self.planner, self.cfg
        state, object_rotation, shoulder = self._configure(m)
        tcp, tcp_rotation = p.fk(p.initial_arm)
        heading, width, twist = side_heading(state, cfg)
        opening = m.joints['endleft'] - m.joints['endright']
        if m.elapsed_s == 0 and initial_gate_reason is None:
            initial_gate_reason = gate_reason(
                state, shoulder[:2], m.base_pose[:2], state.base_yaw, m.velocity, m.object_pose[:3])

        phase, reason = 'rejected', None
        target, rotation = tcp.copy(), tcp_rotation
        axis = np.array([math.cos(heading), math.sin(heading), 0.])
        carrying = all(m.finger_contact)
        closing = False
        distance = depth_geometry = None
        if initial_gate_reason:
            reason = 'initial_gate:' + initial_gate_reason
        elif m.forbidden_contact or m.elapsed_s >= 15.:
            reason = 'terminal_safety_hold'
        elif not carrying and object_rotation[2, 2] < math.cos(math.radians(15.)):
            reason = 'tilted_object_hold'
        elif width >= UPPER[6]:
            reason = 'opening_hold'
        else:
            depth, depth_geometry = p.grasp_depth(axis)
            goal = m.object_pose[:3] - depth * axis
            distance = p.pregrasp_distance(axis, depth)
            pre = goal - distance * axis
            upright = np.array([[axis[0], -axis[1], 0], [axis[1], axis[0], 0], [0, 0, 1.]])
            u = float(np.dot(goal - tcp, axis))
            off_axis = float(np.linalg.norm(goal - tcp - u * axis))
            orient_error = float(np.linalg.norm(
                Rotation.from_matrix(upright @ tcp_rotation.T).as_rotvec()))
            pad_clearance = .5 * (UPPER[6] - width)
            grasp_opening = min(UPPER[6] - .0015, width + min(.006, pad_clearance))
            grasp_off_axis = float(np.clip(pad_clearance - .001, .002, .014))
            grasp_orient = float(np.clip(8. * pad_clearance, .02, .10))
            close_off_axis = min(.012, grasp_off_axis + .001)
            close_orient = min(.08, grasp_orient)
            min_thickness = float(min(state.size[:2]))
            enter_close = (opening >= grasp_opening and -.010 <= u <= .010
                           and off_axis <= close_off_axis and orient_error <= close_orient)
            keep_closing = (min_thickness - .005 < opening < grasp_opening
                            and -.015 <= u <= .018 and off_axis <= .016 and orient_error <= .12)
            if carrying:
                target[2] = max(tcp[2], state.support + state.size[2] / 2 + cfg.lift
                                - (m.object_pose[:3] - tcp)[2])
                phase = 'hold' if target[2] == tcp[2] else 'lift'
                closing = True
            elif enter_close or keep_closing:
                phase, target, rotation, closing = 'close', goal, upright, True
            elif (-.008 <= u <= distance + .015 and off_axis <= grasp_off_axis
                  and orient_error <= grasp_orient and opening >= grasp_opening):
                phase, target, rotation = 'grasp', goal, upright
            elif opening < grasp_opening or (u < distance - .005 and off_axis > .015):
                phase, target, rotation = 'pre', pre, upright
            elif tcp[2] < goal[2] - .012 and u > .015:
                phase = 'clear'
                target[2] = goal[2]
            else:
                phase, target, rotation = 'pre', pre, upright

        first_target = tcp
        previous = np.asarray(m.last_command).copy()
        labels = []
        if carrying and reason is None:
            relative = tcp_rotation.T @ (m.object_pose[:3] - tcp)
            relative_rotation = tcp_rotation.T @ object_rotation
        for index in range(horizon):
            if reason is None:
                pending_tcp, pending_rotation = p.fk(previous[:6])
                step_tcp, step_rotation = local_pose_step(pending_tcp, pending_rotation, target, rotation)
                if index == 0:
                    first_target = step_tcp
                solved = p.ik(step_tcp, step_rotation, previous[:6])
                if solved is None:
                    reason = 'ik_hold'
                else:
                    desired = np.r_[solved, 0. if closing else .072]
                    delta = desired - previous
                    delta /= max(1., float(np.max(abs(delta) / STEP_LENGTH)))
                    candidate = np.clip(previous + delta, LOWER, UPPER)
                    object_delta = object_quaternion = None
                    if carrying:
                        carried_tcp, carried_rotation = p.fk(candidate[:6])
                        object_delta = carried_tcp + carried_rotation @ relative - m.object_pose[:3]
                        quat = Rotation.from_matrix(carried_rotation @ relative_rotation).as_quat()
                        object_quaternion = np.r_[quat[3], quat[:3]]
                    collision = p.collision(candidate[:6], object_delta, object_quaternion)
                    if collision:
                        reason = 'collision_hold:' + collision
                    else:
                        previous = candidate
            labels.append(previous.copy())
        return np.asarray(labels), dict(
            phase=phase, reason=reason, initial_gate_reason=initial_gate_reason,
            live_object_yaw=state.object_yaw, pregrasp_distance=distance, grasp_depth_geometry=depth_geometry,
            target_tcp=target.tolist(), first_target_tcp=first_target.tolist(), width=float(width),
            twist=float(twist), opening=float(opening), pending_queue_last=m.delay_queue[-1].tolist())


class EpisodeLabels:
    """Bind the locked episode's live queue readback to the learning hook.

    The episode constructs its teachers inside run_episode and does not pass the
    measured queue or label metadata to decision. The scoped adapter taps those
    exact values, including the certified teacher's unmodified reason.
    """
    def __init__(self, *, drive=False):
        self.drive = drive
        self.frames = {}

    def install(self):
        from contextlib import contextmanager
        from . import closed_loop

        @contextmanager
        def scope():
            original = closed_loop.ClosedLoopTeacher
            self.frames.clear()

            class EpisodeTeacher(original):
                def __init__(teacher, planner, state, cfg=PlannerConfig()):
                    super().__init__(planner, state, cfg)
                    teacher.stateless = StatelessDaggerTeacher(planner, state, cfg)

                def label(teacher, measured, horizon=8):
                    if self.drive:
                        label, metadata = teacher.stateless.label(
                            measured, horizon, initial_gate_reason=teacher.initial_gate_reason)
                        teacher.initial_gate_reason = metadata['initial_gate_reason']
                        teacher.phase = metadata['phase']
                    else:
                        label, metadata = super().label(measured, horizon)
                    self.frames[teacher.state.seed] = (teacher, measured, metadata)
                    return label, metadata

            closed_loop.ClosedLoopTeacher = EpisodeTeacher
            try:
                yield
            finally:
                closed_loop.ClosedLoopTeacher = original
                self.frames.clear()

        return scope()

    def take(self, seed, elapsed_s):
        teacher, measured, metadata = self.frames.pop(seed)
        if measured.elapsed_s != elapsed_s:
            raise RuntimeError('DAgger label is not from the current decision')
        return teacher, measured, metadata


class StatelessTeacherEvaluation:
    """Exercise only the new teacher on the certified physical episode path."""
    exhausted = False
    render_enabled = False
    teacher_drives = True
    driver_name = 'stateless_teacher'

    def reset(self, states, scene, simulation, out):
        pass

    def decision(self, scene, robot, joints, arm_ids, finger_ids, last_command, labels,
                 outcomes, frame, phases=None):
        return labels

    def finish(self, outcomes):
        pass
