"""Task-defined distribution and separately labelled lower-level design choices."""
from dataclasses import asdict, dataclass
import importlib
import math
from pathlib import Path
import sys
import types

import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
W8 = Path('/home/bdi/wang-sm/logdir/legmm/v4/w8')
OUTPUT = Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_1_random_reachable_20261114')
STOW = (-.00002, -1.22810, .03377, 1.57077, 1.19431, -1.57078)
LOWER = np.array((-3.14, -2.96, -.087, -3.01, -1.76, -3.01, 0.0))
UPPER = np.array((2.089, .17, 3.14, 3.01, 1.76, 3.01, .072))
STEP_LENGTH = np.array((.01,) * 6 + (.005,))
RESIDUAL_SCALE = np.array((.05,) * 6 + (.01,))
PHYSICS_DT, CONTROL_DT, STAND_DT, EXECUTE = .01, .04, .02, 4
COMMAND_DELAY = 10
SPLITS = {'train': range(0, 10000), 'val': range(10000, 10256), 'test': range(20000, 20256)}


def contract_modules():
    # Load the actual immutable contract and its relative dependencies without importing
    # embodied.__init__ (JAX and training services). No W9 module is rewritten.
    name = '_w8_readonly_legmm'
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(WORKTREE / 'airdreamer/dreamerv3/embodied/envs/legmm')]
        sys.modules[name] = pkg
    return tuple(importlib.import_module(f'{name}.{m}') for m in ('ag_contract', 'scenes'))


@dataclass(frozen=True)
class PlannerConfig:
    # Design choices authorised by W8-1, not measured performance or new success gates.
    pre_back: float = .08  # separate alignment from final straight side approach
    lift: float = .08  # command margin above the task's 5 cm success threshold
    squeeze: float = .008  # command closure past the physical object width
    ik_pos_tol: float = .001
    ik_rot_tol: float = .01
    path_joint_step: float = .02
    ik_iterations: int = 160
    close_s: float = .6
    hold_s: float = 2.4  # does not replace measured continuous 2 s hold
    grasp_back: float = .02  # leave palm clearance on the long box side


@dataclass(frozen=True)
class InitialState:
    seed: int
    attempt: int
    shape: str
    size: tuple  # box thickness,width,height; cylinder diameter,diameter,height
    mass: float
    friction: float
    table_size: tuple
    object_xy: tuple
    object_yaw: float
    base_xy: tuple
    base_yaw: float
    radius: float
    bearing: float
    velocity: tuple

    @property
    def support(self):
        return self.table_size[2]

    @property
    def center(self):
        return np.array((*self.object_xy, self.support + self.size[2] / 2))

    @property
    def inertia(self):
        x, y, z = self.size
        if self.shape == 'box':
            return self.mass / 12 * np.array((y*y+z*z, x*x+z*z, x*x+y*y))
        r = x / 2
        return self.mass * np.array(((3*r*r+z*z)/12, (3*r*r+z*z)/12, r*r/2))

    def json(self):
        return asdict(self)


def sample_candidate(seed, attempt=0, residual=False):
    """No rollout-dependent resampling. Each attempt has its own reproducible stream."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
    # Choose the family once per episode seed. Rejection must not convert a hard
    # box episode into an easier cylinder and silently change the 70/30 mixture.
    shape = 'box' if np.random.default_rng(seed).random() < .7 else 'cylinder'
    rng.random()  # keep attempt-zero streams identical to the original sampler
    size = (rng.uniform(.03,.05), rng.uniform(.10,.14), rng.uniform(.12,.15)) if shape == 'box' else (
        (d := rng.uniform(.04,.06)), d, rng.uniform(.10,.15))
    mass, friction = rng.uniform(.1,.3), rng.uniform(.6,1.0)
    table = (rng.uniform(.6,1.2), rng.uniform(.6,1.0), rng.uniform(.50,.65))
    # Uniform on the tabletop strip whose nearest-edge distance is 5–25 cm.
    while True:
        xy = rng.uniform(-np.array(table[:2])/2 + .05, np.array(table[:2])/2 - .05)
        edge = min(np.array(table[:2])/2 - abs(xy))
        if edge <= .25:
            break
    yaw = rng.uniform(0, math.pi)
    contract, _ = contract_modules()
    radius = rng.uniform(.45, contract.reach_limit(table[2]))
    bearing = rng.uniform(-math.pi/4, math.pi/4)
    # All world azimuths, not a patch next to one table edge.
    heading = rng.uniform(-math.pi, math.pi)
    base_yaw = heading - bearing
    shoulder_xy = xy - radius * np.array((math.cos(heading), math.sin(heading)))
    base_xy = shoulder_xy - .03 * np.array((math.cos(base_yaw), math.sin(base_yaw)))
    velocity_rng = np.random.default_rng(np.random.SeedSequence([seed, attempt, 2718]))
    speed = .05 * math.sqrt(velocity_rng.random())
    angle = velocity_rng.uniform(-math.pi, math.pi)
    velocity = (speed * math.cos(angle), speed * math.sin(angle), velocity_rng.uniform(-.1,.1))
    return InitialState(seed, attempt, shape, tuple(size), mass, friction, tuple(table), tuple(xy), yaw,
                        tuple(base_xy), base_yaw, radius, bearing, velocity if residual else (0.,0.,0.))


def gate_diagnostics(state, shoulder_xy=None, base_xy=None, base_yaw=None, velocity=None, object_center=None):
    contract, scenes = contract_modules()
    x,y = state.base_xy if base_xy is None else base_xy
    a = state.base_yaw if base_yaw is None else base_yaw
    shoulder = (x + .03*math.cos(a), y + .03*math.sin(a)) if shoulder_xy is None else shoulder_xy
    velocity = state.velocity if velocity is None else velocity
    from importlib import import_module
    shaping = import_module('_w8_readonly_legmm.ag_shaping')
    report = shaping.gate_report(shoulder, a, state.center if object_center is None else object_center, state.support, math.hypot(*velocity[:2]), contract.gate_cfg(state.support))
    sx,sy,h = state.table_size
    table = np.array([[-sx/2,-sy/2,0.,sx/2,sy/2,h]])
    clearance_ok = bool(scenes.footprint_free(table, x, y, a, contract.BODY_MARGIN - 1e-9))
    extent = np.array(state.size[:2])/2
    if state.shape == 'box':
        c,s = abs(math.cos(state.object_yaw)), abs(math.sin(state.object_yaw))
        extent = np.array([[c,s],[s,c]]) @ extent
    supported = bool(np.all(abs(np.array(state.object_xy)) + extent <= np.array(state.table_size[:2])/2))
    settled = bool(math.hypot(*velocity[:2]) < .05 and abs(velocity[2]) < .1)
    reason = ('envelope' if not report['ok'] else 'clearance' if not clearance_ok
              else 'unsettled' if not settled else 'object_overhang' if not supported else None)
    return dict(contract=report,limits=contract.gate_cfg(state.support),
                shoulder_xy=[float(v) for v in shoulder],base_xy=[float(x),float(y)],base_yaw=float(a),
                velocity=[float(v) for v in velocity],clearance_ok=clearance_ok,settled=settled,
                supported=supported,reason=reason)


def gate_reason(state, shoulder_xy=None, base_xy=None, base_yaw=None, velocity=None, object_center=None):
    return gate_diagnostics(state,shoulder_xy,base_xy,base_yaw,velocity,object_center)['reason']


def json_normalized(value):
    import json
    return json.loads(json.dumps(value))


def validate_reused_sampling(states, seed, num_envs, residual):
    if [s.seed for s in states] != list(range(seed,seed+num_envs)):
        raise ValueError('Reused initial seeds do not match requested cohort')
    # Existing static records must never be relabelled as residual sampling.
    for state in states:
        expected = sample_candidate(state.seed,state.attempt,residual)
        if json_normalized(state.json()) != json_normalized(expected.json()):
            raise ValueError('Reused sampling differs from deterministic requested distribution')
