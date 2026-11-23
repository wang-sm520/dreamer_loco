"""Pure command expansion and reset-only delay state used by the new skill."""
import numpy as np
from .spec import COMMAND_DELAY, LOWER, UPPER, STEP_LENGTH, RESIDUAL_SCALE, EXECUTE


def teacher_commands(plan, initial, diagnostics=None):
    """Time the checked piecewise path continuously at the native speed limits.

    Rounding every geometry subdivision up to a whole control tick slows the
    arm according to collision-sampling density. Carry fractional segment time
    across waypoints instead; each interpolated segment retains its speed bound.
    """
    from .spec import PlannerConfig, CONTROL_DT
    cfg=PlannerConfig()
    values=[np.asarray(initial,dtype=float).copy()];times=[0.];labels=[]
    phase="initial"
    def append(value,dwell=None):
        value=np.asarray(value,dtype=float)
        duration=float(np.max(abs(value-values[-1])/STEP_LENGTH)) if dwell is None else dwell/CONTROL_DT
        if duration>1e-12:
            times.append(times[-1]+duration);values.append(value.copy());labels.append(phase)
    closed=False
    for joints,phase in zip(plan.joints,plan.phases):
        if phase=='lift' and not closed:
            phase='close'
            # Projected whole-object width is an opening feasibility bound, not
            # the local chord at finite pads. Close to the existing lower stop;
            # physical contact and the unchanged 15 N actuator cap stop fingers.
            close=values[-1].copy();close[6]=LOWER[6]
            append(close);append(close,cfg.close_s);closed=True
            phase='lift'
        append(np.r_[joints,values[-1][6]])
    phase='hold'
    append(values[-1],cfg.hold_s)
    ticks=np.minimum(np.arange(1,int(np.ceil(times[-1]))+1),times[-1])
    if diagnostics is not None:
        diagnostics.extend(dict(phase=labels[min(np.searchsorted(times[1:],t,side='left'),len(labels)-1)],command_time_s=float(t*CONTROL_DT)) for t in ticks)
    return np.column_stack([np.interp(ticks,times,np.array(values)[:,j]) for j in range(7)])


def expand_chunk(previous, base_chunk, residual=None):
    base = np.asarray(base_chunk)
    if base.shape != (EXECUTE, 7):
        raise ValueError('Expected four native seven-dimensional targets')
    raw = base if residual is None else base + RESIDUAL_SCALE * np.tanh(np.asarray(residual).reshape(EXECUTE,7))
    if not np.isfinite(raw).all():
        raise FloatingPointError('Nonfinite command')
    clipped = np.clip(raw, LOWER, UPPER)
    previous = np.asarray(previous).copy()
    points = []
    for target in clipped:
        delta = target - previous
        delta[:6] = np.clip(delta[:6], -.1, .1)
        count = max(1, int(np.ceil(np.max(abs(delta)/STEP_LENGTH))))
        points.extend(previous + delta*k/count for k in range(1,count+1))
        previous = previous + delta
    return np.array(points)


class DelayedCommands:
    def __init__(self, initial, delay=COMMAND_DELAY):
        self.delay = delay
        self.queue = np.repeat(np.asarray(initial)[None], delay+1, axis=0)
        self.tick = 0

    def step(self, target):
        self.queue[self.tick % len(self.queue)] = target
        result = self.queue[(self.tick-self.delay) % len(self.queue)].copy()
        self.tick += 1
        return result

    def state_dict(self):
        return dict(delay=self.delay, tick=self.tick, queue=self.queue.copy())

    def load_state_dict(self, state):
        if state['delay'] != self.delay or state['queue'].shape != self.queue.shape:
            raise ValueError('Incompatible control-state checkpoint')
        self.tick, self.queue = state['tick'], state['queue'].copy()
