"""Batch-independent verdicts, updated AFTER EVERY physics step, before auto-reset."""
from dataclasses import dataclass
import math

import numpy as np


def lowest_point(center_z, rotation, shape, size):
    """Support function of the actual oriented primitive, not its centre."""
    zrow = np.asarray(rotation)[2]
    if shape == 'box':
        return float(center_z - np.dot(abs(zrow), np.asarray(size)/2))
    radius, height = size[0]/2, size[2]
    return float(center_z - height/2*abs(zrow[2]) - radius*math.hypot(zrow[0], zrow[1]))


@dataclass
class EpisodeOutcome:
    support: float
    dt: float = .01
    steps: int = 0
    held_steps: int = 0
    lifted: bool = False
    reason: str | None = None
    success: bool = False
    peak_lift: float = 0.

    def update(self, bottom, vertical_velocity, finger_contacts, forbidden=(), fallen=False):
        if self.reason is not None:
            return
        if not all(math.isfinite(x) for x in (bottom, vertical_velocity)):
            raise FloatingPointError('Nonfinite physical state')
        self.steps += 1
        clearance = bottom - self.support
        self.peak_lift = max(self.peak_lift, clearance)
        # At a shared step failure dominates success. A stable zero-force sample
        # alone is never a drop; loss of grip AND downward motion is required.
        if forbidden:
            self.reason = 'collision:' + ','.join(sorted(set(forbidden)))
        elif fallen:
            self.reason = 'base_fall'
        elif self.lifted and (clearance <= .01 + 1e-10 or (not any(finger_contacts) and vertical_velocity < -1e-6)):
            self.reason = 'drop'
        elif self.steps * self.dt > 15. + 1e-10:
            self.reason = 'timeout'
        else:
            self.lifted |= clearance >= .05 - 1e-10
            self.held_steps = self.held_steps + 1 if clearance >= .05 - 1e-10 else 0
            if self.held_steps * self.dt >= 2. - 1e-10:
                self.success, self.reason = True, 'success'
            elif self.steps * self.dt >= 15. - 1e-10:
                self.reason = 'timeout'


def classify_contacts(pairs, lifted=False, threshold=1.):
    """Pairs of semantic body groups and measured force norms, direction independent.

    Sensors must resolve each pair; net body forces cannot implement this API.
    Positive tests here do NOT replace the required Isaac physical sensor probes.
    """
    fingers, forbidden = set(), set()
    for a,b,force in pairs:
        if not math.isfinite(force) or force < 0:
            raise ValueError('Expected finite nonnegative force norm')
        if force <= threshold:
            continue
        pair = frozenset((a,b))
        if 'object' in pair and (a in ('left','right') or b in ('left','right')):
            fingers.add(b if a == 'object' else a)
        elif pair == {'foot','floor'} or (pair == {'object','table'} and not lifted):
            continue
        elif pair == {'object','table'}:
            # Return to support is classified by height in EpisodeOutcome.
            continue
        else:
            forbidden.add(':'.join(sorted((a,b))))
    return ('left' in fingers, 'right' in fingers), tuple(sorted(forbidden))
