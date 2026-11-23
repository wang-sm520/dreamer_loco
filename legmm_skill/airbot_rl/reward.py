"""Staged grasp reward (table user-approved for a first trial, 10-11). Pure torch, so it can be checked on the CPU.

Every decision scores how far the grasp has got, and each stage pays more than the one before it. Positions are taken
in the can's own frame (axis u), so once the can is held the earlier stages stay at their maximum however it tilts.
"""

import torch

REACH_ABOVE = 0.10  # m over the can top; the demos pass 4-7 cm from the axis with the pads ~13 cm above the top
GRASP_BELOW_TOP = 0.025  # m; demo grasp frames put the pad midpoint at 8.9 cm (median), 2.5 cm under the top
HELD_WIDTH = 0.055  # m; pads squeezing the 6.5 cm can stay wider, a closed gripper pressing on its top does not
LIFTED = 0.02  # m of can bottom above the floor before carrying home pays
HELD_AXIS_DISTANCE = 0.015  # m; user-confirmed 10-13


def staged_reward(grasp, can, axis, width, finger_can, home, can_bottom, can_radius, can_height, max_width,
                  previous_candidate):
    """Positions [N,3] m, width/bottom [N] m, contacts [N,2], previous [N] bool -> reward/held/gripped/candidate [N]."""
    rel = grasp - can
    along = (rel * axis).sum(1)
    off_axis = (rel - along[:, None] * axis).norm(dim=1)
    gripped = finger_can.all(1) & (width > HELD_WIDTH) & (along < can_height / 2)
    candidate = (can_bottom > LIFTED) & (off_axis < HELD_AXIS_DISTANCE) & (along.abs() < can_height / 2)
    held = candidate & previous_candidate
    clear = off_axis < (width - 2 * can_radius) / 2
    over = can + (can_height / 2 + REACH_ABOVE) * axis
    reach = 1 - torch.tanh((grasp - over).norm(dim=1) / 0.2)
    align = 1 - torch.tanh(off_axis / 0.01)
    opened = (width / max_width).clamp(max=1)
    descend = (1 - torch.tanh((along - (can_height / 2 - GRASP_BELOW_TOP)).abs() / 0.1)) * (clear | held)
    carry = (1 - torch.tanh((grasp - home).norm(dim=1) / 0.3)) * (held & (can_bottom > LIFTED))
    reward = 0.1 * reach + 0.1 * align + 0.05 * opened + 0.2 * descend + 0.2 * held + 0.35 * carry
    return reward, held, gripped, candidate
