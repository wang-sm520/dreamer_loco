"""Unadopted v8 candidate: same foot geometry, one body-wide host readback.

The supplied quat_apply and tensor factory are the runtime's existing functions.
No classifier, threshold, contact order, environment mask or outcome is changed.
"""


def batched_foot_centers(body_positions,body_quaternions,body_id,tensor,quat_apply):
    n=body_positions.shape[0]
    offsets=tensor([[0,0,-.275]]).expand(n,3)
    feet=body_positions[:,body_id]+quat_apply(body_quaternions[:,body_id],offsets)
    return feet.cpu().numpy()
