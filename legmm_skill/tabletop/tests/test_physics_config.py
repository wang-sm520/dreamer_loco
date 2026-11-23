from types import SimpleNamespace
from tabletop.physics_config import apply_task_solver_limits
from tabletop.physics_config import verify_task_physics_audit
import pytest


def test_scene_limits_prevent_another_dynamic_actor_raising_task_solver():
    cfg=SimpleNamespace(physx=SimpleNamespace(min_position_iteration_count=1,max_position_iteration_count=255,
        min_velocity_iteration_count=0,max_velocity_iteration_count=255))
    apply_task_solver_limits(cfg)
    actor_position,actor_velocity=16,1
    effective_position=max(cfg.physx.min_position_iteration_count,min(actor_position,cfg.physx.max_position_iteration_count))
    effective_velocity=max(cfg.physx.min_velocity_iteration_count,min(actor_velocity,cfg.physx.max_velocity_iteration_count))
    assert (effective_position,effective_velocity)==(4,0)


def test_articulation_only_4_0_evidence_cannot_qualify_validation():
    audit=dict(base_fixed=False,environments=[dict(solver_position_iterations=4,
        solver_velocity_iterations=0,self_collisions=True)])
    with pytest.raises(ValueError):verify_task_physics_audit(audit)
    audit['scene_solver_iteration_limits']={'physxScene:minPositionIterationCount':4,'physxScene:maxPositionIterationCount':4,
        'physxScene:minVelocityIterationCount':0,'physxScene:maxVelocityIterationCount':0}
    audit['environments'][0]['rigid_body_solver']={'Object':dict(position=16,velocity=1),'Table':dict(position=16,velocity=1)}
    with pytest.raises(ValueError):verify_task_physics_audit(audit)
    audit['environments'][0]['rigid_body_solver']={'Object':dict(position=4,velocity=0),'Table':dict(position=4,velocity=0)}
    assert verify_task_physics_audit(audit)
