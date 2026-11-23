"""Enforce the task's 4/0 solver for the scene, not only its articulation.

IsaacLab PhysxCfg documents that the highest actor counts are clamped to the
scene limits. Dynamic rigid bodies otherwise default to 16/1 in this installation.
"""


def apply_task_solver_limits(simulation_config):
    for name,value in dict(min_position_iteration_count=4,max_position_iteration_count=4,
                           min_velocity_iteration_count=0,max_velocity_iteration_count=0).items():
        setattr(simulation_config.physx,name,value)
    return simulation_config


def verify_task_physics_audit(audit):
    expected={'physxScene:minPositionIterationCount':4,'physxScene:maxPositionIterationCount':4,
              'physxScene:minVelocityIterationCount':0,'physxScene:maxVelocityIterationCount':0}
    if audit.get('base_fixed') is not False or any(audit.get('scene_solver_iteration_limits',{}).get(k)!=v for k,v in expected.items()):
        raise ValueError('Missing or non-task scene-wide 4/0 solver evidence')
    envs=audit.get('environments',[])
    if not envs:raise ValueError('Missing per-environment physics evidence')
    for env in envs:
        if (env.get('solver_position_iterations'),env.get('solver_velocity_iterations'),env.get('self_collisions'))!=(4,0,True):
            raise ValueError('Articulation must be free, self-colliding, solver 4/0')
        for body in ('Object','Table'):
            if env.get('rigid_body_solver',{}).get(body)!=dict(position=4,velocity=0):
                raise ValueError('Rigid body solver differs from task 4/0')
    return True
