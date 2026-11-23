"""Read authored USD and live PhysX properties without changing the simulation."""
import math


def usd_value(value):
    # USD uses +/-inf as an unset COM sentinel; preserve that meaning as text
    # rather than emitting nonstandard JSON numeric literals.
    if isinstance(value,(bool,int,str)):
        return value
    if isinstance(value,float):
        return value if math.isfinite(value) else str(value)
    try:
        return [usd_value(v) for v in value]
    except TypeError:
        return str(value)


def physics_audit(simulation, robot, target, table, states):
    from pxr import Usd, UsdPhysics, PhysxSchema
    stage=simulation.stage
    records=[]
    for i,state in enumerate(states):
        base=f'/World/envs/env_{i}'
        root=stage.GetPrimAtPath(base+'/Robot/base_link/base_link')
        api=PhysxSchema.PhysxArticulationAPI(root)
        entry=dict(seed=state.seed,solver_position_iterations=api.GetSolverPositionIterationCountAttr().Get(),
                   solver_velocity_iterations=api.GetSolverVelocityIterationCountAttr().Get(),
                   self_collisions=api.GetEnabledSelfCollisionsAttr().Get(),prims=[])
        entry['rigid_body_solver']={}
        for name in ('Object','Table'):
            rigid_api=PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(base+'/'+name))
            entry['rigid_body_solver'][name]=dict(position=rigid_api.GetSolverPositionIterationCountAttr().Get(),
                velocity=rigid_api.GetSolverVelocityIterationCountAttr().Get())
        for name in ('Object','Table','Robot'):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(base+'/'+name)):
                attrs={}
                for attr in prim.GetAttributes():
                    key=attr.GetName()
                    if (key.startswith(('physics:','physxMaterial:','physxJoint:','physxRigidBody:','drive:'))
                        or key in ('size','radius','height','xformOp:scale')):
                        value=attr.Get()
                        if value is not None:
                            attrs[key]=usd_value(value)
                relations={rel.GetName():[str(t) for t in rel.GetTargets()] for rel in prim.GetRelationships()
                           if 'filteredPairs' in rel.GetName()}
                if attrs or relations:
                    entry['prims'].append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),attributes=attrs,relations=relations))
        records.append(entry)
    scene_prim=stage.GetPrimAtPath(simulation.cfg.physics_prim_path)
    scene_limits={attr.GetName():usd_value(attr.Get()) for attr in scene_prim.GetAttributes()
        if 'IterationCount' in attr.GetName() and attr.Get() is not None}
    return dict(scene_solver_iteration_limits=scene_limits,base_fixed=bool(robot.is_fixed_base),body_names=robot.body_names,
                robot_masses=robot.root_physx_view.get_masses().cpu().tolist(),
                object_masses=target.root_physx_view.get_masses().cpu().tolist(),
                object_inertias=target.root_physx_view.get_inertias().cpu().tolist(),
                object_materials=target.root_physx_view.get_material_properties().cpu().tolist(),
                table_materials=table.root_physx_view.get_material_properties().cpu().tolist(),
                environments=records)
