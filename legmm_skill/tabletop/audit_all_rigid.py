"""Read every authored rigid actor, including the shared floor, before collection."""


def audit_all_rigid(stage):
    from pxr import UsdPhysics,PhysxSchema
    records=[]
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
        api=PhysxSchema.PhysxRigidBodyAPI(prim)
        position=api.GetSolverPositionIterationCountAttr().Get()
        velocity=api.GetSolverVelocityIterationCountAttr().Get()
        rigid=UsdPhysics.RigidBodyAPI(prim)
        row=dict(path=str(prim.GetPath()),position=position,velocity=velocity,
            kinematic=rigid.GetKinematicEnabledAttr().Get(),enabled=rigid.GetRigidBodyEnabledAttr().Get())
        records.append(row)
        if (position,velocity)!=(4,0):raise ValueError('Non-task rigid actor solver: '+str(row))
    if not records or not any(x['path']=='/World/floor' for x in records):
        raise ValueError('Missing floor or rigid actor evidence')
    return records
