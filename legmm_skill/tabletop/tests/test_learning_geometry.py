import json
from dataclasses import replace
import numpy as np
import pytest
from tabletop.geometry import GeometryPlanner
from tabletop.learning_geometry import LearningGeometryPlanner
from tabletop.spec import OUTPUT,InitialState


def test_large_teacher_cache_reuses_compiled_models_with_full_live_reset():
    result=json.loads((OUTPUT/'teacher_pregrasp_geometry_residual_retry1_01/result.json').read_text())
    state=InitialState(**result['states'][0]);base=np.array(result['initial_base_pose'][0])
    pose=np.array(result['initial_object_pose'][0]);joints=dict(zip(result['joint_names'],result['initial_joint_positions'][0]))
    model=OUTPUT/'models/a2_tabletop_mesh_shoulder_v2.xml'
    cached=LearningGeometryPlanner(model,max_models=40)
    cached.configure(state,base,pose,joints);first=cached.m
    for length in np.linspace(.1001,.1399,33):
        different=replace(state,size=(state.size[0],float(length),state.size[2]))
        cached.configure(different,base,pose,joints)
    shifted=pose.copy();shifted[:2]+=[.003,-.002]
    changed=dict(joints);changed['joint1']+=.03
    cached.configure(state,base,shifted,changed)
    assert cached.m is first and len(cached.full_cache)==34
    fresh=GeometryPlanner(model);fresh.configure(state,base,shifted,changed)
    assert cached.fk(cached.initial_arm)[0]==pytest.approx(fresh.fk(fresh.initial_arm)[0])
    assert cached.fk(cached.initial_arm)[1]==pytest.approx(fresh.fk(fresh.initial_arm)[1])
    assert cached.collision(cached.initial_arm)==fresh.collision(fresh.initial_arm)
    assert cached.grasp_depth(np.array([1.,0.,0.]))==pytest.approx(fresh.grasp_depth(np.array([1.,0.,0.])))
    from tabletop.lifecycle_cleanup import clear_planner_cache
    counts=clear_planner_cache(cached)
    assert counts['full_cache']==34 and not cached.full_cache and cached.m is None
    cached.configure(state,base,shifted,changed)
    assert cached.m is not first
    assert cached.fk(cached.initial_arm)[0]==pytest.approx(fresh.fk(fresh.initial_arm)[0])
    assert cached.fk(cached.initial_arm)[1]==pytest.approx(fresh.fk(fresh.initial_arm)[1])
    assert cached.collision(cached.initial_arm)==fresh.collision(fresh.initial_arm)
    assert cached.grasp_depth(np.array([1.,0.,0.]))==pytest.approx(fresh.grasp_depth(np.array([1.,0.,0.])))
