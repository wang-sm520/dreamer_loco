"""Exercise prepared diagnostic copied-state cases on CPU, with no App/GPU."""
from types import SimpleNamespace
import numpy as np
import torch
from tabletop.outcome import lowest_point
from tabletop.outcome_gpu_proof import OutcomeProof


def test_prepared_diagnostic_cases_do_not_mutate_supplied_scene_arrays(tmp_path):
    proof=OutcomeProof(tmp_path);n=64;position=torch.zeros(n,3);velocity=torch.ones(n,3);heights=torch.full((n,),.38);upz=torch.ones(n)
    scope=dict(n=n,position=position,target=SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=velocity)),heights=heights,upz=upz,states=[SimpleNamespace(support=.6,shape='box',size=(.04,.06,.08)) for _ in range(n)],rotations=np.repeat(np.eye(3)[None],n,axis=0),lowest_point=lowest_point,np=np)
    result=proof.synthetic_priority(scope);assert result['all_equal'] and result['device']=='cpu'
    assert torch.equal(position,torch.zeros(n,3)) and torch.equal(velocity,torch.ones(n,3)) and torch.equal(heights,torch.full((n,),.38)) and torch.equal(upz,torch.ones(n))
