"""CPU equivalence of actual original/candidate outcome-loop ASTs."""
import ast,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tabletop.outcome import EpisodeOutcome,lowest_point
from tabletop.outcome_readback_candidate import candidate_source,outcome_loop

ROOT=Path(__file__).parents[1]
CURRENT=(ROOT/'learning_episode.py').read_text()
BASELINE=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_1_random_reachable_20261114/dagger_true4_visual64_retry1_resume_02/source_snapshot/learning_episode.py')
ORIGINAL=BASELINE.read_text() if 'outcome_position_z=position[:,2].cpu().numpy()' in CURRENT else CURRENT
CANDIDATE=CURRENT if 'outcome_position_z=position[:,2].cpu().numpy()' in CURRENT else candidate_source(ORIGINAL)
CODES=[compile(ast.fix_missing_locations(ast.Module(body=[outcome_loop(s)],type_ignores=[])),'actual_outcome_loop','exec') for s in (ORIGINAL,CANDIDATE)]


def test_threshold_ulp_boundary_terminal_priority_and_all_rows_preserved():
    n=64;rng=np.random.default_rng(892);heights=torch.full((n,),.38);upz=torch.ones(n)
    hv=np.float32(.2);uv=np.float32(.5)
    for i,h in enumerate([np.nextafter(hv,-np.inf,dtype=np.float32),hv,np.nextafter(hv,np.inf,dtype=np.float32)]):heights[i]=float(h)
    for i,u in enumerate([np.nextafter(uv,-np.inf,dtype=np.float32),uv,np.nextafter(uv,np.inf,dtype=np.float32)],3):upz[i]=float(u)
    position=torch.tensor(rng.uniform(.62,.71,(n,3)),dtype=torch.float32);velocity=torch.tensor(rng.uniform(-.4,.2,(n,3)),dtype=torch.float32)
    states=[SimpleNamespace(shape='box' if i%2 else 'cylinder',size=(.03,.04,.08)) for i in range(n)]
    rotations=np.repeat(np.eye(3)[None],n,axis=0);bad=[set() for _ in range(n)];touch=[[True,True] for _ in range(n)]
    bad[7]={'link6:table'};bad[8]={'FL_calf:floor'};touch[10]=[False,False]
    original=[EpisodeOutcome(.6,steps=100,held_steps=199,lifted=True) for _ in range(n)]
    original[11].reason='success';original[11].success=True;bad[11]={'link6:table'}
    original[12].reason='collision:left:table';position[9,2]=.601;velocity[9,2]=-.2
    first,second=copy.deepcopy(original),copy.deepcopy(original)
    common=dict(np=np,torch=torch,n=n,position=position,target=SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=velocity)),heights=heights,upz=upz,rotations=rotations,states=states,bad=bad,touch=touch,lowest_point=lowest_point)
    a=dict(common,outcomes=first);b=dict(common,outcomes=second,outcome_position_z=position[:,2].numpy(),outcome_velocity_z=velocity[:,2].numpy(),outcome_height=heights.numpy(),outcome_upz=upz.numpy())
    for _ in range(3):exec(CODES[0],a);exec(CODES[1],b);assert first==second
    assert [first[i].reason=='base_fall' for i in range(6)]==[True,False,False,True,True,False]
    assert first[7].reason.startswith('collision:') and first[9].reason=='drop' and first[11].success


def test_candidate_does_not_change_contact_loop_or_global_base_safety():
    from tabletop.prepare_v8_foot_patch import contact_loop
    assert ast.dump(contact_loop(ORIGINAL),include_attributes=False)==ast.dump(contact_loop(CANDIDATE),include_attributes=False)
    assert "if bool(((heights<.2)|(upz<=.5)).any()):" in ORIGINAL and "if bool(((heights<.2)|(upz<=.5)).any()):" in CANDIDATE
