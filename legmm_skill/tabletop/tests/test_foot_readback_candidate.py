"""CPU replay of the actual original/candidate contact-loop AST, not Kit tests."""
import ast,copy,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tabletop.collision import excluded_pair
from tabletop.outcome import EpisodeOutcome
from tabletop.foot_readback_candidate import batched_foot_centers
from tabletop.segment_timing import SegmentTimings
from tabletop.prepare_v8_foot_patch import candidate_source,contact_loop

ROOT=Path(__file__).parents[1]
CURRENT=(ROOT/'learning_episode.py').read_text()
BASELINE=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_1_random_reachable_20261114/gpu_foot_same_state_v9_retry1_01/source_snapshot/learning_episode.py')
ORIGINAL=BASELINE.read_text() if 'use_batched_foot_readback=' in CURRENT else CURRENT
CANDIDATE=CURRENT if 'use_batched_foot_readback=' in CURRENT else candidate_source(ORIGINAL)
MATH=Path('/home/bdi/IsaacLab/source/isaaclab/isaaclab/utils/math.py')
node=next(n for n in ast.parse(MATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='quat_apply')
node.decorator_list=[];ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(MATH),'exec'),ns)
QUAT_APPLY=ns['quat_apply']
constants=[n for n in ast.parse((ROOT/'isaac_scene.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('DOG_BODIES','ARM_BODIES') for t in n.targets)]
NAMES={};exec(compile(ast.Module(body=constants,type_ignores=[]),'actual_scene_body_lists','exec'),NAMES)
DOG,ARM=NAMES['DOG_BODIES'],NAMES['ARM_BODIES']


def tensor(x):return torch.as_tensor(x,dtype=torch.float32)


def fixture(n,boundary):
    rng=np.random.default_rng(91+n);names=DOG+ARM+['arm_base'];permutation=rng.permutation(len(names));ids=dict(zip(names,permutation.tolist()))
    origins=rng.uniform(-320,320,(n,1,3)).astype(np.float32);origins[:16]=0
    positions=tensor(origins+rng.uniform(-.3,.3,(n,len(names),3)).astype(np.float32))
    quat=tensor(rng.normal(size=(n,len(names),4)));quat=quat/quat.norm(dim=-1,keepdim=True)
    references={}
    for body in DOG:
        if not body.endswith('_calf'):continue
        bid=ids[body];references[body]=torch.stack([positions[i,bid]+QUAT_APPLY(quat[i,bid][None],tensor([[0,0,-.275]]))[0] for i in range(n)]).numpy()
    # Make the threshold sample's foot exactly world zero, so one-ULP points
    # test the original0.04 comparison without origin-translation rounding.
    bid=ids['FL_calf'];positions[2,bid]=-QUAT_APPLY(quat[2,bid][None],tensor([[0,0,-.275]]))[0];references['FL_calf'][2]=0
    scene={};filters={}
    for body in DOG+ARM:
        partners=['table','object','floor']
        if body in ARM:partners+=DOG+[b for b in ['arm_base']+ARM if b!=body and not excluded_pair(body,b,DOG)]
        filters[body]=partners
        scene['contact_'+body]=SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(n,1,len(partners),3),contact_pos_w=torch.zeros(n,1,len(partners),3)))
    def put(i,body,partner,force,point=(0,0,0)):
        j=filters[body].index(partner);sensor=scene['contact_'+body].data;sensor.force_matrix_w[i,0,j,2]=force;sensor.contact_pos_w[i,0,j]=tensor(point)
    for i in range(n):
        case=i%16;foot=references['FL_calf'][i]
        if case==0:put(i,'FL_calf','floor',10,foot+[0,0,-.03])
        if case==1:put(i,'FL_calf','floor',11,foot+[.15,0,0])
        if case==2:put(i,'FL_calf','floor',12,foot+[boundary,0,0])
        if case==3:put(i,'FL_calf','floor',13,foot+[.039,0,0])
        if case==4:put(i,'FL_calf','floor',14,foot+[.041,0,0])
        if case==5:put(i,'FL_calf','table',3,foot)
        if case==6:put(i,'FR_calf','object',4,references['FR_calf'][i])
        if case==7:put(i,'link6','object',5)
        if case==8:put(i,'left','object',6);put(i,'right','object',7)
        if case==9:
            put(i,'FL_calf','floor',15,foot+[0,0,-.03]);put(i,'link6','table',8);put(i,'left','object',9)
        if case==10:put(i,'FL_calf','floor',1.,foot+[.15,0,0])
        if case==11:put(i,'FL_calf','floor',.999,foot+[.15,0,0])
        if case==12:put(i,'link2','link5',9)
        if case==13:put(i,'base_link','floor',16)
        if case==14:
            put(i,'FL_calf','floor',17,foot+[0,0,-.03]);put(i,'FR_calf','floor',18,references['FR_calf'][i]+[0,0,-.03])
        if case==15:put(i,'link6','table',19)
    robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=positions,body_quat_w=quat),find_bodies=lambda name:([ids[name]],[name]))
    return scene,robot,ids,references


def classify(source,use_batch,scene,robot,ids,event_counts=None,peaks=None):
    n=robot.data.body_pos_w.shape[0]
    scope=dict(timings=SegmentTimings(False),np=np,maths=SimpleNamespace(quat_apply=QUAT_APPLY),tensor=tensor,robot=robot,scene=scene,
        DOG_BODIES=DOG,ARM_BODIES=ARM,excluded_pair=excluded_pair,args=SimpleNamespace(audit_w2_model=False),n=n,
        bad=[set() for _ in range(n)],touch=[[False,False] for _ in range(n)],feet_peak=0.,finger_peak=0.,
        event_counts=copy.deepcopy(event_counts) if event_counts is not None else [{} for _ in range(n)],
        contact_peaks=copy.deepcopy(peaks) if peaks is not None else [{} for _ in range(n)],
        use_batched_foot_readback=use_batch,calf_body_ids={b:ids[b] for b in DOG if b.endswith('_calf')},batched_foot_centers=batched_foot_centers)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[contact_loop(source)],type_ignores=[])),'actual_contact_loop','exec'),scope)
    return {k:scope[k] for k in ('bad','touch','feet_peak','finger_peak','event_counts','contact_peaks')}


@pytest.mark.parametrize('n',[16,64])
@pytest.mark.parametrize('boundary',[np.nextafter(np.float32(.04),np.float32(-np.inf)),np.float32(.04),np.nextafter(np.float32(.04),np.float32(np.inf))])
def test_actual_contact_loops_match_origins_orientations_and_boundary(n,boundary):
    scene,robot,ids,ref=fixture(n,boundary)
    for body in ref:
        batch=batched_foot_centers(robot.data.body_pos_w,robot.data.body_quat_w,ids[body],tensor,QUAT_APPLY)
        np.testing.assert_array_equal(batch,ref[body])
    original=classify(ORIGINAL,False,scene,robot,ids)
    assert original==classify(CANDIDATE,False,scene,robot,ids)==classify(CANDIDATE,True,scene,robot,ids)
    assert not original['bad'][0] and 'FL_calf:floor' in original['bad'][1]
    assert bool(original['bad'][2])==bool(boundary>np.float32(.04))
    assert not original['bad'][3] and 'FL_calf:floor' in original['bad'][4]
    assert 'FL_calf:table' in original['bad'][5] and 'FR_calf:object' in original['bad'][6]
    assert original['touch'][8]==[True,True] and not original['bad'][8]
    assert not original['bad'][10] and not original['bad'][11]
    assert 'link2:link5' in original['bad'][12]


def test_multiple_steps_terminal_priority_raw_events_and_fresh_body_poses():
    scene,robot,ids,ref=fixture(16,np.float32(.04));before=classify(ORIGINAL,False,scene,robot,ids)
    old=[EpisodeOutcome(0.) for _ in range(16)];new=copy.deepcopy(old)
    for states in (old,new):
        for i in (8,9):states[i].held_steps=199;states[i].lifted=True
        states[10].lifted=True;states[15].reason='success';states[15].success=True
    oldcounts=newcounts=None;oldpeaks=newpeaks=None
    for step in range(3):
        a=classify(ORIGINAL,False,scene,robot,ids,oldcounts,oldpeaks);b=classify(CANDIDATE,True,scene,robot,ids,newcounts,newpeaks);assert a==b
        for i in range(16):
            bottom=.005 if i==10 else .06
            for verdict,contacts in ((old,a),(new,b)):
                verdict[i].update(bottom,-.1 if i==10 else 0.,contacts['touch'][i],contacts['bad'][i],i==14)
        assert old==new
        oldcounts,newcounts=a['event_counts'],b['event_counts'];oldpeaks,newpeaks=a['contact_peaks'],b['contact_peaks']
        robot.data.body_pos_w[:,ids['FL_calf'],0]+=.025
    assert old[8].success and old[9].reason.startswith('collision:') and old[10].reason=='drop' and old[14].reason=='base_fall'
    assert old[15].success and oldcounts[15]['link6:table']==3
    assert oldcounts[0].get('FL_calf:floor',0)>before['event_counts'][0].get('FL_calf:floor',0)
