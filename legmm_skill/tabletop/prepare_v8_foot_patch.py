"""Prepare, never apply, original and timing-compatible foot-readback patches."""
import ast,difflib,hashlib,json
from pathlib import Path
from .prepare_v7_timing_patch import revised_sources


def candidate_source(source,timed=False):
    s=source
    needle='    from .geometry import GeometryPlanner,sample_reachable'
    assert needle in s;s=s.replace(needle,"    from .foot_readback_candidate import batched_foot_centers\n    use_batched_foot_readback=bool(getattr(args,'audit_batched_foot_readback',False))\n"+needle,1)
    needle="        finger_ids=robot.find_joints(['endleft','endright'],preserve_order=True)[0]"
    assert needle in s;s=s.replace(needle,needle+"\n        calf_body_ids={body:robot.find_bodies(body)[0][0] for body in DOG_BODIES if body.endswith('_calf')} if use_batched_foot_readback else {}",1)
    needle='                    points=sensor.data.contact_pos_w[:,0].cpu().numpy()'
    readback="batched_foot_centers(robot.data.body_pos_w,robot.data.body_quat_w,calf_body_ids[body],tensor,maths.quat_apply)"
    if timed:readback="timings.call('foot_batch_gpu_cpu_readback',batched_foot_centers,robot.data.body_pos_w,robot.data.body_quat_w,calf_body_ids[body],tensor,maths.quat_apply)"
    assert needle in s;s=s.replace(needle,needle+"\n                    batch_foot_cpu="+readback+" if use_batched_foot_readback and body.endswith('_calf') else None",1)
    old="""                                bid=robot.find_bodies(body)[0][0]
                                foot=robot.data.body_pos_w[i,bid]+maths.quat_apply(robot.data.body_quat_w[i,bid][None],tensor([[0,0,-.275]]))[0]
"""
    condition="timings.call('foot_cpu_readback_distance',lambda:np.linalg.norm(points[i,j]-foot.cpu().numpy()))" if timed else 'np.linalg.norm(points[i,j]-foot.cpu().numpy())'
    old+='                                if '+condition+' <= .04:'
    new="""                                if use_batched_foot_readback:
                                    legal_foot=np.linalg.norm(points[i,j]-batch_foot_cpu[i]) <= .04
                                else:
                                    bid=robot.find_bodies(body)[0][0]
                                    foot=robot.data.body_pos_w[i,bid]+maths.quat_apply(robot.data.body_quat_w[i,bid][None],tensor([[0,0,-.275]]))[0]
                                    legal_foot="""+condition+""" <= .04
                                if legal_foot:"""
    assert old in s;s=s.replace(old,new,1);ast.parse(s);return s


def contact_loop(source):
    tree=ast.parse(source)
    return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='body' and isinstance(n.iter,ast.BinOp) and isinstance(n.iter.left,ast.Name) and n.iter.left.id=='DOG_BODIES')


def main():
    root=Path(__file__).parent;record=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_true4_contact_retry1_20261115')
    old=(root/'learning_episode.py').read_text();timed=revised_sources(root)['learning_episode.py'][1]
    records={}
    for name,before,is_timed in [('v8_batched_foot_standalone',old,False),('v8_batched_foot_after_v7_timing',timed,True)]:
        after=candidate_source(before,is_timed);path=record/(name+'_unapplied.patch')
        patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/legmm_skill/tabletop/learning_episode.py',tofile='b/legmm_skill/tabletop/learning_episode.py'));path.write_text(patch)
        records[name]=dict(path=str(path),baseline_sha256=hashlib.sha256(before.encode()).hexdigest(),candidate_sha256=hashlib.sha256(after.encode()).hexdigest(),applied=False,enabled_by_default=False)
    (record/'v8_foot_candidate_source.json').write_text(json.dumps(dict(candidates=records,helper=str(root/'foot_readback_candidate.py'),helper_sha256=hashlib.sha256((root/'foot_readback_candidate.py').read_bytes()).hexdigest(),GPU_equivalence_measured=False,speedup_measured=False),indent=2)+'\n')


if __name__=='__main__':main()
