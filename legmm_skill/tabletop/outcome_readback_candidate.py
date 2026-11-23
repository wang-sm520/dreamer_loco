"""Unapplied candidate: batch existing GPU scalar reads without changing verdicts."""
import ast,difflib
from pathlib import Path


def candidate_source(source):
    needle='                for i,o in enumerate(outcomes):'
    assert source.count(needle)==1
    before='''                outcome_position_z=position[:,2].cpu().numpy()
                outcome_velocity_z=target.data.root_lin_vel_w[:,2].cpu().numpy()
                outcome_height=heights.cpu().numpy()
                outcome_upz=upz.cpu().numpy()
'''
    source=source.replace(needle,before+needle,1)
    source=source.replace('bottom=lowest_point(float(position[i,2]),rotations[i],states[i].shape,states[i].size)', 'bottom=lowest_point(float(outcome_position_z[i]),rotations[i],states[i].shape,states[i].size)',1)
    needle="o.update(bottom,float(target.data.root_lin_vel_w[i,2]),touch[i],bad[i],bool(heights[i]<.2 or upz[i]<=.5))"
    assert source.count(needle)==1
    source=source.replace(needle,"o.update(bottom,float(outcome_velocity_z[i]),touch[i],bad[i],bool(outcome_height[i]<np.float32(.2) or outcome_upz[i]<=np.float32(.5)))",1)
    ast.parse(source);return source


def outcome_loop(source):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [e.id for e in n.target.elts if isinstance(e,ast.Name)]==['i','o'])


def prepare(root,record):
    source=(root/'learning_episode.py').read_text();candidate=candidate_source(source)
    patch=''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile='a/legmm_skill/tabletop/learning_episode.py',tofile='b/legmm_skill/tabletop/learning_episode.py'))
    (record/'v9_outcome_readback_unapplied.patch').write_text(patch)
    return source,candidate
