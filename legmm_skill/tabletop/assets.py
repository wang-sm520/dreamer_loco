"""Generate the new collidable A2 model; source W2 and M1 assets stay read-only."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from .spec import W8, WORKTREE


def build_robot(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'a2_tabletop_mesh_shoulder_v2.xml'
    if destination.exists():
        return destination
    root = ET.parse(W8 / 'stand_probe/model/a2_arm.xml').getroot()
    source = ET.parse(WORKTREE / 'legmm_skill/airbot_rl/assets/airbot_play.xml').getroot()
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    for name, target in [('a2', W8/'stand_probe/from_mac/robots/a2/assets/a2'),
                         ('airbot_play', WORKTREE/'legmm_skill/airbot_rl/assets/meshes/airbot_play')]:
        link = assets / name
        if not link.exists():
            link.symlink_to(target.resolve(), target_is_directory=True)
    # Restore collision and visual geometry stripped from the W2 probe, without
    # replacing its verified inertials, mount transform, joints, or free base.
    for item in source.find('asset'):
        root.find('asset').append(deepcopy(item))
    source_bodies = {b.get('name'): b for b in source.iter('body')}
    for body in root.iter('body'):
        name = body.get('name')
        if name not in source_bodies:
            continue
        for geom in list(body.findall('geom')):
            body.remove(geom)
        for original in source_bodies[name].findall('geom'):
            # The source shoulder boxes overlap the mount during legal motion.
            # Use the supplied separate convex mesh parts for these bodies,
            # rather than exempting a nonadjacent pair from self collision.
            if name in ('arm_base','link2') and original.get('class')=='collision':
                continue
            geom = deepcopy(original)
            geom.set('class', 'arm_' + geom.get('class'))
            geom.set('mass', '0')  # retain explicit W2 inertials, including mount
            body.append(geom)
        # Source has no colliders on link1/4/5: use its existing mesh as a convex
        # collider, massless (no guessed primitive size, no extra camera mass).
        if name in ('arm_base','link1','link2','link4','link5'):
            for original in source_bodies[name].findall('geom'):
                if original.get('mesh') is None:
                    continue
                geom = deepcopy(original)
                geom.set('class', 'arm_collision')
                geom.set('mass','0')
                geom.attrib.pop('material',None)
                body.append(geom)
    # Contact geometry is active. Mount is bolted; adjacent link overlaps are
    # filtered in the scene, not by disabling table/object collisions.
    for default in root.findall('.//default'):
        if (default.get('class') or '').startswith('arm_'):
            for joint in default.findall('joint'):
                joint.set('frictionloss','0')
    root.find('compiler').set('meshdir','assets')
    ET.ElementTree(root).write(destination)
    return destination


def geometry_xml(robot_path, shape, state=None):
    """MuJoCo is used ONLY for geometric queries, never as the training physics."""
    root = ET.parse(robot_path).getroot()
    root.find('compiler').set('meshdir',str(Path(robot_path).parent/'assets'))
    world = root.find('worldbody')
    ET.SubElement(world,'geom',name='floor',type='plane',size='5 5 .1')
    table_size = (.5,.4,.3) if state is None else tuple(v/2 for v in state.table_size)
    table = ET.SubElement(world,'body',name='table',pos=f'0 0 {table_size[2]}')
    ET.SubElement(table,'geom',name='table_geom',type='box',size=' '.join(map(str,table_size)))
    obj = ET.SubElement(world,'body',name='object',pos='0 0 .7')
    ET.SubElement(obj,'freejoint',name='object_free')
    obj_size = '.02 .06 .07' if shape == 'box' else '.025 .06'
    if state is not None:
        dimensions = [v/2 for v in state.size] if shape=='box' else [state.size[0]/2,state.size[2]/2]
        obj_size = ' '.join(map(str,dimensions))
    ET.SubElement(obj,'geom',name='object_geom',type=shape,size=obj_size,
                  mass=str(.2 if state is None else state.mass))
    return ET.tostring(root,encoding='unicode')
