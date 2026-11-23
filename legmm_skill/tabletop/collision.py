"""Mechanical neighbours excluded from self contact; all other arm pairs collide."""
ARM_CHAIN = ('base_link', 'arm_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6')
NEIGHBOURS = {frozenset((a, b)) for a, b in zip(ARM_CHAIN, ARM_CHAIN[1:])}
NEIGHBOURS.update(frozenset(('link6', b)) for b in ('left', 'right', 'payload'))


def excluded_pair(a, b, dog_bodies):
    return (a in dog_bodies and b in dog_bodies) or frozenset((a, b)) in NEIGHBOURS
