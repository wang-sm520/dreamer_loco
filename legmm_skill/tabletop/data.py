"""Bounded DAgger reservoir. Failure labels are retained by the same sampling law."""
from collections import Counter
import copy
import random
import numpy as np


class Reservoir:
    def __init__(self, capacity, seed, state_dim=7):
        if not 1 <= capacity <= 50000:
            raise ValueError('Capacity must be between 1 and 50000 frames')
        if state_dim not in (7,14):raise ValueError('Unsupported actor state dimension')
        self.state_dim=state_dim
        self.capacity, self.seed = capacity, seed
        self.rng = random.Random(seed)
        self.items = []
        self.seen = 0
        self.composition = Counter()

    def add(self, images, proprio, label, metadata):
        # Labels and outcome metadata never enter the actor input dictionary.
        if images.shape != (2,2,128,128) or proprio.shape != (self.state_dim,) or label.shape != (8,7):
            raise ValueError('Invalid depth/proprio/action tensor contract')
        if not all(np.isfinite(v).all() for v in (images,proprio,label)):
            raise FloatingPointError('Invalid DAgger sample')
        self.seen += 1
        self.composition[(metadata['shape'], metadata['outcome'],metadata['round'])] += 1
        index = len(self.items) if len(self.items)<self.capacity else self.rng.randrange(self.seen)
        if index < self.capacity:
            value = (images.astype(np.float16,copy=True),proprio.astype(np.float32,copy=True),
                     label.astype(np.float32,copy=True),copy.deepcopy(metadata))
            if index == len(self.items):
                self.items.append(value)
            else:
                self.items[index] = value

    @property
    def tensor_bytes(self):
        return sum(sum(v.nbytes for v in row[:3]) for row in self.items)

    def sample(self,batch):
        if not self.items:
            raise ValueError('Empty reservoir')
        rows = [self.items[self.rng.randrange(len(self.items))] for _ in range(batch)]
        return {key:np.stack([row[i] for row in rows]) for i,key in enumerate(('images','proprio','label'))}



class SplitReservoir:
    """Independent demonstration and recovery pools with balanced updates."""
    def __init__(self, seed, state_dim=14, capacity=15000):
        self.demo = Reservoir(capacity, seed, state_dim)
        self.dagger = Reservoir(capacity, seed + 1, state_dim)

    def for_round(self, round_index):
        return self.demo if round_index == 0 else self.dagger

    @property
    def retained(self):
        return len(self.demo.items) + len(self.dagger.items)

    @property
    def tensor_bytes(self):
        return self.demo.tensor_bytes + self.dagger.tensor_bytes

    def sample(self, batch):
        recovery_count = min(batch // 2, len(self.dagger.items))
        demo = self.demo.sample(batch - recovery_count)
        if not recovery_count:
            return demo
        recovery = self.dagger.sample(recovery_count)
        return {key: np.concatenate((demo[key], recovery[key])) for key in demo}

def actor_inputs(images, proprio, state_dim=7):
    """Structural whitelist; neither metadata nor critic dict is accepted."""
    if images.ndim != 5 or images.shape[1:] != (2,2,128,128) or proprio.shape != (len(images),state_dim) or state_dim not in (7,14):
        raise ValueError('Expected dual-depth images and explicit native actor state dimension')
    return {'images':images, 'proprio':proprio}
