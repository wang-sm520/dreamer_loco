"""Value-level full-state fingerprints; device/storage padding are excluded."""
import hashlib,random
import numpy as np
import torch


def digest(value):
    h=hashlib.sha256()
    def visit(x):
        h.update(type(x).__name__.encode()+b':')
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=repr):visit(key);visit(x[key])
        elif isinstance(x,(list,tuple)):
            h.update(str(len(x)).encode())
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def expected_fingerprints(payload):
    s=payload['student'];r=payload['reservoir']
    return dict(model=digest(s['model']),Adam=digest(s['optimizer']),normalization=digest(dict(mean=s['mean'],scale=s['scale'])),
        reservoir=digest(r),global_rng=digest(payload['rng']),collection=digest(payload['collection']))


def capture_rng(cuda=False):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if cuda else [])


def session_fingerprints(session):
    s=session.student;r=session.reservoir
    reservoir=dict(capacity=r.capacity,seed=r.seed,items=r.items,seen=r.seen,composition=r.composition,rng=r.rng.getstate())
    collection=dict(seen=session.seen,budget=session.budget,round=session.round,teacher_drives=session.teacher_drives,
        episode_serial=session.episode_serial,physical_env_decisions=session.physical_env_decisions,collected_samples=session.collected_samples)
    return dict(model=digest(s.model.state_dict()),Adam=digest(s.optimizer.state_dict()),normalization=digest(dict(mean=s.mean,scale=s.scale)),
        reservoir=digest(reservoir),global_rng=digest(capture_rng(s.device.type=='cuda')),collection=digest(collection))


def expected_next_batch(payload,batch=64):
    r=payload['reservoir'];rng=random.Random();rng.setstate(r['rng']);rows=[r['items'][rng.randrange(len(r['items']))] for _ in range(batch)]
    return digest({key:np.stack([row[i] for row in rows]) for i,key in enumerate(('images','proprio','label'))})
