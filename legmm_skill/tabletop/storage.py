"""Sparse rolling writes authorised by w8_1_sparse_storage_addendum_20261114.md."""
from contextlib import contextmanager
from contextvars import ContextVar
import io
import json
import os
from pathlib import Path
import shutil

FLOOR_BYTES = 10_000_000_000
BUDGET_BYTES = 10_000_000_000
_ACTIVE_GUARD = ContextVar('w8_active_storage_guard',default=None)


@contextmanager
def scoped_storage_guard(root,code_root,guard):
    """Route the legacy audit helpers to the active run's storage accounting."""
    token=_ACTIVE_GUARD.set((Path(root),Path(code_root),guard))
    try:
        yield
    finally:
        _ACTIVE_GUARD.reset(token)


def allocated_bytes(root):
    return sum(p.stat().st_blocks*512 for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def save_due(stage,completed):
    cadence = {'dagger':3,'ppo':500}[stage]
    return completed > 0 and completed % cadence == 0


class StorageGuard:
    def __init__(self, root, code_root, free=shutil.disk_usage):
        self.root,self.code_root,self.free = Path(root),Path(code_root),free

    def check(self,write_bytes=0):
        override=_ACTIVE_GUARD.get()
        if override is not None:
            root,code_root,guard=override
            if self.root==root and self.code_root==code_root:
                return guard.check(write_bytes)
        available = self.free(self.root).free
        # Includes any existing rolling checkpoint plus a full atomic temp copy.
        usage = allocated_bytes(self.root)+allocated_bytes(self.code_root)
        peak = ((write_bytes+4095)//4096)*4096
        if available-peak < FLOOR_BYTES:
            raise OSError(f'W8 storage stop: free={available}, write_peak={peak}, required_after={FLOOR_BYTES}, path={self.root}')
        if usage+peak > BUDGET_BYTES:
            raise OSError(f'W8 budget stop: allocated={usage}, write_peak={peak}, budget={BUDGET_BYTES}')
        return dict(free_bytes=available, allocated_bytes=usage, write_peak_bytes=peak)

    def atomic_bytes(self, name, data):
        if name not in ('latest.pt','best.pt','student.pt','status.json'):
            raise ValueError('Only predefined W8 rolling paths may be replaced')
        target = self.root/name
        temp = self.root/(name+'.pending')
        self.check(len(data))
        # A pre-existing temp belongs to an interrupted write: never discard it
        # silently. Recovery must inspect it and use a new attempt directory.
        with temp.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp,target)
        return len(data)

    def checkpoint(self,name,payload):
        import torch
        stream = io.BytesIO()
        torch.save(payload,stream)
        # Measured serialization size, including optimizer; no parameter-count guess.
        return self.atomic_bytes(name,stream.getvalue())


class SparseSaveGate:
    """Use around framework save callbacks; defaults cannot bypass the cadence."""
    def __init__(self,stage,writer):
        self.stage,self.writer = stage,writer
        self.calls = []
        self.writes = []

    def save(self,completed,payload,final=False):
        self.calls.append(completed)
        if not final and not save_due(self.stage,completed):
            return False
        self.writer(payload)
        self.writes.append(completed)
        return True
