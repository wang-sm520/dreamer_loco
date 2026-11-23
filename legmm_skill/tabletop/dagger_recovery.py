"""Atomic bounded full-state recovery with explicit partial-update handling.

Complete-round states resume at the next round. A partial dagger_update state
requires explicit authorization and SHA, then resumes its exact remaining updates.
Collection/validation partial stages are recorded but rejected by this loader.
"""
import copy,gzip,hashlib,json,os,random,resource,time
from pathlib import Path
import numpy as np
import torch
from .lifecycle_cleanup import memory_snapshot


class _HashingWriter:
    """Hash a stream, with bounded writes and periodic disk guard checks."""
    chunk_bytes=8*1024**2
    check_interval=64*1024**2

    def __init__(self,file,guard=None):
        self.file=file
        self.guard=guard
        self.bytes=0
        self.sha256=hashlib.sha256()
        self.next_check=0

    def write(self,data):
        view=memoryview(data).cast('B')
        for offset in range(0,view.nbytes,self.chunk_bytes):
            chunk=view[offset:offset+self.chunk_bytes]
            if self.guard is not None and self.bytes+len(chunk)>self.next_check:
                self.guard.check(len(chunk))
                self.next_check=self.bytes+self.check_interval
            if self.file.write(chunk)!=len(chunk):
                raise OSError('Short recovery checkpoint write')
            self.sha256.update(chunk)
            self.bytes+=len(chunk)
        return view.nbytes

    def flush(self):
        self.file.flush()



def reservoir_state(reservoir):
    from .data import SplitReservoir
    if isinstance(reservoir, SplitReservoir):
        return dict(demo=reservoir_state(reservoir.demo), dagger=reservoir_state(reservoir.dagger))
    return dict(capacity=reservoir.capacity,seed=reservoir.seed,items=reservoir.items,
                seen=reservoir.seen,composition=reservoir.composition,rng=reservoir.rng.getstate())


def reservoir_targets(reservoir, state):
    from .data import SplitReservoir
    split = isinstance(reservoir, SplitReservoir)
    if split != ('demo' in state and 'dagger' in state):
        raise ValueError('Recovery reservoir mode differs')
    if split:
        return [(reservoir.demo, state['demo']), (reservoir.dagger, state['dagger'])]
    return [(reservoir, state)]

def save_latest(session,status,next_round,next_seed,episode,out,guard,validation_pending=None,complete_round_boundary=True,compress=False):
    if session.student.device.type=='cuda':torch.cuda.synchronize()
    student=session.student;reservoir=session.reservoir
    before=memory_snapshot();started=time.monotonic()
    payload=dict(format='w8_dagger_full_round_v1' if complete_round_boundary else 'w8_dagger_partial_v1',complete_round_boundary=complete_round_boundary,
        student=dict(model=student.model.state_dict(),optimizer=student.optimizer.state_dict(),updates=student.updates,
            model_config=student.model.config,mean=student.mean,scale=student.scale),
        reservoir=reservoir_state(reservoir),
        collection=dict(seen=session.seen,budget=session.budget,round=session.round,teacher_drives=session.teacher_drives,
            episode_serial=session.episode_serial,physical_env_decisions=session.physical_env_decisions,collected_samples=session.collected_samples),
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if student.device.type=='cuda' else []),
        cursor=dict(next_round=next_round,next_seed=next_seed,episode=episode,validation_pending=validation_pending,
            partial_phase=None if complete_round_boundary else status.get('last_phase'),
            updates_completed_in_round=status.get('current_round_update_count',0),
            loss_sum=status.get('current_round_loss_sum',0.),update_elapsed_s=status.get('current_round_update_elapsed_s',0.)),
        status=copy.deepcopy(status),visual=session.visual.manifest())
    if getattr(student,'command_history',False):payload['student']['input_contract']=student.input_contract
    if getattr(student,'relative_delta',False):
        payload['student'].update(relative_delta=True,output_contract=student.output_contract,delta_scale=student.delta_scale)
    target=Path(out)/'latest.pt';temporary=Path(out)/'latest.pt.pending'
    file=temporary.open('xb')
    try:
        with file:
            outer=_HashingWriter(file,guard)
            if compress:
                with gzip.GzipFile(filename='',fileobj=outer,mode='wb',compresslevel=3,mtime=0) as zipped:
                    inner=_HashingWriter(zipped)
                    # Torch's ZIP writer first pickles all NumPy reservoir arrays
                    # into a single BytesIO. The sequential Torch format sends
                    # protocol-5 Pickler writes directly to our bounded writer.
                    torch.save(payload,inner,pickle_protocol=5,_use_new_zipfile_serialization=False)
            else:
                inner=outer
                torch.save(payload,inner,pickle_protocol=5,_use_new_zipfile_serialization=False)
            serialized=memory_snapshot()
            file.flush()
            guard.check()
            os.fsync(file.fileno())
    except BaseException:
        temporary.unlink()
        raise
    os.replace(temporary,target)
    # fsync the directory as well: a completed atomic checkpoint survives a
    # process exit without relying on the directory entry's writeback timing.
    descriptor=os.open(out,os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    record=dict(path=str(target),bytes=outer.bytes,sha256=outer.sha256.hexdigest(),
        encoding='gzip' if compress else 'torch',uncompressed_bytes=inner.bytes,uncompressed_sha256=inner.sha256.hexdigest(),
        optimizer_updates=student.updates,env_decisions=session.seen,rounds_completed=next_round,validation_pending=validation_pending,complete_round_boundary=complete_round_boundary,
        before=before,after_serialization=serialized,after_write=memory_snapshot(),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_s=time.monotonic()-started)
    with (Path(out)/'latest_save_memory.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
    return record


def partial_update_plan(payload,updates_per_round=2000):
    cursor=payload['cursor'];collection=payload['collection']
    if cursor.get('partial_phase')!='dagger_update':raise ValueError('Unsupported partial stage; only dagger_update is authorized')
    completed=cursor['updates_completed_in_round']
    if not 0<completed<updates_per_round:raise ValueError('Invalid partial update count')
    if collection['seen']!=collection['budget']:raise ValueError('Partial update cannot skip missing collection')
    if cursor['next_round']!=collection['round']:raise ValueError('Partial round cursor differs')
    if payload['student']['updates']!=cursor['next_round']*updates_per_round+completed:raise ValueError('Model and phase update counters differ')
    cache=payload['reservoir']
    pools=[cache['demo'],cache['dagger']] if 'demo' in cache else [cache]
    if not pools[0]['items'] or any(len(pool['items'])>pool['capacity'] for pool in pools):
        raise ValueError('Recovery buffer is incomplete')
    return dict(round=cursor['next_round'],completed_updates=completed,remaining_updates=updates_per_round-completed,
        collection_decisions=collection['seen'],missing_collection_decisions=0,next_seed=cursor['next_seed'])


def restore_latest(session,path,allow_partial=False,expected_sha256=None,expected_formal=None,updates_per_round=2000,allow_closed_collection=False,expected_num_envs=64,allow_resource_abort=False,validation_manifest=None):
    if (allow_partial or allow_closed_collection or allow_resource_abort or validation_manifest) and expected_sha256 is None:raise ValueError('Explicit SHA required for partial recovery')
    if sum(map(bool,(allow_partial,allow_closed_collection,allow_resource_abort,validation_manifest)))>1:raise ValueError('Recovery stage flags are mutually exclusive')
    if allow_resource_abort and expected_formal is None:raise ValueError('Explicit formal/diagnostic identity required')
    if validation_manifest and (expected_formal is not True or expected_num_envs!=64):raise ValueError('Explicit formal64 validation recovery required')
    with Path(path).open('rb') as file:
        if expected_sha256 is not None:
            digest=hashlib.sha256()
            while chunk:=file.read(8*1024**2):digest.update(chunk)
            if digest.hexdigest()!=expected_sha256:raise ValueError('Recovery SHA differs')
            file.seek(0)
        signature=file.read(2);file.seek(0)
        reader=gzip.GzipFile(fileobj=file,mode='rb') if signature==b'\x1f\x8b' else file
        try:payload=torch.load(reader,map_location='cpu',weights_only=False)
        finally:
            if reader is not file:reader.close()
    if payload.get('format') not in ('w8_dagger_full_round_v1','w8_dagger_partial_v1'):
        raise ValueError('A full-state recovery checkpoint is required; model-only is insufficient')
    if not payload.get('complete_round_boundary') and not (allow_partial or allow_closed_collection or allow_resource_abort or validation_manifest):raise ValueError('Partial recovery requires explicit stage handling')
    if expected_formal is not None and bool(payload['status'].get('formal_training'))!=expected_formal:
        raise ValueError('Recovery formal/diagnostic identity differs')
    if allow_resource_abort and (payload.get('complete_round_boundary') or payload['cursor'].get('partial_phase')!='dagger_student_collection'):raise ValueError('Resource recovery requires interrupted student collection stage')
    if validation_manifest and (payload.get('complete_round_boundary') or payload['cursor'].get('partial_phase')!='student_fixed_validation'):raise ValueError('Validation recovery requires interrupted fixed validation')
    if not payload.get('complete_round_boundary'):
        if 'demo' in payload['reservoir'] and (allow_closed_collection or allow_resource_abort or validation_manifest):
            raise ValueError('Use the previous complete split-pool boundary for interrupted rollouts')
        if validation_manifest:
            from .validation_recovery import recovery_plan
            plan=recovery_plan(payload,validation_manifest,expected_sha256)
            payload['cursor'].update(next_round=plan['next_round'],validation_pending=plan['validation_pending'],partial_phase=None,validation_recovery_plan=plan)
        elif payload['cursor'].get('partial_phase')=='dagger_update':
            if not allow_partial:raise ValueError('Explicit partial update flag required')
            partial_update_plan(payload,updates_per_round)
        elif allow_resource_abort:
            from .resource_abort_recovery import resource_abort_plan
            payload['cursor']['resource_abort_plan']=resource_abort_plan(payload,updates_per_round,expected_num_envs)
        elif allow_closed_collection:
            from .closed_collection_recovery import closed_collection_plan
            payload['cursor']['closed_collection_plan']=closed_collection_plan(payload,updates_per_round,expected_num_envs)
        else:partial_update_plan(payload,updates_per_round)
    student=session.student;source=payload['student']
    if source['model_config']!=student.model.config:raise ValueError('Recovery model dimensions differ')
    if hasattr(student,'validate_recovery_contract'):student.validate_recovery_contract(source)
    elif any(key in source for key in ('input_contract','relative_delta','output_contract','delta_scale')):
        raise ValueError('Unknown student recovery consumer contract')
    expected_state=14 if getattr(student,'command_history',False) else 7
    targets=reservoir_targets(session.reservoir,payload['reservoir'])
    for target, state in targets:
        if state['capacity']!=target.capacity or len(state['items'])>target.capacity:
            raise ValueError('Recovery reservoir capacity differs')
        if any(row[1].shape!=(expected_state,) for row in state['items']):
            raise ValueError('Recovery buffer actor-state dimension differs')
    if not torch.equal(source['mean'],student.mean.cpu()) or not torch.equal(source['scale'],student.scale.cpu()):raise ValueError('Recovery normalization differs')
    if payload['visual']!=session.visual.manifest():raise ValueError('Recovery visual contract differs')
    student.model.load_state_dict(source['model']);student.optimizer.load_state_dict(source['optimizer']);student.updates=source['updates'];student.model.eval()
    for target, state in targets:
        for name in ('capacity','seed','items','seen','composition'):setattr(target,name,state[name])
        target.rng.setstate(state['rng'])
    for name,value in payload['collection'].items():setattr(session,name,value)
    random.setstate(payload['rng']['python']);np.random.set_state(payload['rng']['numpy']);torch.set_rng_state(payload['rng']['torch'].cpu())
    if student.device.type=='cuda':torch.cuda.set_rng_state_all([state.cpu() for state in payload['rng']['cuda']])
    return payload['status'],payload['cursor']
