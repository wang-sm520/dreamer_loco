"""Fail-closed source and evidence checks for the opt-in W8 relative-delta run."""
import hashlib
import json
from pathlib import Path
import shutil

from .foot_admission import approved_foot_source
from .segment_timing import observer_source_compatible
from .spec import OUTPUT

CAPACITY_SOURCES = (
    'student.py', 'learning.py', 'visual.py', 'learning_episode.py', 'isaac_scene.py',
    'geometry.py', 'closed_loop.py', 'audit_all_rigid.py', 'learning_geometry.py',
    'lifecycle_cleanup.py', 'control.py', 'spec.py', 'outcome.py', 'collision.py',
    'physics_config.py', 'assets.py', 'physics_audit.py',
)
LEARNED_SOURCES = ('student.py', 'learning.py')
RUNTIME_SOURCES = (*LEARNED_SOURCES, 'dagger_recovery.py', 'train_dagger.py',
                   'learning_launch.py', 'reldelta_admission.py', 'compact_current_logs.py', 'storage.py')
STATELESS_RUNTIME_SOURCES = ('data.py', 'stateless_teacher.py', 'gpu_guard.py')
RESTORE_CHECKS = {
    'model', 'Adam', 'normalization', 'delta_scale', 'reservoir', 'global_rng',
    'input_contract', 'output_contract',
}
MIN_FREE_BYTES = 20_000_000_000
MAX_ADDED_BYTES = 25_000_000_000


def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def source_inventory(directory):
    return {Path(row['path']).name:row['sha256']
            for row in read_json(directory/'source_inventory.json')}


def runtime_hashes(stateless_dagger=False):
    root=Path(__file__).parent
    sources = RUNTIME_SOURCES + STATELESS_RUNTIME_SOURCES if stateless_dagger else RUNTIME_SOURCES
    return {name:sha256(root/name) for name in sources}


def require_parent_sources(capacity_evidence,record_root):
    inventory=source_inventory(Path(capacity_evidence).parent)
    source_root=Path(__file__).parent
    record_root=Path(record_root)
    for name in CAPACITY_SOURCES:
        if name in LEARNED_SOURCES:
            continue
        baseline=inventory.get(name)
        current=sha256(source_root/name)
        if baseline==current:
            continue
        if name in ('visual.py','learning_episode.py') and (
                observer_source_compatible(name,baseline,current,record_root)
                or approved_foot_source(name,baseline,current,record_root)):
            continue
        raise ValueError('Relative-delta capacity source differs: '+name)
    return inventory


def require_reldelta_admission(record_root,capacity_evidence,*,gpu=0,stateless_dagger=False):
    root=Path(record_root)
    if gpu not in (0, 1, 2, 3):
        raise ValueError('Relative-delta GPU must be in {0, 1, 2, 3}')
    baseline=require_parent_sources(capacity_evidence,root)
    admission=read_json(root/'reldelta_admission.json')
    if admission.get('parent_capacity_status')!=str(capacity_evidence):
        raise ValueError('Relative-delta parent capacity identity differs')
    ack=read_json(root/'coordinator_update_ack.json')
    if ack.get('coordinator_updates_sha256')!=sha256(root/'COORDINATOR_UPDATES.md'):
        raise ValueError('Relative-delta coordinator authorization differs')
    cpu=read_json(root/'cpu_paired_400updates.json')
    if not (cpu.get('completed') and cpu.get('gate_passed') and cpu.get('completed_updates_each')==400):
        raise ValueError('Relative-delta paired CPU gate did not pass')

    short_path=Path(admission['short_capacity_status'])
    short=read_json(short_path)
    training=read_json(short['training_status'])
    if not (short.get('completed') and short.get('CUDA_VISIBLE_DEVICES')==str(gpu)
            and short['num_envs']==64 and 0<short['peak_gpu_mib']<=18432
            and training.get('command_history') and training.get('relative_delta')
            and training.get('actor_state_dim')==14 and training.get('optimizer_updates',0)>=3
            and training.get('full_session_restore_passed')):
        raise ValueError('Relative-delta short GPU capacity gate did not pass')
    save=training['latest_save']
    if save['encoding']!='gzip' or save['after_serialization']['rss_bytes']-save['before']['rss_bytes']>=2_200_000_000:
        raise ValueError('Relative-delta streaming memory evidence failed')
    proof=read_json(root/'real_GPU_reldelta_restore.json')
    if not (proof.get('completed') and proof.get('CUDA_VISIBLE_DEVICES')==str(gpu)
            and 0<proof.get('peak_gpu_mib',0)<=18432 and proof.get('source_sha256')==save['sha256']
            and proof.get('next_batch_exact') and proof.get('next_random_exact')
            and set(proof['checks'])==RESTORE_CHECKS and all(proof['checks'].values())):
        raise ValueError('Relative-delta CUDA state restoration proof failed')
    if stateless_dagger:
        smoke_path=Path(admission['teacher_smoke_status'])
        if not smoke_path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Stateless teacher smoke must belong to this task')
        smoke=read_json(smoke_path.with_name('summary.json'))
        smoke_status=read_json(smoke_path)
        smoke_inventory=source_inventory(smoke_path.parent)
        if not (smoke['teacher_smoke_passed'] and smoke['success_count']>=54 and smoke['num_episodes']==64
                and smoke['collision_count']==0 and smoke_status.get('completed')
                and smoke_status['CUDA_VISIBLE_DEVICES']==str(gpu)
                and 0<smoke_status['peak_gpu_mib']<=18432
                and smoke_inventory.get('stateless_teacher.py')==sha256(Path(__file__).with_name('stateless_teacher.py'))
                and training.get('stateless_dagger') and all(proof.get('pool_frames',{}).get(pool,0)>0
                                                            for pool in ('demo','dagger'))):
            raise ValueError('Stateless teacher smoke or dual-pool CUDA evidence failed')
    current=runtime_hashes(stateless_dagger)
    if admission.get('runtime_sha256')!=current or proof.get('runtime_sha256')!=current:
        raise ValueError('Relative-delta runtime sources changed after evidence')
    short_inventory=source_inventory(short_path.parent)
    if any(short_inventory.get(name)!=digest for name,digest in current.items()):
        raise ValueError('Relative-delta short probe source inventory differs')
    expected={name:dict(parent_capacity_sha256=baseline[name],after_sha256=current[name])
              for name in LEARNED_SOURCES}
    if admission.get('files')!=expected:
        raise ValueError('Relative-delta learning source admission differs')


class RelativeDeltaStorageGuard:
    """Apply this task's budget to its own control, probe and formal output."""
    def __init__(self,record_root,out,recovery_root=None):
        owner=OUTPUT.parent.resolve()
        roots=sorted({Path(path).resolve() for path in (record_root,out,recovery_root) if path is not None},
                     key=lambda path:len(path.parts))
        if any(path==owner or path==OUTPUT or not path.is_relative_to(owner) for path in roots):
            raise ValueError('Relative-delta evidence requires a dedicated W8 output directory')
        self.roots=[]
        for path in roots:
            if not any(path.is_relative_to(root) for root in self.roots):
                self.roots.append(path)
        self.partition=owner

    def check(self,write_bytes=0):
        free=shutil.disk_usage(self.partition).free
        usage=0
        for root in self.roots:
            for path in root.rglob('*'):
                if path.is_symlink():continue
                try:
                    if path.is_file():usage+=path.stat().st_blocks*512
                except FileNotFoundError:
                    continue  # a concurrent atomic checkpoint/log move
        peak=((write_bytes+4095)//4096)*4096
        if free-peak<MIN_FREE_BYTES or usage+peak>MAX_ADDED_BYTES:
            raise OSError(f'Relative-delta storage stop: free={free}, allocated={usage}, write_peak={peak}')
        return dict(free_bytes=free,allocated_bytes=usage,write_peak_bytes=peak)
