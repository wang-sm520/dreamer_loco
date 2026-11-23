"""Optional bounded host-wall telemetry; no CUDA calls or simulation updates."""
import json,time
from contextlib import contextmanager
from pathlib import Path


class SegmentTimings:
    def __init__(self,enabled=False,clock_ns=time.perf_counter_ns):
        self.enabled=bool(enabled);self.clock_ns=clock_ns;self.segments={}

    def begin(self,name):return self.clock_ns() if self.enabled else None

    def end(self,name,start):
        if start is None:return
        elapsed=self.clock_ns()-start
        if elapsed<0:raise ValueError('Nonmonotonic timing clock')
        if name not in self.segments:
            if len(self.segments)>=16:raise ValueError('Bounded telemetry segment count exceeded')
            self.segments[name]=dict(count=0,total_ns=0,max_ns=0)
        row=self.segments[name];row['count']+=1;row['total_ns']+=elapsed;row['max_ns']=max(row['max_ns'],elapsed)

    def call(self,name,function,*args,**kwargs):
        start=self.begin(name)
        try:return function(*args,**kwargs)
        finally:self.end(name,start)

    @contextmanager
    def measure(self,name):
        start=self.begin(name)
        try:yield
        finally:self.end(name,start)

    def snapshot(self):
        return dict(enabled=self.enabled,clock='perf_counter_ns host wall',GPU_synchronization_added=False,
            asynchronous_GPU_work_may_be_charged_to_later_existing_wait=True,segments_inclusive=True,
            segments={k:dict(count=v['count'],total_s=v['total_ns']/1e9,max_s=v['max_ns']/1e9) for k,v in self.segments.items()})

    def write(self,path):
        if not self.enabled:return False
        data=(json.dumps(self.snapshot(),indent=2)+'\n').encode()
        from .storage import StorageGuard
        from .spec import OUTPUT,WORKTREE
        StorageGuard(OUTPUT,WORKTREE/'legmm_skill/tabletop').check(len(data))
        with Path(path).open('xb') as file:file.write(data)
        return True


_DISABLED=SegmentTimings()
def timer_for(owner):return getattr(owner,'segment_timings',_DISABLED)


def observer_source_compatible(name,baseline_sha,current_sha,record_root):
    """Accept only the exact prepared observer patch after coordinator v7 read.

    Called only when the parent already found a source hash mismatch; it does
    not bypass student/teacher/physics identities or a changed observer hash.
    """
    if name not in ('learning_episode.py','learning.py','visual.py'):return False
    import hashlib
    root=Path(record_root);ack=json.loads((root/'coordinator_update_ack.json').read_text())
    if ack['version']<7:return False
    if hashlib.sha256((root/'COORDINATOR_UPDATES.md').read_bytes()).hexdigest()!=ack['coordinator_updates_sha256']:return False
    proof=json.loads((Path(__file__).parent/'diagnostics/v7_segment_timing_proof.json').read_text())
    if proof['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():return False
    row=proof['files'].get(name,{})
    return (row.get('before_sha256')==baseline_sha and row.get('prepared_sha256')==current_sha
        and proof.get('qualified_physics_step_AST_unchanged_after_timing_removal') is True
        and proof.get('functional_observation_and_policy_AST_unchanged') is True)
