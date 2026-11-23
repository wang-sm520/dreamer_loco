"""One-shot v10 observer of owned files/PID; never modifies the training process.

A single SIGTERM is allowed only after the atomic full target exists, is fully
verified, and the next cohort directory has not been created. Missing that
conservative window means continuing training, never chasing a collection.
"""
import argparse,gc,hashlib,json,os,pickle,signal,time,datetime
from pathlib import Path


def pidfd_open(pid):
    import ctypes
    if os.uname().machine!='x86_64':raise RuntimeError('Unreviewed pidfd syscall architecture')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    result=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if result<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return result


def pidfd_signal(fd,number):
    import ctypes
    if os.uname().machine!='x86_64':raise RuntimeError('Unreviewed pidfd syscall architecture')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    result=libc.syscall(ctypes.c_long(424),ctypes.c_int(fd),ctypes.c_int(number),ctypes.c_void_p(0),ctypes.c_uint(0))
    if result<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))


def starttime(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])


def identity(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)


def read_owned_source(path):
    # The pathname may be atomically renamed while the descriptor remains valid.
    import torch
    with Path(path).open('rb') as file:
        payload=torch.load(file,map_location='cpu',weights_only=False)
        before=os.fstat(file.fileno());before_key=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
        file.seek(0);digest=hashlib.sha256()
        for part in iter(lambda:file.read(8*1024**2),b''):digest.update(part)
        after=os.fstat(file.fileno());after_key=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
        if before_key!=after_key:raise RuntimeError('Owned snapshot still changing')
    return payload,digest.hexdigest(),after_key


def eligible(payload):
    c=payload.get('collection',{});s=payload.get('student',{});u=payload.get('cursor',{});status=payload.get('status',{});r=payload.get('reservoir',{})
    return bool(payload.get('format')=='w8_dagger_full_round_v1' and payload.get('complete_round_boundary') is True
        and status.get('formal_training') is True and status.get('diagnostic_only') is False and status.get('num_envs')==64
        and c.get('seen')==c.get('budget')==c.get('physical_env_decisions')==64000 and c.get('round')==1 and c.get('teacher_drives') is False
        and s.get('updates')==4000 and u.get('next_round')==2 and u.get('next_seed')==1024 and u.get('episode')==12
        and u.get('partial_phase') is None and u.get('validation_pending') is None
        and r.get('capacity')==10000 and len(r.get('items',[]))==10000 and r.get('seen')==c.get('collected_samples')
        and sum(r.get('composition',{}).values())==r.get('seen') and set(payload.get('rng',{}))=={'python','numpy','torch','cuda'})


def read_json(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--record-root',type=Path,required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--starttime',type=int,required=True);a=p.parse_args()
    from .spec import OUTPUT
    if not a.out.resolve().is_relative_to(OUTPUT):raise ValueError('Only owned W8 output')
    import torch
    torch.set_num_threads(2)
    r=a.record_root;target=a.out/'latest.pt';pending=a.out/'latest.pt.pending';next_cohort=a.out/'round_02_episode_0012'
    status_path=r/'v10_round_end_observer_v2_status.json';assert not status_path.exists()
    source_sha=hashlib.sha256((r/'COORDINATOR_UPDATES.md').read_bytes()).hexdigest()
    ack=read_json(r/'coordinator_update_ack.json');assert ack['version']==10 and ack['coordinator_updates_sha256']==source_sha
    state=dict(observer_pid=os.getpid(),worker_pid=a.pid,worker_starttime=a.starttime,phase='watching_owned_round_end',SIGTERM_sent=False,signal_count=0,preverification_attempts=0,coordinator_sha256=source_sha,next_cohort_must_not_exist=str(next_cohort),expected_decisions=64000,expected_updates=4000)
    def save(**fields):
        state.update(fields,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status_path)
    def same_worker():
        try:
            cmd=Path(f'/proc/{a.pid}/cmdline').read_bytes()
            return starttime(a.pid)==a.starttime and Path(f'/proc/{a.pid}').stat().st_uid==os.getuid() and b'tabletop.train_dagger' in cmd and str(a.out).encode() in cmd
        except FileNotFoundError:return False
    assert same_worker();pidfd=pidfd_open(a.pid);save()
    cached_identity=None;payload=None;sha=None;last_attempt=None
    try:
        while same_worker():
            if hashlib.sha256((r/'COORDINATOR_UPDATES.md').read_bytes()).hexdigest()!=source_sha:
                save(phase='disarmed_new_coordinator_version');return
            training=read_json(a.out/'training_status.json')
            if training.get('env_decisions',0)>64000 or training.get('optimizer_updates',0)>4000:
                save(phase='missed_window_training_continues',reason='Observed next-round decisions/updates; no signal');return
            source=target if target.exists() else pending if pending.exists() else None
            # Existing non-target partial checkpoints are rechecked only if replaced.
            if source is not None:
                try:key=identity(source)
                except FileNotFoundError:continue
                if key!=cached_identity and key!=last_attempt:
                    available=int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
                    if available<16*1024**3:
                        save(phase='verification_skipped_RAM_training_continues',available_ram_bytes=available);return
                    state['preverification_attempts']+=1;last_attempt=key
                    try:
                        candidate,verified_sha,verified_key=read_owned_source(source)
                        key=verified_key
                        if not eligible(candidate):
                            save(last_rejected_payload=dict(format=candidate.get('format'),complete=candidate.get('complete_round_boundary'),decisions=candidate.get('collection',{}).get('seen'),updates=candidate.get('student',{}).get('updates')));candidate=None;gc.collect();continue
                        from .physics_config import verify_task_physics_audit
                        episode=Path(candidate['status']['current_episode']);verify_task_physics_audit(read_json(episode/'physics_audit.json'))
                        assert (episode/'post_stage_memory_release.json').is_file()
                        payload=candidate;cached_identity=key;sha=verified_sha;save(phase='full_payload_preverified_waiting_atomic_target',payload_read_from=str(source),source_identity=list(key),source_sha256=sha)
                    except (OSError,ValueError,EOFError,RuntimeError,AssertionError,pickle.UnpicklingError) as error:
                        save(last_preverification_error=repr(error));candidate=None;payload=None;gc.collect()
                if payload is not None and target.exists() and identity(target)==cached_identity:
                    # Deliberately conservative: no next cohort directory, not merely a stale counter.
                    now=read_json(a.out/'training_status.json')
                    if next_cohort.exists() or now.get('env_decisions',0)>64000 or now.get('optimizer_updates',0)>4000:
                        save(phase='missed_window_training_continues',reason='Next cohort exists or new decisions; no chasing',next_cohort_exists=next_cohort.exists());return
                    if not same_worker():break
                    save(phase='verified_atomic_round_end_requesting_SIGTERM',atomic_target=str(target),atomic_target_identity=list(identity(target)),source_sha256=sha,next_cohort_absent_at_final_check=True,raw_status_before_signal=now)
                    pidfd_signal(pidfd,signal.SIGTERM)
                    save(phase='SIGTERM_requested_at_verified_round_end',SIGTERM_sent=True,signal_count=1,signal='SIGTERM',natural_exit=False)
                    from .state_fingerprints import expected_fingerprints,expected_next_batch
                    proof=dict(observer_pid=os.getpid(),observer_status_file=str(status_path),worker_pid=a.pid,worker_starttime=a.starttime,source_path=str(target),source_sha256=sha,bytes=cached_identity[2],format=payload['format'],complete_round_boundary=True,formal=True,num_envs=64,model_updates=4000,cursor=payload['cursor'],collection=payload['collection'],source_fingerprints=expected_fingerprints(payload),expected_next_batch64_sha256=expected_next_batch(payload),GPU_restore_verified=False,source_unmodified=True)
                    (r/'v10_full_round_source_snapshot.json').write_text(json.dumps(proof,indent=2)+'\n');save(source_snapshot=str(r/'v10_full_round_source_snapshot.json'));return
            time.sleep(.05 if pending.exists() or training.get('current_round_update_count',0)>=1900 else 1.)
        save(phase='worker_exited_without_observer_signal')
    finally:os.close(pidfd)
if __name__=='__main__':main()
