"""Narrow admission of the opt-in actor-history repair over measured long capacity."""
import hashlib,json
from pathlib import Path


def approved_history_source(name,baseline,current,record_root,capacity_evidence):
    if name not in ('student.py','learning.py'):return False
    root=Path(record_root);path=root/'command_history_admission.json'
    if not path.exists():return False
    p=json.loads(path.read_text())
    if not p.get('completed') or p.get('parent_capacity_status')!=str(capacity_evidence):return False
    ack=json.loads((root/'coordinator_update_ack.json').read_text())
    if ack.get('coordinator_updates_sha256')!=hashlib.sha256((root/'COORDINATOR_UPDATES.md').read_bytes()).hexdigest():return False
    cpu=json.loads((root/'legacy_default_equivalence.json').read_text())
    if not cpu.get('completed') or not cpu.get('all_exact') or cpu.get('before')!=cpu.get('after'):return False
    cap=json.loads(Path(p['new_short_capacity_status']).read_text())
    training=json.loads(Path(cap['training_status']).read_text())
    if not cap.get('completed') or not training.get('command_history') or training.get('actor_state_dim')!=14:return False
    if cap.get('num_envs')!=64 or training.get('optimizer_updates',0)<3 or not training.get('full_session_restore_passed'):return False
    proof=json.loads((root/'real_GPU_history_restore.json').read_text())
    required={'model','Adam','normalization','reservoir','global_rng','input_contract'}
    if not proof.get('completed') or set(proof.get('checks',{}))!=required or not all(proof['checks'].values()):return False
    if not proof.get('next_batch_exact') or not proof.get('next_random_exact'):return False
    if proof.get('source_sha256')!=training.get('latest_save',{}).get('sha256'):return False
    row=p.get('files',{}).get(name,{})
    if row.get('parent_capacity_sha256')!=baseline or row.get('after_sha256')!=current:return False
    # Hash every changed source: an unreviewed subsequent edit cannot reuse this proof.
    for n,digest in p['current_source_sha256'].items():
        source=Path(__file__).with_name(n)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:return False
    return bool(p.get('physics_visual_lifecycle_unchanged') and p.get('short_probe_is_not_new_long_stability_claim'))
