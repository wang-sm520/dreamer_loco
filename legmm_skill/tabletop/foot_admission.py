"""Exact source admission for the separately proven transport-only foot patch."""
import hashlib,json
from pathlib import Path

def approved_foot_source(name,baseline,current,record_root):
    if name!='learning_episode.py':return False
    root=Path(record_root);proof_path=root/'v9_foot_adoption.json'
    if not proof_path.is_file():return False
    proof=json.loads(proof_path.read_text());ack=json.loads((root/'coordinator_update_ack.json').read_text())
    if ack['version']<9 or hashlib.sha256((root/'COORDINATOR_UPDATES.md').read_bytes()).hexdigest()!=ack['coordinator_updates_sha256']:return False
    helper=Path(__file__).with_name('foot_readback_candidate.py')
    return bool(proof.get('adopted') and proof.get('actual_GPU_all_pairs_exact') and proof.get('same_state_timing_meaningful_improvement') and proof.get('resource_guards_satisfied')
        and proof.get('capacity_baseline_sha256')==baseline and proof.get('adopted_runtime_sha256')==current
        and proof.get('helper_sha256')==hashlib.sha256(helper.read_bytes()).hexdigest())
