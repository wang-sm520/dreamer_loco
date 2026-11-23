"""Unenabled v8 proposal: recover only fully closed student-cohort boundaries.

This does not authorize or implement mid-episode recovery. It reads immutable
closure witnesses and rejects interrupted/unresolved cohorts and changed cursors.
"""
import hashlib,json
from pathlib import Path
from .spec import OUTPUT


def closed_collection_plan(payload,updates_per_round=2000,expected_num_envs=64):
    if payload.get('complete_round_boundary') or payload.get('format')!='w8_dagger_partial_v1':raise ValueError('Explicit partial collection state required')
    cursor=payload['cursor'];collection=payload['collection'];status=payload['status']
    if cursor.get('partial_phase')!='dagger_student_collection':raise ValueError('Unsupported closed collection phase')
    if cursor.get('validation_pending') is not None:raise ValueError('Pending validation cannot be skipped')
    round_index=collection['round'];seen=collection['seen'];budget=collection['budget'];n=status['num_envs']
    if n!=expected_num_envs or n!=64:raise ValueError('Closed collection requires unchanged64 environments')
    if round_index<1 or cursor['next_round']!=round_index or collection['teacher_drives']:raise ValueError('Wrong student collection round')
    if budget!=(round_index+1)*32000 or not round_index*32000<seen<budget:raise ValueError('Not an incomplete collection quota')
    if payload['student']['updates']!=round_index*updates_per_round:raise ValueError('Model counter differs from pre-update collection')
    if cursor.get('updates_completed_in_round')!=0 or cursor.get('loss_sum',0.)!=0.:raise ValueError('Collection has partial optimizer work')
    if not 0<len(payload['reservoir']['items'])<=payload['reservoir']['capacity']:raise ValueError('Missing bounded recovery buffer')
    if sum(payload['reservoir']['composition'].values())!=payload['reservoir']['seen']:raise ValueError('Reservoir composition differs')
    episode=Path(status['current_episode'])
    if episode.is_symlink() or not episode.resolve().is_relative_to(OUTPUT):raise ValueError('W8 owned closed episode required')
    required=['result.json','collection.json','post_stage_memory_release.json','learning_decisions.jsonl']
    if any(not (episode/name).is_file() for name in required):raise ValueError('Missing closed cohort witness')
    result=json.loads((episode/'result.json').read_text());record=json.loads((episode/'collection.json').read_text())
    if result.get('requested_safe_stop') or record.get('interrupted'):raise ValueError('Interrupted collection is not a closed-boundary recovery')
    if result.get('safety_stop') is not None:raise ValueError('Standing safety stop requires separate decision')
    if result['num_envs']!=n or len(result['episodes'])!=n or any(not x.get('reason') for x in result['episodes']):raise ValueError('Unresolved physical cohort')
    if sorted(x['seed'] for x in result['episodes'])!=list(range(result['seed'],result['seed']+n)):raise ValueError('Cohort seed identity differs')
    if record['env_decisions']!=seen or record['round']!=round_index:raise ValueError('Witness collection quota differs')
    if record['physical_env_decisions']!=collection['physical_env_decisions'] or record['collected_samples']!=collection['collected_samples']:raise ValueError('Witness physical/sample counters differ')
    steps=result['physical_steps']
    if result['engine_controlled_steps']!=steps or result['native_delay_queue_advances']!=steps or result['actual_standing_policy_calls']!=(steps+1)//2:raise ValueError('Engine/queue/control counts differ')
    if not 300<steps<=1800:raise ValueError('Unexpected completed cohort duration')
    decision_rows=[]
    with (episode/'learning_decisions.jsonl').open() as file:
        for line in file:decision_rows.append(json.loads(line))
    if [x['frame'] for x in decision_rows]!=list(range(0,steps-300,16)):raise ValueError('Decision frame ledger differs')
    if not decision_rows or decision_rows[-1]['env_decisions']!=seen:raise ValueError('Missing final decision cursor')
    if any(x['actual_envs']!=n or x['teacher_drives'] for x in decision_rows):raise ValueError('Wrong actual cohort driver/count')
    consumed=len(decision_rows)*n
    if decision_rows[-1]['env_decisions']-decision_rows[0]['env_decisions']+n!=consumed:raise ValueError('Decision ledger gap')
    if decision_rows[0]['env_decisions']-n<round_index*32000:raise ValueError('Cohort overlaps prior round')
    expected_seed=result['seed']+n
    if expected_seed+n>10000:expected_seed=256
    if cursor['next_seed']!=expected_seed or cursor['episode']!=collection['episode_serial']:raise ValueError('Next seed/episode cursor differs')
    if status.get('formal_training'):
        from .physics_config import verify_task_physics_audit
        verify_task_physics_audit(json.loads((episode/'physics_audit.json').read_text()));required.append('physics_audit.json')
    return dict(phase='closed_student_collection_boundary',round=round_index,collected_in_round=seen-round_index*32000,
        remaining_collection_decisions=budget-seen,remaining_round_updates=updates_per_round,next_seed=expected_seed,
        next_episode=cursor['episode'],physical_episode_resumed=False,cohort_env_decisions=consumed,
        evidence_sha256={name:hashlib.sha256((episode/name).read_bytes()).hexdigest() for name in required})
