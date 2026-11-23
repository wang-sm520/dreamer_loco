"""Strict authorized stage: learner continuation after a witnessed RAM-aborted cohort.

This never restores the interrupted physical episode, calls it closed, or
replays its initial states. Only a separately authorized explicit loader flag
may retain its consumed prefix and continue at the already advanced next seed.
"""
import gzip,hashlib,json
from pathlib import Path
from .spec import OUTPUT


def resource_abort_plan(payload,updates_per_round=2000,expected_num_envs=64,source_sha256=None):
    if payload.get('format')!='w8_dagger_partial_v1' or payload.get('complete_round_boundary'):raise ValueError('Explicit interrupted collection checkpoint required')
    c=payload['collection'];u=payload['cursor'];s=payload['status'];b=payload['reservoir'];r=c['round'];n=s['num_envs']
    if u.get('partial_phase')!='dagger_student_collection' or u.get('validation_pending') is not None:raise ValueError('Only student collection without pending validation supported')
    if n!=expected_num_envs or n!=64 or r<1 or u['next_round']!=r or c['teacher_drives']:raise ValueError('Unchanged64 student cohort required')
    if c['budget']!=(r+1)*32000 or not r*32000<c['seen']<c['budget']:raise ValueError('Partial quota differs')
    if payload['student']['updates']!=r*updates_per_round or u.get('updates_completed_in_round')!=0 or u.get('loss_sum',0.)!=0. or s.get('current_round_update_count')!=0:raise ValueError('Collection has optimizer work')
    if c['physical_env_decisions']!=c['seen'] or b['seen']!=c['collected_samples'] or sum(b['composition'].values())!=b['seen'] or not 0<len(b['items'])<=b['capacity']:raise ValueError('Incomplete logical learner counters/buffer')
    previous=s.get('latest_save',{})
    prior_plan=None;prior_source=None;boundary_seen=r*32000
    if previous:
        if not previous.get('complete_round_boundary') or previous.get('env_decisions')!=r*32000 or previous.get('optimizer_updates')!=r*updates_per_round or previous.get('rounds_completed')!=r:raise ValueError('Proven preceding complete round required')
    else:
        # A resumed attempt clears latest_save until its next save. Verify the
        # actual immutable predecessor, including its entire interruption chain.
        origin=s.get('source_recovery',{})
        prior_source=Path(origin.get('path',''))
        if not prior_source.is_absolute() or prior_source.is_symlink() or not prior_source.resolve().is_relative_to(OUTPUT) or not prior_source.is_file():raise ValueError('Owned immutable resource predecessor required')
        if not isinstance(origin.get('sha256'),str) or len(origin['sha256'])!=64:raise ValueError('Predecessor SHA required')
        available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
        if available<20*1024**3:raise MemoryError('Historical learner audit needs original20GiB available margin')
        import torch
        from .state_fingerprints import digest
        h=hashlib.sha256()
        with prior_source.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
            if h.hexdigest()!=origin['sha256']:raise ValueError('Predecessor SHA differs')
            f.seek(0);signature=f.read(2);f.seek(0)
            reader=gzip.GzipFile(fileobj=f,mode='rb') if signature==b'\x1f\x8b' else f
            try:prior_payload=torch.load(reader,map_location='cpu',weights_only=False)
            finally:
                if reader is not f:reader.close()
        prior_plan=resource_abort_plan(prior_payload,updates_per_round,expected_num_envs,source_sha256=origin['sha256'])
        prior_cursor=dict(prior_payload['cursor'],resource_abort_plan=prior_plan)
        prior_collection=prior_payload['collection'];prior_buffer=prior_payload['reservoir']
        if origin.get('cursor')!=prior_cursor or origin.get('env_decisions')!=prior_collection['seen'] or origin.get('optimizer_updates')!=prior_payload['student']['updates']:raise ValueError('Actual predecessor cursor/counters differ')
        if prior_collection['round']!=r or prior_collection['budget']!=c['budget'] or not r*32000<prior_collection['seen']<c['seen']:raise ValueError('Resource continuation round/quota differs')
        if prior_payload['status'].get('formal_training')!=s.get('formal_training') or prior_payload['visual']!=payload['visual'] or prior_payload['student']['model_config']!=payload['student']['model_config']:raise ValueError('Predecessor formal/model/visual identity differs')
        for key in ('model','optimizer','mean','scale','updates'):
            if digest(prior_payload['student'][key])!=digest(payload['student'][key]):raise ValueError('Student changed during collection: '+key)
        if prior_buffer['capacity']!=b['capacity'] or prior_buffer['seed']!=b['seed'] or not prior_buffer['seen']<=b['seen']:raise ValueError('Predecessor reservoir contract differs')
        boundary_seen=prior_collection['seen'];origin_samples=prior_collection['collected_samples']
        del prior_payload,prior_buffer,prior_collection

    episode=Path(s['current_episode']);root=episode.parent
    cohort_start_seen=boundary_seen;prefix=None
    if prior_plan is not None:
        from .resource_closed_prefix import closed_prefix_plan
        prefix=closed_prefix_plan(payload,prior_cursor,boundary_seen,origin_samples,updates_per_round)
        cohort_start_seen=prefix['seen']
    else:
        rows=[json.loads(line) for line in (episode/'learning_decisions.jsonl').read_text().splitlines()]
        if rows and rows[0]['env_decisions']-n!=boundary_seen:
            from .first_abort_prefix import first_abort_prefix_plan
            prefix=first_abort_prefix_plan(payload,previous,updates_per_round)
            cohort_start_seen=prefix['seen']

    if episode.is_symlink() or not episode.resolve().is_relative_to(OUTPUT):raise ValueError('Owned W8 episode required')
    # The runner deliberately keeps its rolling recovery root across attempts.
    # Bind that path to the preceding full-save record and the stopped attempt's
    # final save receipt; never guess another directory or copy old witnesses.
    stopped_receipt=json.loads((root/'training_status.json').read_text()) if prior_plan is not None and (root/'training_status.json').is_file() else {}
    save_path=Path(previous.get('path') or stopped_receipt.get('latest_save',{}).get('path',root/'latest.pt'))
    if not save_path.is_absolute() or save_path.name!='latest.pt' or save_path.is_symlink() or not save_path.resolve().is_relative_to(OUTPUT):raise ValueError('Owned rolling checkpoint path required')
    save_log=save_path.parent/'latest_save_memory.jsonl'
    if save_log.is_symlink() or not save_log.resolve().is_relative_to(OUTPUT):raise ValueError('Owned rolling save witness required')
    names=['result.json','collection.json','post_stage_memory_release.json','learning_decisions.jsonl']
    if any(not (episode/k).is_file() for k in names) or not (root/'status.json').is_file() or not save_log.is_file():raise ValueError('Missing interrupted/resource witnesses')
    parent=json.loads((root/'status.json').read_text());x=json.loads((episode/'result.json').read_text());record=json.loads((episode/'collection.json').read_text());requested=parent.get('safe_stop_requested',{})
    reason=s.get('reason','')
    if 'Stop signal 15' not in reason or 'collection action-block boundary after scene cleanup' not in reason or parent.get('reason')!=reason:raise ValueError('Not a witnessed graceful post-cleanup SIGTERM boundary')
    if parent.get('PID')!=s.get('PID') or parent.get('num_envs')!=n or parent.get('completed') or parent.get('exit_code') is None:raise ValueError('Parent identity/lifecycle witness differs')
    if not 8*1024**3<=requested.get('available_ram_bytes',0)<12*1024**3 or parent.get('minimum_available_ram_bytes',0)<8*1024**3:raise ValueError('Only RAM soft-guard case, not hard-kill/other cause supported')
    if not x.get('requested_safe_stop') or not record.get('interrupted') or x.get('safety_stop') is not None:raise ValueError('Not an explicit resource interruption; standing failure requires another decision')
    if x['num_envs']!=n or len(x['episodes'])!=n or sorted(z['seed'] for z in x['episodes'])!=list(range(x['seed'],x['seed']+n)):raise ValueError('Attempted initial-state identities differ')
    if not any(z.get('reason') is None for z in x['episodes']):raise ValueError('This proposal targets unresolved RAM-aborted cohorts only')
    if record['env_decisions']!=c['seen'] or record['physical_env_decisions']!=c['seen'] or record['collected_samples']!=c['collected_samples'] or record['round']!=r:raise ValueError('Collection witness counters differ')
    steps=x['physical_steps'];controlled=steps-300
    if not 0<controlled<1500 or controlled%16 or x['engine_controlled_steps']!=steps or x['native_delay_queue_advances']!=steps or x['actual_standing_policy_calls']!=(steps+1)//2:raise ValueError('Not a completed action-block prefix with audited engine/queue/standing counts')
    rows=[json.loads(line) for line in (episode/'learning_decisions.jsonl').read_text().splitlines()]
    if [z['frame'] for z in rows]!=list(range(0,controlled,16)) or any(z['actual_envs']!=n or z['teacher_drives'] for z in rows):raise ValueError('Decision frame/count/driver ledger differs')
    before=rows[0]['env_decisions']-n;consumed=len(rows)*n
    if before!=cohort_start_seen or c['seen']!=before+consumed or [z['env_decisions'] for z in rows]!=list(range(before+n,c['seen']+1,n)):raise ValueError('No replay/ledger gap after the proven predecessor')
    if prefix is not None and (x['seed']!=prefix['seed'] or int(episode.name.rsplit('_',1)[1])!=prefix['episode']):raise ValueError('Predecessor next episode/seed differs')
    if prefix and prefix['cohorts']:
        from .resource_closed_prefix import sample_ledger
        sample_ledger(rows,prefix['samples'],c['collected_samples'],n)
    seed=x['seed']+n
    if seed+n>10000:seed=256
    if u['next_seed']!=seed or u['episode']!=c['episode_serial'] or u['episode']!=int(episode.name.rsplit('_',1)[1])+1:raise ValueError('Advanced seed/episode cursor differs')
    log_lines=save_log.read_bytes().splitlines(keepends=True);saves=[json.loads(line) for line in log_lines]
    if source_sha256 is not None:
        matches=[i for i,z in enumerate(saves) if z.get('sha256')==source_sha256]
        if len(matches)!=1:raise ValueError('Unique source save provenance required')
        end=matches[0]+1;log_lines=log_lines[:end];saves=saves[:end]
    log_prefix=b''.join(log_lines)
    if not saves or any(z.get('path')!=str(save_path) for z in saves):raise ValueError('Rolling save path ledger differs')
    if prior_plan is None:
        if not any(z.get('sha256')==previous.get('sha256') and z.get('complete_round_boundary') and z.get('env_decisions')==boundary_seen for z in saves):raise ValueError('Missing preceding full-source provenance')
    elif not any(z.get('sha256')==s['source_recovery']['sha256'] and not z.get('complete_round_boundary') and z.get('env_decisions')==boundary_seen and z.get('optimizer_updates')==payload['student']['updates'] for z in saves[:-1]):raise ValueError('Missing actual predecessor save provenance')
    if saves[-1].get('env_decisions')!=c['seen'] or saves[-1].get('optimizer_updates')!=payload['student']['updates'] or saves[-1].get('complete_round_boundary'):raise ValueError('Saved partial-source record differs')
    split_receipt=None
    if save_path.parent.resolve()!=root.resolve():
        split_receipt=root/'training_status.json'
        if split_receipt.is_symlink() or not split_receipt.is_file():raise ValueError('Missing stopped-attempt rolling save receipt')
        final=json.loads(split_receipt.read_text())
        if final.get('PID')!=s.get('PID') or final.get('latest_save')!=saves[-1]:raise ValueError('Stopped-attempt rolling save receipt differs')
    if s.get('formal_training'):
        from .physics_config import verify_task_physics_audit
        verify_task_physics_audit(json.loads((episode/'physics_audit.json').read_text()));names.append('physics_audit.json')
        cmd=Path('/proc')/str(s['PID'])/'cmdline'
        if cmd.exists() and b'tabletop.train_dagger' in cmd.read_bytes():raise ValueError('Original training worker still alive')
    evidence={str(episode/k):hashlib.sha256((episode/k).read_bytes()).hexdigest() for k in names}
    evidence[str(root/'status.json')]=hashlib.sha256((root/'status.json').read_bytes()).hexdigest()
    evidence[str(save_log)]=hashlib.sha256(log_prefix).hexdigest()
    if split_receipt is not None:evidence[str(split_receipt)]=hashlib.sha256(split_receipt.read_bytes()).hexdigest()
    result=dict(phase='resource_aborted_student_cohort_learner_only',round=r,consumed_prefix_decisions=consumed,collected_in_round=c['seen']-r*32000,remaining_collection_decisions=c['budget']-c['seen'],remaining_round_updates=updates_per_round,next_seed=seed,next_episode=u['episode'],physical_episode_resumed=False,aborted_initial_states_replayed=False,aborted_initial_seeds=list(range(x['seed'],x['seed']+n)),unresolved_initial_seeds=[z['seed'] for z in x['episodes'] if z.get('reason') is None],previous_terminal_outcomes_preserved=True,all_existing_buffer_and_RNG_retained=True,evidence_sha256=evidence)

    if prior_plan is not None:
        result.update(resource_abort_chain_depth=prior_plan.get('resource_abort_chain_depth',0)+1,predecessor_source=str(prior_source),predecessor_sha256=s['source_recovery']['sha256'],predecessor_consumed_decisions=boundary_seen,predecessor_plan=prior_plan)
    if prefix and prefix['cohorts']:
        result['closed_cohorts_between_aborts' if prior_plan is not None else 'closed_cohorts_after_full_round']=prefix['cohorts']
        if prior_plan is None:result['full_boundary_anchor']=prefix['full_boundary_anchor']
        result['closed_cohort_consumed_decisions']=prefix['consumed']
        result['evidence_sha256'].update(prefix['evidence_sha256'])
    return result
