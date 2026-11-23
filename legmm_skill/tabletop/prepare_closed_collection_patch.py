"""Create an unapplied explicit closed-cohort recovery patch; no live edits."""
import ast,difflib,hashlib,json
from pathlib import Path
from .prepare_v7_timing_patch import revised_sources


def recovery_source(source):
    s=source
    s=s.replace('def restore_latest(session,path,allow_partial=False,expected_sha256=None,expected_formal=None,updates_per_round=2000):','def restore_latest(session,path,allow_partial=False,expected_sha256=None,expected_formal=None,updates_per_round=2000,allow_closed_collection=False,expected_num_envs=64):',1)
    s=s.replace('    if allow_partial and expected_sha256 is None:', '    if (allow_partial or allow_closed_collection) and expected_sha256 is None:',1)
    s=s.replace("if not payload.get('complete_round_boundary') and not allow_partial:","if not payload.get('complete_round_boundary') and not (allow_partial or allow_closed_collection):",1)
    needle="    if not payload.get('complete_round_boundary'):partial_update_plan(payload,updates_per_round)"
    replacement="""    if not payload.get('complete_round_boundary'):
        if payload['cursor'].get('partial_phase')=='dagger_update':
            if not allow_partial:raise ValueError('Explicit partial update flag required')
            partial_update_plan(payload,updates_per_round)
        elif allow_closed_collection:
            from .closed_collection_recovery import closed_collection_plan
            payload['cursor']['closed_collection_plan']=closed_collection_plan(payload,updates_per_round,expected_num_envs)
        else:partial_update_plan(payload,updates_per_round)"""
    assert needle in s;s=s.replace(needle,replacement,1);ast.parse(s);return s


def runner_sources(root,timed=False):
    timing=revised_sources(root) if timed else {};out={}
    for name in ('train_dagger.py','learning_launch.py'):
        old=timing[name][1] if timed else (root/name).read_text();s=old
        needle="    p.add_argument('--resume-sha256')";s=s.replace(needle,needle+"\n    p.add_argument('--allow-closed-collection-recovery',action='store_true')",1)
        if name=='train_dagger.py':
            needle='expected_sha256=a.resume_sha256,expected_formal=status[\'formal_training\'],updates_per_round=a.updates);resume_cursor=cursor'
            replacement="expected_sha256=a.resume_sha256,expected_formal=status['formal_training'],updates_per_round=a.updates,allow_closed_collection=a.allow_closed_collection_recovery,expected_num_envs=a.num_envs);resume_cursor=cursor"
            assert needle in s;s=s.replace(needle,replacement,1)
            needle='        start_round=cursor[\'next_round\'];seed=cursor[\'next_seed\'];episode=cursor[\'episode\'];pending_validation=cursor[\'validation_pending\']'
            s=s.replace(needle,needle+"\n        if cursor.get('closed_collection_plan'):status.update(closed_collection_boundary_recovered=True,closed_collection_recovery_plan=cursor['closed_collection_plan'])",1)
        else:
            needle="    if a.allow_partial_recovery:command.append('--allow-partial-recovery')"
            assert needle in s;s=s.replace(needle,needle+"\n    if a.allow_closed_collection_recovery:command.append('--allow-closed-collection-recovery')",1)
        ast.parse(s);out[name]=(old,s)
    return out


def main():
    root=Path(__file__).parent;record=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_true4_contact_retry1_20261115');source=(root/'dagger_recovery.py').read_text();patched=recovery_source(source);info={}
    for timed in (False,True):
        changes={'dagger_recovery.py':(source,patched),**runner_sources(root,timed)}
        name='v8_closed_collection_after_v7' if timed else 'v8_closed_collection_standalone';path=record/(name+'_unapplied.patch')
        patch=''.join(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/legmm_skill/tabletop/'+file,tofile='b/legmm_skill/tabletop/'+file)) for file,(old,new) in changes.items());path.write_text(patch)
        info[name]=dict(path=str(path),files={file:dict(before=hashlib.sha256(old.encode()).hexdigest(),prepared=hashlib.sha256(new.encode()).hexdigest()) for file,(old,new) in changes.items()})
    (record/'v8_closed_collection_source.json').write_text(json.dumps(dict(applied=False,supported_only='fully closed student cohort; no interrupted collection',candidates=info),indent=2)+'\n')


if __name__=='__main__':main()
