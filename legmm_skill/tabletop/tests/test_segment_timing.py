import ast,hashlib,json
from pathlib import Path
import numpy as np
import pytest
from tabletop.segment_timing import SegmentTimings,observer_source_compatible
from tabletop.prepare_v7_timing_patch import revised_sources,StripTiming,step_ast


def test_disabled_timer_has_no_clock_io_or_value_changes(tmp_path):
    def forbidden():raise AssertionError('Disabled telemetry read clock')
    timer=SegmentTimings(False,forbidden);data=np.arange(7.)
    assert timer.call('forward',lambda x:x,data) is data
    with timer.measure('physics'):assert np.array_equal(data,np.arange(7.))
    assert timer.snapshot()['segments']=={}
    assert not timer.write(tmp_path/'disabled.json') and not list(tmp_path.iterdir())


def test_enabled_timer_records_bounded_host_time_preserving_exception():
    calls=iter(range(100));timer=SegmentTimings(True,lambda:next(calls))
    assert timer.call('forward',lambda:42)==42
    with pytest.raises(ValueError,match='body error'):
        with timer.measure('contact'):raise ValueError('body error')
    assert timer.snapshot()['segments']['forward']['count']==1
    for i in range(14):timer.call(str(i),lambda:None)
    with pytest.raises(ValueError,match='Bounded'):timer.call('extra',lambda:None)
    assert len(timer.segments)==16


def test_prepared_patch_preserves_physics_observation_and_policy_ast():
    root=Path(__file__).parents[1]
    baseline=Path('/home/bdi/wang-sm/logdir/legmm/v4/w8_true4_contact_retry1_20261115/v9_before_foot_timing_integration') if 'learning.segment_timings=timings' in (root/'learning_episode.py').read_text() else root
    changes=revised_sources(baseline)
    for _,new in changes.values():ast.parse(new)
    assert step_ast(changes['learning_episode.py'][0])==step_ast(changes['learning_episode.py'][1],True)
    for name in ('learning.py','visual.py'):
        old,new=changes[name]
        assert ast.dump(ast.parse(old),include_attributes=False)==ast.dump(StripTiming().visit(ast.parse(new)),include_attributes=False)


def test_observer_source_gate_rejects_unread_updates_and_other_changes(tmp_path):
    root=Path(__file__).parents[1];proof=json.loads((root/'diagnostics/v7_segment_timing_proof.json').read_text())
    text='version7';(tmp_path/'COORDINATOR_UPDATES.md').write_text(text)
    (tmp_path/'coordinator_update_ack.json').write_text(json.dumps(dict(version=7,coordinator_updates_sha256=hashlib.sha256(text.encode()).hexdigest())))
    row=proof['files']['visual.py']
    assert observer_source_compatible('visual.py',row['before_sha256'],row['prepared_sha256'],tmp_path)
    assert not observer_source_compatible('visual.py',row['before_sha256'],'wrong',tmp_path)
    assert not observer_source_compatible('student.py','any','any',tmp_path)
    (tmp_path/'COORDINATOR_UPDATES.md').write_text(text+'new decision')
    assert not observer_source_compatible('visual.py',row['before_sha256'],row['prepared_sha256'],tmp_path)
