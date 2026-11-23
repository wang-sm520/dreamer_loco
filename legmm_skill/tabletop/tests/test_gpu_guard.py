from tabletop.gpu_guard import memory_bounds


def test_scene_release_race_does_not_become_fake_foreign_memory():
    rows=[dict(pid=1,memory_mib=5000),dict(pid=2,memory_mib=1000)]
    result=memory_bounds(10000,6000,rows,1,0,get_group=lambda pid:pid)
    assert result['external_upper_bound_mib']==1000
    assert result['owned_upper_bound_mib']==9000
    assert 10000-5000>4096 # the old asynchronous subtraction falsely stopped


def test_real_foreign_and_own_overages_remain_visible():
    result=memory_bounds(12000,12000,[dict(pid=1,memory_mib=5000),dict(pid=2,memory_mib=5000)],1,0,get_group=lambda pid:pid)
    assert result['external_upper_bound_mib']>4096
    own=memory_bounds(21000,21000,[dict(pid=1,memory_mib=19500),dict(pid=2,memory_mib=1000)],1,0,get_group=lambda pid:pid)
    assert own['owned_upper_bound_mib']>18*1024
