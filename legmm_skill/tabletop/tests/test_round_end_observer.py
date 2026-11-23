import copy
from tabletop.round_end_observer import eligible


def valid():
    return dict(format='w8_dagger_full_round_v1',complete_round_boundary=True,status=dict(formal_training=True,diagnostic_only=False,num_envs=64),collection=dict(seen=64000,budget=64000,physical_env_decisions=64000,round=1,teacher_drives=False,collected_samples=60000),student=dict(updates=4000),cursor=dict(next_round=2,next_seed=1024,episode=12,partial_phase=None,validation_pending=None),reservoir=dict(capacity=10000,items=[None]*10000,seen=60000,composition={'all':60000}),rng=dict(python=1,numpy=1,torch=1,cuda=1))


def test_only_exact_complete_formal_round_target_is_eligible():
    p=valid();assert eligible(p)
    for key,value in [('complete_round_boundary',False),('format','w8_dagger_partial_v1')]:
        q=copy.deepcopy(p);q[key]=value;assert not eligible(q)
    for section,key,value in [('student','updates',3999),('student','updates',4001),('collection','seen',63936),('collection','seen',64064),('collection','round',2),('collection','teacher_drives',True),('status','diagnostic_only',True),('status','formal_training',False),('status','num_envs',128),('cursor','next_round',1),('cursor','next_seed',1088),('cursor','episode',13),('cursor','validation_pending',3),('cursor','partial_phase','dagger_update'),('reservoir','seen',59999)]:
        q=copy.deepcopy(p);q[section][key]=value;assert not eligible(q),(section,key,value)


def test_pidfd_pins_current_process_without_interrupting_it():
    import os
    from tabletop.round_end_observer import pidfd_open,pidfd_signal
    fd=pidfd_open(os.getpid())
    try:pidfd_signal(fd,0) # existence/permission probe only, no delivered signal
    finally:os.close(fd)
