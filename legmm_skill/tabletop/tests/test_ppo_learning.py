import numpy as np
import pytest
pytest.importorskip('rsl_rl')
from tabletop.ppo_learning import transition_reward
from tabletop.outcome import EpisodeOutcome


def test_reward_uses_actual_clock_terminal_potential_and_collision_priority():
    outcomes=[EpisodeOutcome(.6,steps=16,peak_lift=.025),
        EpisodeOutcome(.6,steps=7,peak_lift=.08,held_steps=200,reason='success',success=True),
        EpisodeOutcome(.6,steps=3,peak_lift=.08,held_steps=199,reason='collision:palm:object')]
    rewards,duration,done,potential,steps=transition_reward(np.array([.1,.8,.9]),np.zeros(3),outcomes)
    assert np.allclose(duration,[1,7/16,3/16])
    assert np.array_equal(done,[False,True,True])
    assert np.allclose(potential,[.5,0,0])
    assert rewards[0]==pytest.approx(.99*.5-.1-.001)
    assert rewards[1]==pytest.approx(20-.8-.001*7/16)
    assert rewards[2]==pytest.approx(-7-.9-.001*3/16)
    with pytest.raises(ValueError):transition_reward(np.zeros(3),np.array([17,0,0]),outcomes)


def test_session_real_optimizer_across_decisions_and_terminal_boundary(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    import tabletop.ppo_learning as module
    torch.set_num_threads(2)
    class Features:
        def __init__(self,student):pass
        def observe(self,images,proprio,last):
            return dict(policy=torch.as_tensor(proprio,dtype=torch.float32)),last[:,None].repeat(1,4,1)
    monkeypatch.setattr(module,'FrozenStudentFeatures',Features)
    visual=SimpleNamespace(manifest=lambda:{},observe=lambda scene:(None,None,None))
    session=module.ResidualSession(SimpleNamespace(device=torch.device('cpu')),visual)
    session.critic=lambda scene,robot:torch.zeros(32,5)
    session.reset([None]*32,None,None,tmp_path)
    outcomes=[EpisodeOutcome(.6) for _ in range(32)]
    joints=np.zeros((32,8),dtype=np.float32);last=torch.zeros(32,7)
    for frame in range(34):
        result=session.decision(None,None,joints,list(range(6)),[6,7],last,None,outcomes,frame*16)
        assert result.shape==(32,8,7)
        for o in outcomes:o.steps+=16
    for o in outcomes:o.reason='timeout'
    session.latest_critic=torch.zeros(32,5)
    session.finish(outcomes)
    assert session.updates==1 and session.valid_samples==34*32
    assert session.pending is None
    assert session.algorithm.storage.valid.sum()==64
    stopped=tmp_path/'stopped';stopped.mkdir()
    session.reset([None]*32,None,None,stopped);session.pending=torch.ones(32,dtype=torch.bool)
    before=session.valid_samples
    session.finish([EpisodeOutcome(.6) for _ in range(32)],aborted='base_fall')
    assert session.pending is None and session.valid_samples==before and session.updates==1
