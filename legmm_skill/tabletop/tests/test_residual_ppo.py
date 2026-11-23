import pytest
pytest.importorskip('rsl_rl')
import torch
from tensordict import TensorDict
from tabletop.residual_ppo import build_residual_ppo


def test_source_timeppo_update_masks_padding_and_actor_has_no_critic_input():
    torch.set_num_threads(2);torch.manual_seed(7)
    active=torch.arange(32)%2==0
    policy=torch.randn(32,11);critic=torch.randn(32,5)
    policy[~active]=10000.;critic[~active]=10000.
    obs=TensorDict(dict(policy=policy,critic=critic),batch_size=[32])
    alg=build_residual_ppo(obs)
    assert torch.equal(alg.actor(obs),torch.zeros(32,28))
    for _ in range(64):
        alg.set_active(active);alg.act(obs)
        reward=torch.ones(32);reward[~active]=10000.
        alg.process_env_step(obs,reward,~active,dict(duration=torch.full((32,),.5)))
    assert alg.storage.valid.sum()==1024
    assert torch.allclose(alg.discounts[0],torch.full((32,1),.99**.5))
    alg.compute_returns(obs)
    batches=list(alg.storage.mini_batch_generator(4,1))
    assert len(batches)==4 and all(x.observations.batch_size[0]==256 for x in batches)
    assert all(float(x.observations['policy'].abs().max())<100 for x in batches)
    alg.update()
    assert alg.storage.step==0
    changed=obs.clone();changed['critic']+=1000.
    assert torch.equal(alg.actor(obs),alg.actor(changed))
    assert not torch.equal(alg.actor(obs),torch.zeros(32,28))


def test_residual_zero_is_exact_base_and_motion_limits_remain_native():
    import numpy as np
    from tabletop.residual_ppo import residual_commands
    from tabletop.spec import STOW,STEP_LENGTH,LOWER,UPPER
    start=torch.tensor([[*STOW,.072]],dtype=torch.float32)
    base=start[:,None].repeat(1,4,1)
    assert torch.equal(residual_commands(base,torch.zeros(1,28),start),base)
    moved=residual_commands(base,torch.full((1,28),10.),start)
    delta=torch.diff(torch.cat((start[:,None],moved),1),dim=1).numpy()
    assert np.max(abs(delta)/STEP_LENGTH)<1.00001
    assert np.all(moved.numpy()>=LOWER-1e-6) and np.all(moved.numpy()<=UPPER+1e-6)
