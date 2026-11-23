"""rsl_rl PPO for the residual actor: settings (user-confirmed 10-03), time-based discounting (user 10-12) and loading
the deterministic actor."""

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

TRAIN_CFG = {
    "num_steps_per_env": 32, "save_interval": 25, "experiment_name": "airbot_can_residual", "run_name": "",
    "logger": "tensorboard", "clip_actions": None, "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
    "actor": {"class_name": "MLPModel", "hidden_dims": [256, 256], "activation": "elu", "obs_normalization": True,
              # user 10-14 (was 0.5)
              "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.2, "std_type": "scalar"}},
    "critic": {"class_name": "MLPModel", "hidden_dims": [256, 256], "activation": "elu", "obs_normalization": True},
    # gamma is per nominal decision, EXECUTE waypoints (0.16 s); TimePPO discounts by the time a decision really takes.
    "algorithm": {"class_name": "airbot_rl.ppo:TimePPO", "num_learning_epochs": 5, "num_mini_batches": 4,
                  "learning_rate": 3e-4, "schedule": "adaptive", "desired_kl": 0.01, "gamma": 0.99, "lam": 0.95,
                  "entropy_coef": 0.0, "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
                  "max_grad_norm": 1.0, "rnd_cfg": None, "symmetry_cfg": None},
}


class TimePPO(PPO):
    """PPO with a per-transition discount gamma ** extras["duration"] (the decision's length in nominal decisions).
    Per-decision discounting let PPO skip discount steps by stretching each decision (10-12: 0.36 -> 0.76 s)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discounts = torch.ones_like(self.storage.rewards)

    def process_env_step(self, obs, rewards, dones, extras):
        discount = self.gamma ** extras["duration"].to(self.device)
        self.discounts[self.storage.step] = discount.view(-1, 1)
        extras = dict(extras)
        time_outs = extras.pop("time_outs", None)  # bootstrapped here, with this transition's discount
        if time_outs is not None:
            rewards = rewards + discount * self.transition.values.squeeze(1) * time_outs.to(self.device)
        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs):
        """rsl_rl's GAE, with self.discounts[step] in place of gamma."""
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            discount = self.discounts[step]
            delta = st.rewards[step] + next_is_not_terminal * discount * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * discount * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)


def load_actor(path, env, obs):
    """rsl_rl model_N.pt -> the actor in eval mode; calling it on an obs TensorDict gives the deterministic mean."""
    kwargs = {key: value for key, value in TRAIN_CFG["actor"].items() if key != "class_name"}
    actor = MLPModel(TensorDict(obs, batch_size=[env.num_envs]), TRAIN_CFG["obs_groups"], "actor",
                     env.cfg.action_space, **kwargs).to(env.device)
    actor.load_state_dict(torch.load(path, map_location=env.device, weights_only=False)["actor_state_dict"])
    return actor.eval()
