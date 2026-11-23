"""Residual PPO (rsl_rl) on a base policy. The actor's mean layer starts at zero, so iteration 0 is the base exactly."""

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", choices=("bc", "script"), default="bc")
parser.add_argument("--checkpoint", help="frozen BC; defaults to CanEnvCfg.checkpoint")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--condition", choices=("clean", "noise1", "random"), default="clean")
parser.add_argument("--log_dir", default="airbot_rl/runs/ppo")
parser.add_argument("--resume", help="rsl_rl checkpoint (model_N.pt) to continue from")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, args.base == "bc"
app = AppLauncher(args).app

from datetime import datetime  # noqa: E402

import torch  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from airbot_rl.env import CONDITIONS, CanEnv, CanEnvCfg  # noqa: E402
from airbot_rl.ppo import TRAIN_CFG  # noqa: E402


def main():
    cfg = CanEnvCfg(**CONDITIONS[args.condition])
    cfg.seed, cfg.scene.num_envs, cfg.base = args.seed, args.num_envs, args.base
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint
    env = RslRlVecEnvWrapper(CanEnv(cfg))
    runner = OnPolicyRunner(env, {**TRAIN_CFG, "seed": args.seed, "max_iterations": args.iterations},
                            log_dir=f"{args.log_dir}/{datetime.now():%Y%m%d-%H%M%S}", device=env.device)
    if args.resume:
        runner.load(args.resume)  # restores actor, critic, normalisers, optimizer and the iteration counter
    else:
        mean = [m for m in runner.alg.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(mean.weight)
        torch.nn.init.zeros_(mean.bias)
    runner.learn(args.iterations, init_at_random_ep_len=True)
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    SimulationContext.clear_instance()
    app.close(wait_for_replicator=False)
