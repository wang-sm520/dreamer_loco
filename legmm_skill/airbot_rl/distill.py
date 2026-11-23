"""DAgger: distil the privileged scripted grasp into the depth BC. Every round resets all envs and runs one episode
each (round 0: the teacher drives, later rounds: the student), storing what the student saw together with the
teacher's next targets; the student then trains on all rounds so far. A round's success is its driver's own, so
round k > 0 evaluates the student trained through round k - 1."""

import argparse
import json
import subprocess
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="BC checkpoint: student weights, statistics and DA2 config")
parser.add_argument("--random_init", action="store_true", help="keep only the architecture, statistics and DA2 config")
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--rounds", type=int, default=6)
parser.add_argument("--steps", type=int, default=2000, help="gradient steps per round")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--condition", choices=("clean", "noise1", "random"), default="clean")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from airbot_depth.model import AirbotPolicyModel, batch_loss  # noqa: E402
from airbot_depth.train import save_checkpoint, to_device  # noqa: E402
from airbot_rl.env import CONDITIONS, CanEnv, CanEnvCfg  # noqa: E402
from airbot_rl.scene import EXECUTE  # noqa: E402


def collect(env, teacher_drives):
    """One episode per env -> (samples {images, state, action} on the CPU in the BC's normalised units, stats).
    A fallen can is skipped: the top-down teacher has no sensible targets for it."""
    bc = env.base
    env.reset()
    stats = {"noise_strength": env.noise_strength.mean().item(), "head_gain": env.head_gain.mean().item(),
             "command_delay_steps": env.delay_steps.float().mean().item()}
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    success, held, fallen = torch.zeros_like(active), torch.zeros_like(active), torch.zeros_like(active)
    samples = {"images": [], "state": [], "action": []}
    zero = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
    while active.any():
        record = active & ~env.fallen
        samples["images"].append(bc.last_depth[record].half().cpu())
        samples["state"].append(((env.native_state() - bc.state_mean) / bc.state_std)[record].cpu())
        samples["action"].append(((env.teacher_chunk - bc.action_mean) / bc.action_std)[record].cpu())
        if teacher_drives:
            env.base_chunk = env.teacher_chunk[:, :EXECUTE]
        _, _, done, timeout, _ = env.step(zero)
        success |= done & active
        held |= env.held & active
        fallen |= env.fallen & active
        active &= ~(done | timeout)
    stats.update({name: flag.float().mean().item() for name, flag in
                  (("success", success), ("held", held), ("fallen", fallen))})
    return {key: torch.cat(value) for key, value in samples.items()}, stats


def train(bc, rounds, optimizer, generator):
    """args.steps AdamW steps on samples drawn uniformly over all rounds (kept apart: concatenating would briefly
    double the images in host RAM); returns the mean loss."""
    model = bc.model.train().requires_grad_(True)
    sizes = torch.tensor([len(r["state"]) for r in rounds])
    ends = sizes.cumsum(0)
    tokens, mask = bc.tokens.expand(args.batch, -1), torch.ones(args.batch, model.horizon, dtype=torch.bool)
    total = 0.0
    for _ in range(args.steps):
        index = torch.randint(int(ends[-1]), (args.batch,), generator=generator)
        which = torch.searchsorted(ends, index, right=True)
        picks = list(zip(which.tolist(), (index - ends[which] + sizes[which]).tolist()))
        batch = {key: torch.stack([rounds[r][key][i] for r, i in picks]) for key in ("images", "state", "action")}
        loss = batch_loss(model, to_device(batch | {"mask": mask}, bc.tokens.device) | {"tokens": tokens})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    model.eval().requires_grad_(False)
    return total / args.steps


def main():
    cfg = CanEnvCfg(**CONDITIONS[args.condition])
    cfg.seed, cfg.scene.num_envs, cfg.teacher = args.seed, args.num_envs, True
    cfg.checkpoint = str(Path(args.checkpoint).resolve())
    env = CanEnv(cfg)
    model = env.base.model
    if args.random_init:
        torch.manual_seed(args.seed)
        model.load_state_dict(AirbotPolicyModel(**model.config).state_dict())
    base = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    rounds = []
    with (out / "metrics.jsonl").open("w") as log:
        for round_ in range(args.rounds + 1):
            memory = subprocess.check_output(["free", "-g"], text=True)
            print(memory, end="", flush=True)
            if int(next(line for line in memory.splitlines() if line.startswith("Mem:")).split()[-1]) < 8:
                raise MemoryError("available memory below 8 GB")
            samples, stats = collect(env, teacher_drives=round_ == 0)
            rounds.append(samples)
            record = {"round": round_, "driver": "teacher" if round_ == 0 else "student", **stats,
                      "new_samples": len(samples["state"]), "samples": sum(len(r["state"]) for r in rounds)}
            if round_ < args.rounds:
                record["loss"] = train(env.base, rounds, optimizer, generator)
                save_checkpoint(out / f"round_{round_:02d}.pt",
                                {**base, "model": model.state_dict(), "round": round_, "distill": vars(args)})
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    SimulationContext.clear_instance()
    app.close(wait_for_replicator=False)
