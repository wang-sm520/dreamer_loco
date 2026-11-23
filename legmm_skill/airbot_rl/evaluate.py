"""Evaluate one first episode per env (envs = episodes): the frozen BC, optionally plus a deterministic residual PPO
actor, or the scripted teacher. --video N also writes an mp4 for each of the first N envs (side view, plus the BC's
depth inputs when the base is the BC)."""

import argparse
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", choices=("bc", "script"), default="bc")
parser.add_argument("--checkpoint", help="BC checkpoint; defaults to CanEnvCfg.checkpoint")
parser.add_argument("--ppo", help="rsl_rl model_N.pt; its deterministic mean is the residual")
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", type=int, default=0, help="record the first N envs")
parser.add_argument("--condition", choices=("clean", "noise1", "random"), default="clean")
parser.add_argument("--output", default="airbot_rl/eval/latest")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, args.base == "bc" or args.video > 0
app = AppLauncher(args).app

import av  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

from airbot_rl.env import CONDITIONS, CanEnv, CanEnvCfg  # noqa: E402
from airbot_rl.ppo import load_actor  # noqa: E402
from airbot_rl.scene import CONTROL_HZ  # noqa: E402

PITCH, YAW = math.radians(12), math.radians(90)  # side view: looking +y, tilted down
SIDE_ROT = (math.cos(YAW / 2) * math.cos(PITCH / 2), -math.sin(YAW / 2) * math.sin(PITCH / 2),
            math.cos(YAW / 2) * math.sin(PITCH / 2), math.sin(YAW / 2) * math.cos(PITCH / 2))


def label(image, text):
    cv2.putText(image, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def video_frame(env, i, side, text):
    """Side view of env i, with the BC's head/wrist inputs (cropped from the 128 x 128 letterbox) below it."""
    image = label(side[i].copy(), text)
    if env.cfg.base != "bc":
        return image
    depth = (env.base.last_depth[i, :, 0, 28:100].cpu().numpy() * 255).astype(np.uint8)
    bottom = np.concatenate([cv2.resize(view, (480, 270), interpolation=cv2.INTER_NEAREST) for view in depth], 1)
    return np.concatenate((image, np.repeat(bottom[..., None], 3, 2), np.zeros((2, 960, 3), np.uint8)), 0)


def main():
    cfg = CanEnvCfg(**CONDITIONS[args.condition])
    cfg.seed, cfg.scene.num_envs, cfg.base = args.seed, args.episodes, args.base
    if args.checkpoint:
        cfg.checkpoint = str(Path(args.checkpoint).resolve())
    if args.video:
        cfg.scene.side_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/side_cam", data_types=["rgb"], width=960, height=540,
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.35, -1.4, 0.55), rot=SIDE_ROT, convention="world"))
    env = CanEnv(cfg)
    obs, _ = env.reset()
    actor = load_actor(args.ppo, env, obs) if args.ppo else None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    videos = []
    for i in range(min(args.video, env.num_envs)):
        container = av.open(str(out / f"env{i}.mp4"), "w")
        stream = container.add_stream("h264", rate=6)
        stream.width, stream.height, stream.pix_fmt = 960, 812 if args.base == "bc" else 540, "yuv420p"
        videos.append((container, stream))

    n = env.num_envs
    flag = lambda: torch.zeros(n, dtype=torch.bool, device=env.device)  # noqa: E731
    active, dog, floor, toppled, held, near_seen = ~flag(), flag(), flag(), flag(), flag(), flag()
    gripped = flag()
    near_min = torch.full((n,), math.inf, device=env.device)
    excess, seconds, longest = (torch.zeros(n, device=env.device) for _ in range(3))
    length = torch.zeros(n, dtype=torch.long, device=env.device)
    records, decision = [None] * n, 0
    while active.any():
        running = active.clone()
        can, grasp = env.can_pos(), env.grasp_point()
        horizontal = (grasp[:, :2] - can[:, :2]).norm(dim=1)
        near = (horizontal >= 0.04) & (horizontal <= 0.07) & (env.can_lowest() < 0.01) & running
        near_min = torch.where(near, torch.minimum(near_min, grasp[:, 2]), near_min)
        near_seen |= near
        if videos:
            if args.base != "bc":
                env.sim.render()
            camera = env.scene["side_cam"]
            camera.update(0.0, force_recompute=True)
            side = camera.data.output["rgb"].cpu().numpy()
            for i, (container, stream) in enumerate(videos):
                if running[i]:
                    width = env.native_state()[i, 6].item()
                    frame = video_frame(env, i, side, f"decision {decision}  gripper {width * 100:.1f} cm")
                    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)

        with torch.inference_mode():
            action = actor(TensorDict(obs, batch_size=[n])) if actor else torch.zeros(n, cfg.action_space,
                                                                                         device=env.device)
        obs, _, success, timeout, extras = env.step(action)
        dog |= env.touched_dog & running
        floor |= env.touched_floor & running
        toppled |= env.episode_toppled & running
        held |= extras["held"] & running
        gripped |= extras["gripped"] & running
        excess += env.excess * running
        seconds += env.waypoint_count / CONTROL_HZ * running  # deployed execution time
        longest = torch.maximum(longest, env.waypoint_count / CONTROL_HZ * running)
        length += running
        for i in ((success | timeout) & running).nonzero().flatten().tolist():
            records[i] = {"success": bool(success[i]), "decisions": int(length[i]), "held": bool(held[i]),
                          "gripped": bool(gripped[i]),
                          "toppled": bool(toppled[i]), "contact_dog": bool(dog[i]), "contact_floor": bool(floor[i]),
                          "clip_excess": float(excess[i]), "seconds": float(seconds[i]),
                          "longest_decision_s": float(longest[i]),
                          "approach_height": float(near_min[i]) if near_seen[i] else None}
            active[i] = False
        decision += 1

    for container, stream in videos:
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    heights = [r["approach_height"] for r in records if r["approach_height"] is not None]
    summary = {key: float(np.mean([r[key] for r in records]))
               for key in ("success", "held", "gripped", "toppled", "contact_dog", "contact_floor", "clip_excess", "decisions",
                           "seconds", "longest_decision_s")}
    summary.update(condition=args.condition, episodes=len(records),
                   approach_height_median=float(np.median(heights)) if heights else None,
                   approach_no_entry_rate=1 - len(heights) / len(records))
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    (out / "summary.json").write_text(json.dumps({"summary": summary, "episodes": records}, indent=1))
    print(json.dumps(summary), flush=True)
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    sim_utils.SimulationContext.clear_instance()
    app.close(wait_for_replicator=False)
