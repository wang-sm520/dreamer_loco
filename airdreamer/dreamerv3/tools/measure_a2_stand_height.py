"""Stand the trained A2JoystickFlat policy with a zero velocity command and
measure the base_link height. Run from /Users/amiao/UniLab: uv run python <this file> 16 out.json"""
import json
import sys

import numpy as np
import onnxruntime as ort

import unilab.envs.locomotion.a2  # noqa: F401  (registers A2JoystickFlat)
from unilab.base import registry

RUN = '/Users/amiao/UniLab/logs/rsl_rl_ppo/A2JoystickFlat/2026-08-20_17-29-21_mujoco'
N_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SECONDS = 10.0

import yaml
task_yaml = yaml.safe_load(open('/Users/amiao/UniLab/conf/ppo/task/a2_joystick_flat/mujoco.yaml'))
override = {
    'reward_config': task_yaml['reward'],
    'commands': {'vel_limit': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 'velocity_curriculum': {'enabled': False}},
    'domain_rand': {'push_robots': False, 'randomize_base_mass': False, 'random_com': False},
}
env = registry.make('A2JoystickFlat', num_envs=N_ENVS, sim_backend='mujoco', env_cfg_override=override)
cfg = env._cfg
sess = ort.InferenceSession(f'{RUN}/policy.onnx')
inp = sess.get_inputs()[0]
print('onnx input', inp.name, inp.shape, 'ctrl_dt', cfg.ctrl_dt)
act_dim = sess.get_outputs()[0].shape[-1]
state = env.step(np.zeros((N_ENVS, act_dim), np.float32))
steps = int(SECONDS / cfg.ctrl_dt)
heights, cmds, falls = [], [], 0
for t in range(steps):
  state.info['commands'] = np.zeros_like(state.info['commands'])
  obs = state.obs['obs'].astype(np.float32)
  act = np.concatenate([sess.run(None, {inp.name: o[None]})[0] for o in obs], 0)
  state = env.step(act)
  falls += int(np.sum(state.terminated))
  heights.append(np.asarray(env._backend.get_base_pos())[:, 2].copy())
  cmds.append(np.abs(state.info['commands']).max())
h = np.array(heights)
settle = int(2.0 / cfg.ctrl_dt)
hs = h[settle:]
res = dict(n_envs=N_ENVS, seconds=SECONDS, ctrl_dt=float(cfg.ctrl_dt), max_abs_command=float(np.max(cmds)),
           falls=falls, base_height_mean=float(hs.mean()), base_height_std_over_time=float(hs.std(0).mean()),
           base_height_env_min=float(hs.mean(0).min()), base_height_env_max=float(hs.mean(0).max()),
           base_height_first_step=float(h[0].mean()),
           reward_base_height_target=float(env._reward_cfg.base_height_target))
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else 'a2_stand_height.json', 'w'), indent=1)
