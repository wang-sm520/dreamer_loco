"""Compute an agent "signature" (param count, loss keys) for the drone presets.

Used once before the legmm changes to write tests/legmm/drone_baseline.json and
afterwards by tests/legmm/test_backward_compat.py. No simulator is started: the
obs/act spaces are written out by hand from drone_navigation.py / drone_multi.py.
"""
import json
import os
import pathlib
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import elements
import numpy as np
import ruamel.yaml as yaml


def drone_spaces(preset):
  f32 = np.float32
  obs = {
      'state': elements.Space(f32, (15,)),
      'depth': elements.Space(f32, (48, 80, 1)),
      'reward': elements.Space(f32),
      'is_first': elements.Space(bool),
      'is_last': elements.Space(bool),
      'is_terminal': elements.Space(bool),
  }
  if preset == 'drone_multi':
    obs['direction'] = elements.Space(f32, (3,))
    obs['lidar'] = elements.Space(f32, (180, 20))
  act = {'action': elements.Space(f32, (5,), -1.0, 1.0)}
  return obs, act


def load_config(presets, overrides=()):
  configs = yaml.YAML(typ='safe').load(
      (ROOT / 'dreamerv3' / 'configs.yaml').read_text())
  config = elements.Config(configs['defaults'])
  for name in presets:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(list(overrides))
  return config


def signature(preset, platform='cpu', logdir='/tmp/legmm_sig'):
  from dreamerv3.agent import Agent
  config = load_config([preset], [
      f'--jax.platform={platform}', '--jax.policy_devices=0',
      '--jax.train_devices=0', '--jax.prealloc=False',
      f'--logdir={logdir}'])
  obs, act = drone_spaces(preset)
  agent_cfg = elements.Config(
      **config.agent, logdir=config.logdir, seed=config.seed,
      jax=dict(config.jax, precompile=False),
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica, replicas=config.replicas)
  agent = Agent(obs, act, agent_cfg)
  params = {k: int(np.prod(v.shape)) for k, v in agent.params.items()}
  return {
      'preset': preset,
      'n_params': int(sum(params.values())),
      'n_param_tensors': len(params),
      'param_keys_sha': _sha(sorted(params.items())),
      'loss_keys': sorted(agent.model.scales.keys()),
      'loss_scales': {k: float(v) for k, v in sorted(agent.model.scales.items())},
      'enc_keys': sorted(agent.model.enc.obs_space.keys()),
      'dec_keys': sorted(agent.model.dec.obs_space.keys()),
  }


def _sha(obj):
  import hashlib
  return hashlib.sha256(json.dumps(obj).encode()).hexdigest()


if __name__ == '__main__':
  out = {p: signature(p) for p in sys.argv[2:] or ['drone_nav', 'drone_multi']}
  pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
  print(json.dumps({k: v['n_params'] for k, v in out.items()}))
