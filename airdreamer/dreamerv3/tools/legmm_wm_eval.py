"""Offline world-model diagnostics for a LegMM checkpoint.

All neural predictions in this file come from ``Agent.model`` through ninjax
pure calls.  The numpy functions at the top are intentionally independent so
they can be unit-tested without JAX or sklearn.
"""

import argparse
import json
import pathlib
import sys

import elements
import jax
import ninjax as nj
import numpy as np


def binary_auroc(labels, scores, mask=None):
  labels, scores = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
  valid = np.isfinite(scores)
  if mask is not None:
    valid &= np.asarray(mask).reshape(-1).astype(bool)
  labels, scores = labels[valid].astype(bool), scores[valid]
  pos, neg = scores[labels], scores[~labels]
  if not len(pos) or not len(neg):
    return float('nan'), int(len(pos)), int(len(neg))
  # Mann-Whitney U with average ranks handles ties without sklearn.
  order = np.argsort(scores, kind='mergesort')
  ranks = np.empty(len(scores), np.float64)
  sorted_scores = scores[order]
  start = 0
  while start < len(scores):
    stop = start + 1
    while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
      stop += 1
    ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1
    start = stop
  u = ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2
  return float(u / (len(pos) * len(neg))), int(len(pos)), int(len(neg))


def ece_equal_width(labels, probabilities, mask=None, bins=10):
  labels, probabilities = np.asarray(labels).reshape(-1), np.asarray(probabilities).reshape(-1)
  valid = np.isfinite(probabilities)
  if mask is not None:
    valid &= np.asarray(mask).reshape(-1).astype(bool)
  labels, probabilities = labels[valid].astype(np.float64), probabilities[valid]
  edges = np.linspace(0.0, 1.0, bins + 1)
  total = len(labels)
  result = 0.0
  details = []
  for i in range(bins):
    selected = (probabilities >= edges[i]) & (
        probabilities <= edges[i + 1] if i == bins - 1 else probabilities < edges[i + 1])
    count = int(selected.sum())
    if count:
      acc = float(labels[selected].mean())
      conf = float(probabilities[selected].mean())
      result += count / max(total, 1) * abs(acc - conf)
    else:
      acc, conf = float('nan'), float('nan')
    details.append({'count': count, 'accuracy': acc, 'confidence': conf})
  return float(result), {'count': int(total), 'bins': details}


def phase_accuracy(target, prediction, classes=3):
  target, prediction = np.asarray(target).reshape(-1), np.asarray(prediction).reshape(-1)
  valid = (target >= 0) & (target < classes)
  target, prediction = target[valid], prediction[valid]
  per_class = {}
  for cls in range(classes):
    sel = target == cls
    per_class[str(cls)] = {
        'accuracy': float((prediction[sel] == cls).mean()) if sel.any() else float('nan'),
        'count': int(sel.sum())}
  return {
      'accuracy': float((target == prediction).mean()) if len(target) else float('nan'),
      'count': int(len(target)), 'per_class': per_class}


def depth_mae(prediction, target, valid=None):
  prediction, target = np.asarray(prediction), np.asarray(target)
  if valid is None:
    valid = target != 0
  valid = np.asarray(valid).astype(bool)
  valid &= np.isfinite(prediction) & np.isfinite(target)
  count = int(valid.sum())
  return (float(np.abs(prediction[valid] - target[valid]).mean()) if count else float('nan'), count)


def phase_transition_events(label_phase, is_first):
  """Return per-sequence pick/place success markers."""
  phase = np.asarray(label_phase)
  first = np.asarray(is_first).astype(bool)
  if phase.shape != first.shape or phase.ndim != 2:
    raise ValueError(f'label_phase and is_first must be (B,T), got {phase.shape} and {first.shape}')
  from embodied.envs.legmm.interfaces import Phase
  previous = np.concatenate([phase[:, :1], phase[:, :-1]], axis=1)
  valid = ~first
  return {
      'pick_success': valid & (previous == int(Phase.TO_PICK)) &
      (phase == int(Phase.HOLDING)),
      'place_success': valid & (previous == int(Phase.HOLDING)) &
      (phase == int(Phase.PLACED)),
  }


def reward_event_errors(reward_hat, reward, events, offsets=range(-5, 6),
                        is_first=None):
  reward_hat, reward = np.asarray(reward_hat), np.asarray(reward)
  if reward_hat.shape != reward.shape:
    raise ValueError(f'reward_hat and reward must have equal shapes, got {reward_hat.shape} and {reward.shape}')
  if reward.ndim == 1:
    reward_hat, reward = reward_hat[None], reward[None]
    first = None if is_first is None else np.asarray(is_first)[None]
  elif reward.ndim == 2:
    first = None if is_first is None else np.asarray(is_first).astype(bool)
    if first is not None and first.shape != reward.shape:
      raise ValueError(f'is_first must match rewards, got {first.shape} and {reward.shape}')
  else:
    raise ValueError(f'rewards must be (T,) or (B,T), got {reward.shape}')
  events = np.asarray(events)
  if events.dtype == bool:
    if events.shape == reward.shape:
      events = np.argwhere(events)
    elif reward.shape[0] == 1 and events.shape == reward.shape[1:]:
      events = np.argwhere(events)
      events = np.concatenate([np.zeros((len(events), 1), np.intp), events], axis=1)
    else:
      raise ValueError(f'boolean events must match rewards, got {events.shape} and {reward.shape}')
  elif events.size:
    events = np.asarray(events, np.intp)
    if events.ndim == 1:
      events = np.stack([np.zeros_like(events), events], axis=1)
    if events.ndim != 2 or events.shape[1] != 2:
      raise ValueError(f'events must be (N,2), got {events.shape}')
  result = {}
  for offset in offsets:
    errors = []
    for batch, event in events:
      batch, event = int(batch), int(event)
      index = event + int(offset)
      if not (0 <= batch < reward.shape[0] and 0 <= index < reward.shape[1]):
        continue
      if first is not None:
        lo, hi = sorted((event, index))
        if np.any(first[batch, lo + 1:hi + 1]):
          continue
      errors.append(abs(float(reward_hat[batch, index]) - float(reward[batch, index])))
    result[str(int(offset))] = {
        'mean': float(np.mean(errors)) if errors else float('nan'),
        'count': len(errors)}
  return result


# Short names keep the metric helpers convenient for downstream notebooks and
# tests while the verbose implementations above make their contracts clear.
def auroc(labels, scores, mask=None):
  return binary_auroc(labels, scores, mask)[0]


def ece(labels, probabilities, mask=None, bins=10):
  return ece_equal_width(labels, probabilities, mask, bins)[0]


def event_offset_errors(reward_hat, reward, events, offsets=range(-5, 6),
                        is_first=None):
  return reward_event_errors(reward_hat, reward, events, offsets, is_first)


def _sanitize(data, spaces):
  first = next(iter(data.values()))
  shape = first.shape[:2]
  return {k: np.asarray(data[k], dtype=s.dtype) if k in data else
          np.zeros((*shape, *s.shape), s.dtype) for k, s in spaces.items()}


def _load_config(names, logdir):
  import ruamel.yaml as yaml
  root = pathlib.Path(__file__).resolve().parents[1]
  raw = yaml.YAML(typ='safe').load((root / 'dreamerv3' / 'configs.yaml').read_text())
  cfg = elements.Config(raw['defaults'])
  for name in names:
    cfg = cfg.update(raw[name])
  return cfg.update(script='train_offline', logdir=str(logdir))


def evaluate(checkpoint, data_dir, configs, output, limit=None, context=16):
  # Importing main after the path setup matches dreamerv3/main.py's CLI layout.
  root = pathlib.Path(__file__).resolve().parents[1]
  module_root = root / 'dreamerv3'
  if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))
  import main
  from embodied.envs.legmm import interfaces as I
  from embodied.run.train_offline import _sanitize_batch

  cfg = _load_config(configs, pathlib.Path(output).parent)
  # Evaluation is deliberately on a single CPU device.  The pure ninjax
  # calls below expose intermediate tensors and do not use the agent's
  # sharded train transform; CPU keeps those calls portable across shared GPU
  # jobs while using the identical model modules and checkpoint parameters.
  cfg = cfg.update({'jax': {'platform': 'cpu', 'jit': True,
                            'policy_devices': [0], 'train_devices': [0]}})
  agent = main.make_agent(cfg)
  cp = pathlib.Path(checkpoint).expanduser()
  if (cp / 'latest').exists():
    cp = cp / (cp / 'latest').read_text().strip()
  elements.checkpoint.load(cp, {'agent': agent.load})
  # Direct pure calls below intentionally materialize host intermediates;
  # disable the training transform's transfer guard after checkpoint loading.
  jax.config.update('jax_transfer_guard', 'allow')
  replay = __import__('embodied').replay.Replay(
      length=int(cfg.consec_train * cfg.batch_length + cfg.replay_context),
      capacity=None, directory=pathlib.Path(output).parent / '_eval_replay', seed=0)
  replay.load(directory=pathlib.Path(data_dir).expanduser())
  raw_stream = main.make_stream(cfg, replay, 'train')
  # Keep batches as ordinary host numpy arrays for the pure calls.  The agent
  # stream wrapper deliberately shards and prefetches for training, which is
  # unnecessary here and makes direct intermediate-tensor inspection harder.
  stream = iter(__import__('embodied').streams.Map(
      raw_stream, _sanitize_batch, agent.spaces))
  eval_device = jax.devices()[0]
  # Start from the model's pure initializer so carries do not retain the
  # train transform's sharding metadata.
  initial_params = {k: jax.device_put(jax.device_get(v), eval_device)
                    for k, v in agent.params.items()}
  init_seed = agent._seeds(0, agent.train_mirrored)
  state, carry = nj.pure(agent.model.init_train)(
      initial_params, cfg.batch_size, seed=init_seed)
  apply = nj.pure(agent.model._apply_replay_context)
  loss = nj.pure(agent.model.loss)
  heads = nj.pure(agent.model.predict_heads)
  samples = 0
  aff_y, aff_p, aff_m = [[], []], [[], []], [[], []]
  phases_y, phases_p = [], []
  reward_y, reward_p = [], []
  event_masks = {'pick_success': [], 'place_success': []}
  first_masks = []
  depth_errors = {str(h): {'sum': 0.0, 'count': 0} for h in (1, 5, 15, 30)}
  while limit is None or samples < limit:
    batch = next(stream)
    # Agent.train mutates the dict by removing seed; copy for diagnostics.
    data = {k: v for k, v in batch.items()}
    data = jax.tree.map(lambda x: jax.device_put(x, eval_device), data)
    params = state
    seed = agent._seeds(samples, agent.train_mirrored)
    state, (carry3, obs, prevact, _) = apply(params, carry, data, seed=seed)
    state, result = loss(state, carry3, obs, prevact, False, seed=seed)
    repfeat = result[1][2]['repfeat']
    state, pred = heads(state, repfeat, seed=seed)
    pred = {k: np.asarray(v) for k, v in pred.items()}
    for i in range(2):
      aff_y[i].append(np.asarray(obs['label_aff'][:, :, i]))
      aff_m[i].append(np.asarray(obs['label_aff_mask'][:, :, i]))
      aff_p[i].append(1 / (1 + np.exp(-pred['aff'][..., i])))
    phases_y.append(np.asarray(obs['label_phase']))
    phases_p.append(np.asarray(pred['phase']))
    reward_y.append(np.asarray(obs['reward']))
    reward_p.append(np.asarray(pred['reward']))
    event_batch = phase_transition_events(obs['label_phase'], obs['is_first'])
    for key in event_masks:
      event_masks[key].append(event_batch[key])
    first_masks.append(np.asarray(obs['is_first']))
    for h in depth_errors:
      h = int(h)
      ctx = min(max(int(context), 1), obs['depth_head'].shape[1])
      if obs['depth_head'].shape[1] < ctx + h:
        continue
      # The model method uses its first context steps and recorded actions.
      state2, pred_depth = nj.pure(agent.model.openloop_depth)(
          state, carry3[:3], obs, prevact, h, context=ctx, seed=seed)
      for key, max_depth in (
          ('depth_head', float(cfg.env.legmm.camera_head.max_z)),
          ('depth_wrist', float(cfg.env.legmm.camera_wrist.max_z))):
        if key not in pred_depth:
          continue
        p = np.asarray(pred_depth[key])[:, :h] * max_depth
        t = np.asarray(obs[key])[:, ctx:ctx + h] / 255 * max_depth
        mae, count = depth_mae(p, t)
        if count:
          depth_errors[str(h)]['sum'] += mae * count
          depth_errors[str(h)]['count'] += count
    samples += cfg.batch_size
    if samples >= (limit or samples):
      break
  sequences_used = int(sum(x.shape[0] for x in reward_y))
  result = {'checkpoint': str(checkpoint), 'data': str(data_dir),
            'samples': int(samples), 'sequences_used': sequences_used}
  result['aff'] = {}
  for i, name in enumerate(('pick', 'place')):
    y, p, m = map(lambda xs: np.concatenate(xs).reshape(-1), (aff_y[i], aff_p[i], aff_m[i]))
    auc, pos, neg = binary_auroc(y >= 0.5, p, m > 0)
    ece, bins = ece_equal_width(y, p, m > 0)
    result['aff'][name] = {'auroc': auc, 'ece': ece, 'count': int((m > 0).sum()),
                           'positive': pos, 'negative': neg, 'ece_bins': bins}
  result['phase'] = phase_accuracy(np.concatenate(phases_y), np.concatenate(phases_p))
  result['depth_openloop'] = {
      h: {'mae_m': (v['sum'] / v['count'] if v['count'] else float('nan')),
          'count': v['count']} for h, v in depth_errors.items()}
  reward_pred = np.concatenate(reward_p)
  reward_true = np.concatenate(reward_y)
  first = np.concatenate(first_masks)
  event_errors = {
      key: reward_event_errors(
          reward_pred, reward_true, np.concatenate(event_masks[key]),
          is_first=first)
      for key in event_masks}
  result['reward_event_error'] = event_errors
  result['reward_event_error_pick_success'] = event_errors['pick_success']
  result['reward_event_error_place_success'] = event_errors['place_success']
  result['reward_event_source'] = 'label_phase TO_PICK->HOLDING and HOLDING->PLACED transitions'
  pathlib.Path(output).write_text(json.dumps(result, indent=2, allow_nan=True))
  return result


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--data', required=True)
  p.add_argument('--configs', nargs='+', default=['legmm', 'legmm_bc'])
  p.add_argument('--output', required=True)
  p.add_argument('--limit', type=int, default=None)
  p.add_argument('--context', type=int, default=16,
                 help='Posterior context steps for open-loop depth (default: 16).')
  a = p.parse_args()
  print(json.dumps(evaluate(a.checkpoint, a.data, a.configs, a.output, a.limit,
                           a.context), indent=2))


if __name__ == '__main__':
  main()
