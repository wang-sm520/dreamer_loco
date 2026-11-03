"""Offline replay training loop.

The on-disk LegMM chunks contain the observation/action keys plus ``sample_id``.
Model-dependent RSSM context keys are intentionally absent, so this runner
zero-fills those keys before handing a batch to the agent.  Keeping that
adaptation here makes the collected dataset usable by all phase presets.
"""

import pathlib
import time
from functools import partial as bind

import elements
import embodied
import numpy as np


def _checkpoint_path(path):
  """Accept either an elements checkpoint directory or its ``ckpt`` root."""
  if not path:
    return path
  path = elements.Path(path)
  latest = path / 'latest'
  if latest.exists():
    return path / latest.read_text().strip()
  return path


def _sanitize_batch(data, spaces):
  """Drop collection-only keys and add absent model context with zeros."""
  # Replay's Consec stream supplies ``consec`` and ``stepid``.  ``seed`` is
  # added by Agent.stream, so every other key must be an agent space key.
  if not data:
    return data
  first = next(iter(data.values()))
  batch_shape = first.shape[:2]
  out = {}
  for key, space in spaces.items():
    if key in data:
      value = data[key]
      expected = (*batch_shape, *space.shape)
      if value.shape != expected:
        raise ValueError(f'Offline key {key!r} has shape {value.shape}, expected {expected}')
      if value.dtype != space.dtype:
        value = value.astype(space.dtype, copy=False)
      out[key] = value
    else:
      out[key] = np.zeros((*batch_shape, *space.shape), space.dtype)
  # ``sample_id`` and future side-table fields are deliberately ignored.
  return out


def train_offline(make_agent, make_replay, make_stream, make_logger, args):
  """Train an agent for ``args.steps`` updates from an offline replay.

  No environment or driver is created.  The checkpoint layout matches
  :func:`embodied.run.train`: ``ckpt/{agent,replay}.pkl`` plus the step counter.
  """
  agent_start = time.perf_counter()
  agent = make_agent()
  agent_ready = time.perf_counter()
  print(f'Offline agent_compile_seconds={agent_ready - agent_start:.6f}')
  replay = make_replay()
  logger = make_logger()
  offline_dir = getattr(args, 'offline_dir', '')
  if not offline_dir:
    raise ValueError('train_offline requires --offline.dir')
  replay.load(directory=elements.Path(pathlib.Path(offline_dir).expanduser()))
  if not len(replay):
    raise RuntimeError(f'Offline replay has no samples: {offline_dir}')

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**getattr(args, 'usage', {}))
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)
  save_every_steps = int(getattr(args, 'save_every_steps', 0))
  train_agg = elements.Agg()
  train_fps = elements.FPS()

  raw = make_stream(replay, 'train')
  stream = embodied.streams.Map(raw, _sanitize_batch, agent.spaces)
  stream_train = iter(agent.stream(stream))
  raw_report = make_stream(replay, 'report')
  report_stream = embodied.streams.Map(raw_report, _sanitize_batch, agent.spaces)
  stream_report = iter(agent.stream(report_stream))
  carry_train = agent.init_train(args.batch_size)
  carry_report = agent.init_report(args.batch_size)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  from_checkpoint = _checkpoint_path(getattr(args, 'from_checkpoint', ''))
  if from_checkpoint:
    # Loading through the agent is intentional: it reports the phase-only
    # differences (agent_step / optimizer state) and rejects all other drift.
    elements.checkpoint.load(from_checkpoint, {
        'agent': bind(agent.load, regex=getattr(args, 'from_checkpoint_regex', '') or None)})
  cp.load_or_save()
  last_save_step = int(step)

  print(f'Loaded offline replay: {len(replay)} sequences from {offline_dir}')
  print('Start offline training loop')
  loop_start = time.perf_counter()
  first_start = None
  first_done = None
  while step < args.steps:
    first_start = time.perf_counter() if first_start is None else first_start
    batch = next(stream_train)
    carry_train, outs, mets = agent.train(carry_train, batch)
    if first_done is None:
      first_done = time.perf_counter()
    if 'replay' in outs:
      replay.update(outs['replay'])
    train_agg.add(mets, prefix='train')
    train_fps.step(args.batch_size * args.batch_length)
    step.increment()

    if should_report(step):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')
    if should_log(step):
      logger.add(train_agg.result())
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if save_every_steps > 0:
      if int(step) >= last_save_step + save_every_steps:
        cp.save()
        last_save_step = int(step)
    elif should_save(step):
      cp.save()

  # Always leave a checkpoint for short smoke runs whose save clock did not
  # fire.  ``run.from_checkpoint`` can therefore point at this directory.
  if int(step) != last_save_step:
    cp.save()
  elapsed = time.perf_counter() - loop_start
  compile_time = (first_done - first_start) if first_start is not None else 0.0
  print(f'Offline timing: first_train_seconds={compile_time:.6f} '
        f'updates={int(step)} steps_per_second={int(step) / max(elapsed, 1e-9):.6f} '
        f'loop_seconds={elapsed:.6f}')
  logger.close()

