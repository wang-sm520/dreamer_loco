import collections
import os
from functools import partial as bind

import elements
import embodied
import numpy as np


def _save_openloop_video(frames, path, fps=4):
  try:
    import imageio
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
      frames = np.clip(frames, 0, 255).astype(np.uint8)
    imageio.mimwrite(str(path), frames, fps=int(fps), quality=7)
    print(f'[Video] Saved {path}  ({frames.shape[0]} frames)')
  except Exception as e:
    print(f'[Video] Failed to save {path}: {e}')


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  term_reason_names = (
      'none', 'timeout', 'nan', 'collision', 'ground', 'out_of_bounds', 'terminal'
  )
  if str(getattr(args, 'task', '')).startswith('legmm_'):
    from embodied.envs.legmm.interfaces import TERM_REASONS as term_reason_names
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)
  video_every = float(getattr(args, 'video_every', 0))
  should_video = embodied.LocalClock(video_every) if video_every > 0 else None
  save_every_steps = int(getattr(args, 'save_every_steps', 0))

  def maybe_save_videos():
    if not len(replay) or 'depth' not in getattr(agent, 'obs_space', {}):
      return
    video_dir = os.path.join(str(logdir), 'video')
    os.makedirs(video_dir, exist_ok=True)
    try:
      batch = next(stream_report)
      _, mets = agent.report(carry_report, batch)
      grid = mets.get('openloop/depth')
      if grid is None:
        return
      grid = np.asarray(grid)
      if grid.ndim == 4 and grid.shape[-1] in (1, 3):
        path = os.path.join(video_dir, f'openloop_depth_{int(step)}.mp4')
        _save_openloop_video(grid, path)
    except Exception as e:
      print(f'[Video] Failed to generate openloop video: {e}')

  #日志功能，仅在回合结束时批量记录
  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        if key == 'log/term_reason':
          continue
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      term_reason_id = int(tran.get('log/term_reason', 0))
      reach_goal = bool(float(tran.get('log/reach_goal', 0.0)))
      term_reason = (
          term_reason_names[term_reason_id]
          if 0 <= term_reason_id < len(term_reason_names) else 'unknown'
      )
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
          'worker': int(worker),
          'term_reason': term_reason,
          'reach_goal': reach_goal,
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  task_name = str(getattr(args, 'task', ''))
  is_drone_task = task_name.startswith('drone_') or task_name.startswith('dronenav_')
  # OmniDrones envs share a single Isaac scene and are not process-safe.
  driver = embodied.Driver(fns, parallel=(not args.debug) and (not is_drone_task))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)
  

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  #carry_train->内部状态（carry state）：在序列处理中保持上下文（如RNN的隐藏状态）?
  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()
  last_save_step = int(step)

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_video and should_video(step) and len(replay):
      maybe_save_videos()

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if save_every_steps > 0:
      if int(step) >= last_save_step + save_every_steps:
        cp.save()
        if should_video is None:
          maybe_save_videos()
        last_save_step = int(step)
    elif should_save(step):
      cp.save()
      if should_video is None:
        maybe_save_videos()

  logger.close()
