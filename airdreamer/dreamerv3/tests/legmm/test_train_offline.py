import pathlib

import elements
import numpy as np

import embodied
from embodied.run.train_offline import train_offline


class _Agent:
  def __init__(self, phase='wm'):
    self.phase = phase
    self.obs_space = {'x': elements.Space(np.float32)}
    self.act_space = {'a': elements.Space(np.float32)}
    self.spaces = {
        **self.obs_space, **self.act_space,
        'consec': elements.Space(np.int32),
        'stepid': elements.Space(np.uint8, (20,)),
    }
    self.updates = 0

  def init_train(self, batch): return 0
  def init_report(self, batch): return 0
  def stream(self, stream):
    for data in stream:
      data = dict(data)
      data['seed'] = np.zeros((data['x'].shape[0],), np.uint32)
      yield data
  def train(self, carry, data):
    data.pop('seed')
    self.updates += 1
    return carry, {}, {'loss/test': np.float32(1)}
  def report(self, carry, data):
    data.pop('seed')
    return carry, {}
  def save(self):
    return {'phase': self.phase, 'updates': self.updates}
  def load(self, data, regex=None):
    self.phase = data['phase']
    self.updates = data['updates']


class _Logger:
  def __init__(self, logdir):
    self.step = elements.Counter()
    self.logdir = logdir
  def add(self, *args, **kwargs): pass
  def write(self): pass
  def close(self): pass


def _args(logdir, data):
  return elements.Config(
      logdir=str(logdir), offline_dir=str(data), steps=2, batch_size=1,
      batch_length=2, replay_context=0, consec_report=1, report_batches=1,
      report_every=10**9, log_every=10**9, save_every=10**9, save_every_steps=0,
      from_checkpoint='', from_checkpoint_regex='', usage={})


def test_train_offline_two_steps_checkpoint_and_bc_load(tmp_path):
  data = tmp_path / 'data'
  writer = embodied.replay.Replay(length=2, capacity=8, directory=data,
                                  chunksize=4, save_wait=True)
  for i in range(4):
    writer.add({'x': np.float32(i), 'a': np.float32(i),
                'is_first': i == 0, 'is_last': False,
                'is_terminal': False, 'sample_id': np.int64(i)}, 0)
  writer.save()

  def make_replay():
    return embodied.replay.Replay(length=2, capacity=8,
                                  directory=tmp_path / 'replay', chunksize=4)
  def make_stream(replay, mode):
    def batches():
      while True:
        batch = replay.sample(1, mode)
        batch['consec'] = np.zeros(batch['is_first'].shape, np.int32)
        yield batch
    return batches()
  logger = _Logger(tmp_path / 'log')
  wm = _Agent('wm')
  train_offline(lambda: wm, make_replay, make_stream, lambda: logger,
                _args(tmp_path / 'log', data))
  assert wm.updates == 2
  ckpts = list((tmp_path / 'log' / 'ckpt').glob('*/agent.pkl'))
  assert ckpts
  bc = _Agent('bc')
  elements.checkpoint.load(ckpts[0].parent, {'agent': bc.load})
  assert bc.phase == 'wm' and bc.updates == 2
