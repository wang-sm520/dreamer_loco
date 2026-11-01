#import pdb
#pdb.set_trace()

import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
bundled_omnidrones = folder.parent.parent / 'OmniDrones'
if bundled_omnidrones.exists():
  bundled_path = str(bundled_omnidrones)
  if bundled_path in sys.path:
    sys.path.remove(bundled_path)
  sys.path.insert(0, bundled_path)
sys.path.insert(1, str(folder.parent))
sys.path.insert(2, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml


def _normalize_list_args(argv):
  if argv is None:
    argv = sys.argv[1:]
  normalized = []
  for arg in argv:
    if arg.startswith('[') or arg.endswith(']'):
      arg = arg.replace('[', '').replace(']', '')
    if arg.endswith(','):
      arg = arg[:-1]
    normalized.append(arg)
  return normalized


def main(argv=None):
  
  #import pdb
  #pdb.set_trace()
  
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  #安全地解析 YAML 格式的配置字符串，将其转换为 Python 数据结构（如字典或列表）
  configs = yaml.YAML(typ='safe').load(configs)
  #创建一个 配置解析器，指定使用 默认配置文件 defaults（例如 config/defaults.yaml）作为基础配置;解析传入的 命令行参数列表 argv
  argv = _normalize_list_args(argv)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  #转换为点号访问对象
  config = elements.Config(configs['defaults'])
  
  #获取并更新具体任务的configs
  for name in parsed.configs:
    #将配置文件内容合并到当前配置对象（覆盖已有值）
    config = config.update(configs[name])
  
  #解析未识别的命令行参数并转换为配置项  
  config = elements.Flags(config).parse(other)
  
  #生成带时间戳的日志目录
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))
  
  #在分布式训练环境中自动识别当前副本（replica）的索引，确保每个训练进程能正确区分自身身份。本地单机训练下没用
  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)
  
  #打印一些配置项
  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  
  #用于智能决定是否保存配置文件，避免在非训练脚本（如环境初始化脚本main_env.py,main_replay.py）中生成不必要的日志目录和配置文件。
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  #根据配置文件（config.yaml）的设置，动态启用或禁用训练过程中的时间记录功能
  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  #启动实时监控服务（Portal）的关键配置，为训练过程提供了一个可视化调试界面（通常通过浏览器访问），方便开发者实时查看训练进度、性能指标和错误日志。
  portal.setup(
      errfile=config.errfile and logdir / 'error', #如果 config.errfile=True 则记录错误到 logdir/error 文件
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init], #初始化函数列表
      ipv6=config.ipv6,
  )

  #将训练所需的全部配置项合并为一个可点式访问的配置对象，用于后续训练初始化
  args = elements.Config(
      **config.run,#自动展开run字典
      task=config.task,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  if config.script == 'train':
    
    #bind返回一个无参函数，调用时自动传入绑定的参数;等效于lambda: make_agent(config);后续config被意外修改也不影响之前的绑定
    embodied.run.train(
        bind(make_agent, config), 
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent
  env = make_env(config, 0)
  #过滤指标键（key）的工具函数，专为避免与框架内部日志前缀冲突
  notlog = lambda k: not k.startswith('log/')  
  #过滤掉观测空间中所有以 log/ 开头的键，确保只保留用户定义的观测指标
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}  
  #移除动作空间中的 reset 键，确保只保留训练需要的常规动作
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}  
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  
  #分布式训练防止日志路径冲突
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir,
          'scores.jsonl',
          'episode/score|episode/length|episode/worker|episode/term_reason|episode/reach_goal',
          strings=True,
      ))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(elements.logger.WandBOutput(name))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report #连续步数（yaml设置为1）
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context #应该是一条轨迹的长度
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  #作为配置参数用于embodied.replay.Replay缓冲区的初始化
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)

  #用于配置 混合选择器（Mixture Selector）当config.replay.fracs.uniform < 1时
  #但是config.replay.fracs.uniform一般设置为1，即采用均匀采样
  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)

def make_env(config, index, **overrides):
  #获取任务的环境类，如config.task=dmc_walker_walk' -> suite='dmc',task='walker_walk' -> ctor='embodied.envs.dmc:DMC'
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'drone': 'embodied.envs.drone_multi:OmniDronesMulti',
      'dronenav': 'embodied.envs.drone_navigation:OmniDronesNavigation',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  env_key_alias = {
      'drone': 'drone_multi',
      'dronenav': 'drone_nav',
  }
  env_key = env_key_alias.get(suite, suite)
  kwargs = dict(config.env.get(env_key, {}))
  kwargs.update(overrides)
  kwargs['logdir'] = config.logdir
  if suite in ('drone', 'dronenav'):
    kwargs.setdefault('num_drones', int(config.run.envs))
    train_devices = tuple(config.jax.train_devices)
    if train_devices:
      kwargs.setdefault('device', f'cuda:{int(train_devices[0])}')
  
  #环境索引，确保每个环境实例的随机种子不同，避免训练过程中的数据碰撞
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  #为分布式训练的每个环境实例指定日志目录，方便后续分析和调试 
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
  #创建环境实例  
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)

  return stream


if __name__ == '__main__':
  main()
