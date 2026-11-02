import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


def anneal_scale(step, scale, scale_end, anneal_steps):
  """Linearly interpolate a loss coefficient and clamp after the schedule."""
  if isinstance(step, (int, float, np.integer, np.floating)):
    frac = min(max(float(step) / max(float(anneal_steps), 1.0), 0.0), 1.0)
    return float(scale) + frac * (float(scale_end) - float(scale))
  steps = jnp.maximum(f32(anneal_steps), 1.0)
  frac = jnp.clip(f32(step) / steps, 0.0, 1.0)
  return f32(scale) + frac * (f32(scale_end) - f32(scale))


def bc_action_targets(prevact):
  """Return actions executed at obs[t], plus a final unpaired position.

  Replay context stores the action arriving before each observation in
  ``prevact[t]``; consequently the action recorded with obs[t] is
  ``prevact[t+1]``.  The returned tree is sliced to T-1 and loss code pads
  its final scalar entry with zero.
  """
  return jax.tree.map(lambda x: x[:, 1:], prevact)


def bc_policy_input(repinp):
  """Features paired with actions at obs[t], excluding the final pad slot."""
  return sg(repinp[:, :-1])


def soft_bce_logits(logits, labels):
  """Stable BCE for soft labels, including logits with magnitude 30+."""
  if isinstance(logits, np.ndarray):
    return np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
  logits, labels = f32(logits), f32(labels)
  return jnp.maximum(logits, jnp.zeros_like(logits)) - logits * labels + jax.nn.softplus(-jnp.abs(logits))


def skill_prior_jax(aff_pick, aff_place, p_holding, eps=0.05):
  """JAX copy of ``embodied.envs.legmm.interfaces.skill_prior``."""
  not_hold = jnp.stack([1 - aff_pick, aff_pick, jnp.zeros_like(aff_pick)], -1)
  hold = jnp.stack([1 - aff_place, jnp.zeros_like(aff_place), aff_place], -1)
  prior = (1 - p_holding)[..., None] * not_hold + p_holding[..., None] * hold
  return (1 - eps) * prior + eps / 3.0


skill_prior_jnp = skill_prior_jax


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    
    #安全排除非必要键，确保编码器和解码器只处理状态相关观测，避免训练时无关数据污染状态表示
    exclude = (
        'is_first', 'is_last', 'is_terminal', 'reward',
        *tuple(config.label_keys))
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    
    #构造enc、dyn、dec
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    #确定性状态和随机状态拼接
    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    self.train_phase = config.train_phase
    self._imag_enabled = bool(config.imag_enabled) and self.train_phase not in ('wm', 'bc')
    bc_cfg = config.bc
    self._bc_enabled = bool(config.skill.enabled) and self.train_phase in ('bc', 'rl', 'online') and (
        float(bc_cfg.scale) != 0.0 or float(bc_cfg.scale_end) != 0.0)
    prior_cfg = config.skill_prior
    self._skill_prior_enabled = bool(config.skill.enabled) and (
        float(prior_cfg.scale) != 0.0 or float(prior_cfg.scale_end) != 0.0)
    self._schedule_enabled = self._bc_enabled or self._skill_prior_enabled
    self.step = (nj.Variable(jnp.array, 0, i32, name='agent_step')
                 if self._schedule_enabled else None)
    
    #构建reward、continue head
    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    #定义动作空间,离散或连续
    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    #创建Policy Head
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')
    
    #普通值函数（self.val）和慢值函数（self.slowval）
    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)
    
    #创建归一化层，用于奖励、值函数和优势函数的归一化
    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # The affordance head emits two unconstrained logits. The masked BCE below
    # is computed directly from those logits so soft labels are supported.
    self.aff = None
    if config.aff_head.enabled:
      kw = dict(config.aff_head)
      kw.pop('enabled', None)
      aff_scale = kw.pop('scale', 1.0)
      self.aff_scale = float(aff_scale)
      aff_space = elements.Space(np.int32, (2,), 0, 2)
      self.aff = embodied.jax.MLPHead(
          aff_space, output='binary', **kw, name='aff')

    self.phase = None
    if config.phase_head.enabled:
      kw = dict(config.phase_head)
      kw.pop('enabled', None)
      phase_scale = kw.pop('scale', 1.0)
      self.phase_scale = float(phase_scale)
      phase_space = elements.Space(np.int32, (), 0, 3)
      self.phase = embodied.jax.MLPHead(
          phase_space, output='categorical', **kw, name='phase')

    if self._skill_prior_enabled and (self.aff is None or self.phase is None):
      raise ValueError(
          'skill_prior requires both aff_head.enabled and phase_head.enabled')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    if self.aff is not None:
      self.modules.append(self.aff)
    if self.phase is not None:
      self.modules.append(self.phase)
    #定义了强化学习模型的优化器（Optimizer），用于训练模型的所有可训练参数
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    #设置损失函数中各部分的权重（loss scales），特别针对重建损失（reconstruction loss） 的权重分配
    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    overrides = {}
    if config.rec_scales:
      overrides = {
          key.strip(): float(value)
          for item in config.rec_scales.split(',') if item.strip()
          for key, value in [item.split('=', 1)]}
    scales.update({k: overrides.get(k, rec) for k in dec_space})
    # The table of enabled losses is built once and is checked at the end of
    # loss().  In wm/bc phases imagination is skipped, so policy/value/repval
    # are absent from both dictionaries.  BC is enabled only when its
    # annealed coefficient can be non-zero; this leaves drone defaults exact.
    if not self._imag_enabled:
      for key in ('policy', 'value', 'repval'):
        scales.pop(key, None)
    elif not config.repval_loss:
      scales.pop('repval', None)
    if self.aff is not None:
      scales['aff'] = self.aff_scale
    if self.phase is not None:
      scales['phase'] = self.phase_scale
    if self._bc_enabled:
      scales['bc'] = 1.0
    if self._skill_prior_enabled and self._imag_enabled:
      scales['skill_prior'] = 1.0
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  def bc_scale(self, step=None):
    step = self.step.read() if step is None and self.step is not None else (0 if step is None else step)
    return anneal_scale(
        step, self.config.bc.scale, self.config.bc.scale_end,
        self.config.bc.anneal_steps)

  def prior_scale(self, step=None):
    step = self.step.read() if step is None and self.step is not None else (0 if step is None else step)
    return anneal_scale(
        step, self.config.skill_prior.scale, self.config.skill_prior.scale_end,
        self.config.skill_prior.anneal_steps)

  def mask_inactive(self, act):
    if not self.config.skill.enabled:
      return act
    skill = act[self.config.skill.key]
    act = dict(act)
    for i, key in enumerate(self.config.skill.action_keys):
      act[key] = act[key] * (skill == jnp.zeros_like(skill) + i)[..., None].astype(
          act[key].dtype)
    return act

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    act = self.mask_inactive(act)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    if self.step is not None and not nj.creating():
      self.step.write(self.step.read() + i32(1))
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # Replay stores one value for every action head. For skill-conditioned
    # data, the RSSM receives inactive heads masked to zero. BC keeps the raw
    # stored action so its skill selector can ignore inactive heads explicitly.
    bc_prevact = prevact
    prevact = self.mask_inactive(prevact)

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      loss = recon.loss(sg(target))
      if self.config.rec_per_pixel_norm and len(space.shape) == 3:
        loss /= space.shape[0] * space.shape[1]
      losses[key] = loss

    # World-model auxiliary supervision.  The returned affordance tensor is
    # constant over (B,T), with value equal to the global masked mean.  Thus
    # the optimizer's ordinary batch mean is exactly
    # sum(mask * BCE) / max(sum(mask), 1), while retaining the required (B,T)
    # loss shape shared by every other loss term.
    repinp = self.feat2tensor(repfeat)
    if self.aff is not None:
      aff_logits = self.aff(repinp, 2).output.logit
      labels = f32(obs['label_aff'])
      mask = f32(obs['label_aff_mask'])
      bce = soft_bce_logits(aff_logits, labels)
      denom = jnp.maximum(mask.sum(), 1.0)
      aff_mean = (mask * bce).sum() / denom
      losses['aff'] = jnp.ones((B, T), f32) * aff_mean
      correct = (jax.nn.sigmoid(aff_logits) >= 0.5) == (labels >= 0.5)
      metrics['aff_acc'] = (f32(correct) * mask).sum() / denom
    if self.phase is not None:
      phase_out = self.phase(repinp, 2)
      losses['phase'] = phase_out.loss(obs['label_phase'])
      metrics['phase_acc'] = f32(
          phase_out.pred() == obs['label_phase']).mean()

    if self._bc_enabled:
      # _apply_replay_context makes prevact[t+1] the action stored alongside
      # obs[t].  We therefore compare policy outputs at t=0..T-2 with
      # prevact[:,1:] and pad the final (unpaired) position with zero.
      # Evaluate the policy on the paired feature prefix.  MLPHead returns
      # Output objects (rather than arrays), so slicing the input sequence is
      # the equivalent of slicing each distribution in P to [:, :-1].
      pol = self.pol(bc_policy_input(repinp), 2)
      target = bc_action_targets(bc_prevact)
      skill = target[self.config.skill.key]
      lp = pol[self.config.skill.key].logp(skill)
      for i, key in enumerate(self.config.skill.action_keys):
        lp += (skill == i) * pol[key].logp(target[key])
      bc = -lp * self.bc_scale()
      # Dataset writers store a zero action on terminal observations.  That
      # action is a reset sentinel, so terminal obs[t] must contribute zero to
      # BC even though it is present at prevact[t+1].
      bc *= 1.0 - f32(obs['is_last'][:, :-1])
      losses['bc'] = jnp.concatenate([bc, jnp.zeros((B, 1), f32)], 1)
      metrics['bc_scale'] = self.bc_scale()

    if not self._imag_enabled:
      # Optimizer modules are still constructed so their parameter trees stay
      # stable. Touch policy/value once to register their parameters, but do
      # not imagine or include any actor-critic loss in wm/bc phases; their
      # resulting gradients are exactly zero.
      self.pol(sg(repinp), 2)
      self.val(repinp, 2)
      self.slowval(repinp, 2)

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    imgloss_out = None
    if self._imag_enabled:
      K = min(self.config.imag_last or T, T)
      H = self.config.imag_length
      starts = self.dyn.starts(dyn_entries, dyn_carry, K)
      policyfn = lambda feat: self.mask_inactive(
          sample(self.pol(self.feat2tensor(feat), 1)))
      _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
      first = jax.tree.map(
          lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
      imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
      lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([imgprevact, lastact], 1)
      assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
      assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
      inp = self.feat2tensor(imgfeat)
      los, imgloss_out, mets = imag_loss(
          imgact,
          self.rew(inp, 2).pred(),
          self.con(inp, 2).prob(1),
          self.pol(inp, 2),
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          skill_key=(self.config.skill.key if self.config.skill.enabled else None),
          action_keys=(tuple(self.config.skill.action_keys)
                       if self.config.skill.enabled else ()),
          switch_cost=(float(self.config.switch_cost)
                       if self.config.skill.enabled else 0.0),
          aff=(self.aff(inp, 2) if self._skill_prior_enabled else None),
          phase=(self.phase(inp, 2) if self._skill_prior_enabled else None),
          skill_prior_scale=(self.prior_scale()
                             if self._skill_prior_enabled else None),
          skill_prior_eps=float(self.config.skill_prior.eps),
          **self.config.imag_loss)
      losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
      metrics.update(mets)

      # Replay value loss is meaningful only when imagination ran.
      if self.config.repval_loss:
        feat = sg(repfeat, skip=self.config.repval_grad)
        last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
        boot = imgloss_out['ret'][:, 0].reshape(B, K)
        feat, last, term, rew, boot = jax.tree.map(
            lambda x: x[:, -K:], (feat, last, term, rew, boot))
        inp = self.feat2tensor(feat)
        los, reploss_out, mets = repl_loss(
            last, term, rew, boot,
            self.val(inp, 2),
            self.slowval(inp, 2),
            self.valnorm,
            update=training,
            horizon=self.config.horizon,
            **self.config.repl_loss)
        losses.update(los)
        metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def predict_heads(self, repfeat):
    """Evaluate auxiliary heads on a representation for offline diagnostics.

    This tiny ninjax-facing method keeps evaluation on the exact modules used
    by training; the evaluator does not copy or reimplement any head network.
    """
    inp = self.feat2tensor(repfeat)
    out = {}
    if self.aff is not None:
      out['aff'] = self.aff(inp, 2).output.logit
    if self.phase is not None:
      out['phase'] = self.phase(inp, 2).pred()
    out['reward'] = self.rew(inp, 2).pred()
    return out

  def openloop_depth(self, carry, obs, prevact, horizon, context=16):
    """Posterior context followed by recorded-action RSSM imagination.

    The carry before this context is the zero-filled replay entry. Offline
    chunks contain no stored RSSM latents, so the context is reconstructed from
    observations and actions before the open-loop rollout begins.
    """
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    _, _, tokens = self.enc(enc_carry, obs, reset, training=False)
    ctx = min(max(int(context), 1), reset.shape[1])
    dyn_carry, _, _ = self.dyn.observe(
        dyn_carry, tokens[:, :ctx], {k: v[:, :ctx] for k, v in prevact.items()},
        reset[:, :ctx], training=False)
    actions = {k: v[:, ctx:ctx + horizon] for k, v in prevact.items()}
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, actions, length=horizon, training=False)
    _, _, recons = self.dec(
        dec_carry, imgfeat, jnp.zeros((reset.shape[0], horizon), bool), training=False)
    return {k: v.pred() for k, v in recons.items()
            if k in ('depth_head', 'depth_wrist')}

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    metrics.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds (uint8 images)
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)
      if video.shape[-1] == 1:  # single-channel uint8 depth (legmm)
        video = jnp.repeat(video, 3, axis=-1)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      # Local names: rebinding T here broke the border of the next image key.
      vB, vT, vH, vW, vC = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((vT, vH, vB * vW, vC))
      metrics[f'openloop/{key}'] = grid

    # Video preds (float32 depth)
    for key in self.dec.depthkeys:
      true = f32(obs[key][:RB])
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      true = jnp.clip(true, 0.0, 1.0)
      pred = jnp.clip(pred, 0.0, 1.0)
      error = (pred - true + 1.0) / 2.0

      if true.shape[-1] == 1:
        true = jnp.repeat(true, 3, axis=-1)
        pred = jnp.repeat(pred, 3, axis=-1)
        error = jnp.repeat(error, 3, axis=-1)

      video = jnp.concatenate([true, pred, error], 2)
      video = jnp.clip(video * 255, 0, 255).astype(jnp.uint8)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      vmask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      vT = video.shape[1]
      border = jnp.full((vT, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(vmask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      vB, vT, vH, vW, vC = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((vT, vH, vB * vW, vC))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    actsmooth=0.0,
    slowreg=1.0,
    skill_key=None,
    action_keys=(),
    switch_cost=0.0,
    aff=None,
    phase=None,
    skill_prior_scale=None,
    skill_prior_eps=0.05,
):
  losses = {}
  metrics = {}

  # imgact[:, t] is the action taken in imagined state t, while rew[:, t]
  # arrives at state t.  A switch observed at t therefore penalizes rew[:, t+1].
  if switch_cost and skill_key in act:
    sw = (act[skill_key][:, 1:] != act[skill_key][:, :-1]).astype(f32)
    rew = rew.at[:, 2:].add(-f32(switch_cost) * sw[:, :-1])

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  smooths = {}
  if skill_key in act and skill_key in policy:
    sk = act[skill_key][:, :-1]
    logpi = policy[skill_key].logp(sg(act[skill_key]))[:, :-1]
    ent = ents[skill_key]
    smooth_total = 0
    for i, k in enumerate(action_keys):
      if k not in policy or k not in act:
        continue
      on = (sk == i).astype(f32)
      logpi += on * policy[k].logp(sg(act[k]))[:, :-1]
      ents[k] = on * ents[k]
      ent += ents[k]
      mean_k = policy[k].pred()
      diff = mean_k[:, 1:] - mean_k[:, :-1]
      smooth = (diff ** 2).sum(axis=tuple(range(2, diff.ndim)))
      # Smoothness is defined only for a continuous skill's own action head;
      # both adjacent imagined states must select that skill.
      both = ((act[skill_key][:, :-1] == i) &
              (act[skill_key][:, 1:] == i)).astype(f32)
      smooths[k] = both * smooth
      smooth_total = smooth_total + smooths[k]
    # The categorical selector has no smoothness penalty.
    smooths[skill_key] = jnp.zeros_like(ents[skill_key])
  else:
    logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
    ent = sum(ents.values())
    smooth_total = 0
    for k, v in policy.items():
      mean_k = v.pred()
      diff = mean_k[:, 1:] - mean_k[:, :-1]
      smooths[k] = (diff ** 2).sum(axis=tuple(range(2, diff.ndim)))
    smooth_total = sum(smooths.values())

  for k, v in policy.items():
    if k in smooths:
      continue
    mean_k = v.pred()
    diff = mean_k[:, 1:] - mean_k[:, :-1]
    smooths[k] = (diff ** 2).sum(axis=tuple(range(2, diff.ndim)))
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * ent
      - actsmooth * smooth_total)
  losses['policy'] = policy_loss

  if skill_prior_scale is not None:
    if skill_key not in policy or aff is None or phase is None:
      raise ValueError(
          'skill_prior requires skill policy, aff head, and phase head outputs')
    def _field(output, name):
      if hasattr(output, name):
        return getattr(output, name)
      if hasattr(output, 'output') and hasattr(output.output, name):
        return getattr(output.output, name)
      return output
    aff_logits = sg(_field(aff, 'logit'))
    phase_logits = sg(_field(phase, 'logits'))
    aff_prob = jax.nn.sigmoid(aff_logits)
    hold = jax.nn.softmax(phase_logits, -1)[..., 1]
    prior = skill_prior_jax(
        aff_prob[..., 0], aff_prob[..., 1], hold, eps=skill_prior_eps)
    pi_logits = policy[skill_key].logits
    logpi_skill = jax.nn.log_softmax(pi_logits, -1)
    pi = jax.nn.softmax(pi_logits, -1)
    kl = (pi * (logpi_skill - jnp.log(prior))).sum(-1)
    losses['skill_prior'] = sg(weight[:, :-1]) * kl[:, :-1] * skill_prior_scale

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    if skill_key in act and k in action_keys:
      i = action_keys.index(k)
      active = (act[skill_key][:, :-1] == i).astype(f32)
      metrics[f'ent/{k}'] = (ents[k] * active).sum() / jnp.maximum(active.sum(), 1.0)
    else:
      metrics[f'ent/{k}'] = ents[k].mean()
    metrics[f'smooth/{k}'] = smooths[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  outs['rew'] = rew
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
