"""
legmm_interfaces.py — A2 Pro + AIRBOT Play 深度输入移动操作（nav → pick → carry → place）
的接口契约与可直接复用的纯 numpy 工具函数。

放置位置：airdreamer/dreamerv3/embodied/envs/legmm/interfaces.py
依赖：numpy（elements 仅在 make_*_space 中惰性导入）。
本文件中的常量 = 单一事实来源；env / expert / labeler / agent 修改 / eval 全部从这里 import。
带 "TODO(calibrate)" 的数值需按 A2 / AIRBOT 实机标定后替换。
"""
from __future__ import annotations

import dataclasses
import enum
import math
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 0. 时序
# ---------------------------------------------------------------------------
PHYSICS_DT = 0.005          # 200 Hz 物理
HL_HZ = 5                   # 高层（Dreamer）控制频率
HL_DT = 1.0 / HL_HZ         # 0.2 s
DECIMATION = int(round(HL_DT / PHYSICS_DT))  # 40 物理步 / 高层步
LL_LOCO_HZ = 50             # 低层 locomotion 策略（P2 阶段，Isaac Lab）
MAX_EPISODE_STEPS = 450     # 90 s @ 5 Hz


# ---------------------------------------------------------------------------
# 1. 技能（"多阶段通过头实现"）
# ---------------------------------------------------------------------------
class Skill(enum.IntEnum):
  NAV = 0     # 底盘导航（未持物时去抓取点；持物时去放置点 —— 由 goal + 夹爪状态区分）
  PICK = 1    # 抓取：臂 + 小幅底盘调整 + 姿态
  PLACE = 2   # 放置：臂 + 小幅底盘调整 + 姿态


NUM_SKILLS = len(Skill)

# 每个技能一个动作 key（= 策略的一个输出头；DreamerV3 MLPHead 天然为每个 key 建独立输出层）
# 布局：name -> (slice, low_phys, high_phys)，策略输出 [-1,1] 线性映射到物理量
# Body height command: base_link origin height above ground. Range = belly on
# the ground (base mesh bottom is 0.078 m below the origin, unitree_a2/xmls/a2.xml)
# up to the standing height of the trained UniLab A2JoystickFlat policy at zero
# velocity command: 0.385 m (16 envs x 10 s, docs/legmm/m1_a2_stand_height.json).
# Pitch is not commanded (user decision 2026-11-02).
BASE_HEIGHT_RANGE = (0.08, 0.385)

NAV_LAYOUT = {
    'vx':        (slice(0, 1), -0.8, 0.8),     # m/s
    'vy':        (slice(1, 2), -0.4, 0.4),     # m/s
    'wz':        (slice(2, 3), -1.0, 1.0),     # rad/s
    'height':    (slice(3, 4), *BASE_HEIGHT_RANGE),   # m, body height command
}
NAV_DIM = 4

MANIP_LAYOUT = {  # PICK 与 PLACE 共享布局，但是两个独立的头/动作 key
    'dee_pos':   (slice(0, 3), -0.05, 0.05),   # m / 高层步（=0.25 m/s），末端位置增量（机体系）
    'dee_rot':   (slice(3, 6), -0.30, 0.30),   # rad / 高层步，轴角增量
    'grip':      (slice(6, 7), 0.0, 1.0),      # 目标开度（0=闭合, 1=全开）
    'base_adj':  (slice(7, 10), -0.15, 0.15),  # vx, vy (m/s), wz 共用同一缩放（wz 实际 ×2）
    'height':    (slice(10, 11), *BASE_HEIGHT_RANGE),  # 同 NAV
}
MANIP_DIM = 11

SKILL_ACTION_KEY = {Skill.NAV: 'nav', Skill.PICK: 'pick', Skill.PLACE: 'place'}
ACTION_DIMS = {'nav': NAV_DIM, 'pick': MANIP_DIM, 'place': MANIP_DIM}


def make_act_space():
  """env.act_space（含 'reset'；make_agent 会去掉 'reset'）。"""
  import elements
  return {
      'reset': elements.Space(bool),
      'skill': elements.Space(np.int32, (), 0, NUM_SKILLS),
      'nav':   elements.Space(np.float32, (NAV_DIM,), -1.0, 1.0),
      'pick':  elements.Space(np.float32, (MANIP_DIM,), -1.0, 1.0),
      'place': elements.Space(np.float32, (MANIP_DIM,), -1.0, 1.0),
  }


def mask_inactive_np(act: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
  """把非当前技能的动作 key 置零。agent.policy() 输出后、imagination policyfn 内、
  scripted expert 写 replay 前都必须调用同一语义（jax 版见 brief §5.3）。
  支持 batch：act['skill'] 形状 (...,)；其他 key 形状 (..., D)。"""
  out = dict(act)
  skill = np.asarray(act['skill'])
  for s, key in SKILL_ACTION_KEY.items():
    if key in out:
      on = (skill == int(s)).astype(np.float32)[..., None]
      out[key] = np.asarray(out[key], np.float32) * on
  return out


def scale_action(vec: np.ndarray, layout) -> Dict[str, np.ndarray]:
  """[-1,1] 策略输出 -> 物理量字典。"""
  vec = np.clip(np.asarray(vec, np.float32), -1.0, 1.0)
  res = {}
  for name, (sl, lo, hi) in layout.items():
    res[name] = lo + (vec[..., sl] + 1.0) * 0.5 * (hi - lo)
  return res


def unscale_action(phys: Dict[str, np.ndarray], layout, dim: int) -> np.ndarray:
  """物理量字典 -> [-1,1]（scripted expert 用）。"""
  vec = np.zeros((dim,), np.float32)
  for name, (sl, lo, hi) in layout.items():
    v = np.asarray(phys.get(name, (lo + hi) / 2), np.float32).reshape(-1)
    vec[sl] = np.clip(2.0 * (v - lo) / (hi - lo) - 1.0, -1.0, 1.0)
  return vec


# ---------------------------------------------------------------------------
# 2. 观测
# ---------------------------------------------------------------------------
DEPTH_H, DEPTH_W = 48, 80  # 两路深度必须同分辨率（rssm.Decoder 按通道拼接）

PROPRIO_LAYOUT = {           # float32 (28,)
    'v_body':      slice(0, 3),
    'w_body':      slice(3, 6),
    'g_proj':      slice(6, 9),
    'base_height': slice(9, 10),
    'base_pitch':  slice(10, 11),
    'q_arm':       slice(11, 17),
    'ee_pos_b':    slice(17, 20),
    'ee_rot6d_b':  slice(20, 26),
    'grip_width':  slice(26, 27),   # 归一化 0..1 —— 必须可观测（区分是否持物）
    'grip_effort': slice(27, 28),   # 归一化 0..1
}
PROPRIO_DIM = 28

GOAL_LAYOUT = {              # float32 (16,)；pick 与 place 各 8 维
    'pick_dir_b':   slice(0, 3),    # 机体系单位方向
    'pick_dxy':     slice(3, 4),    # 水平距离 m
    'pick_dz':      slice(4, 5),    # 高度差 m
    'pick_yaw_sc':  slice(5, 7),    # 物体相对朝向 sin, cos
    'pick_valid':   slice(7, 8),    # 教师信号是否有效（模拟 RGB 检测丢失）
    'place_dir_b':  slice(8, 11),
    'place_dxy':    slice(11, 12),
    'place_dz':     slice(12, 13),
    'place_yaw_sc': slice(13, 15),
    'place_valid':  slice(15, 16),
}
GOAL_DIM = 16

# 仅训练用的标签 key —— 必须加入 agent.py 的 exclude 元组，不进 enc/dec，不能用 'log/' 前缀
LABEL_KEYS = ('label_aff', 'label_aff_mask', 'label_phase')


class Phase(enum.IntEnum):
  TO_PICK = 0
  HOLDING = 1
  PLACED = 2


def make_obs_space(depth_dtype=np.uint8):
  import elements
  return {
      'depth_head':  elements.Space(depth_dtype, (DEPTH_H, DEPTH_W, 1)),
      'depth_wrist': elements.Space(depth_dtype, (DEPTH_H, DEPTH_W, 1)),
      'proprio':     elements.Space(np.float32, (PROPRIO_DIM,)),
      'goal':        elements.Space(np.float32, (GOAL_DIM,)),
      'label_aff':      elements.Space(np.float32, (2,)),  # [aff_pick, aff_place] ∈ [0,1]
      'label_aff_mask': elements.Space(np.float32, (2,)),  # 1 = 该标签有效
      'label_phase':    elements.Space(np.int32, (), 0, 3),
      'reward':      elements.Space(np.float32),
      'is_first':    elements.Space(bool),
      'is_last':     elements.Space(bool),
      'is_terminal': elements.Space(bool),
      'log/term_reason':  elements.Space(np.int32, (), 0, len(TERM_REASONS)),
      'log/success':      elements.Space(np.float32),
      'log/pick_success': elements.Space(np.float32),
      'log/scene_type':   elements.Space(np.int32, (), 0, len(SceneType)),
  }


# ---------------------------------------------------------------------------
# 3. RealSense D435i 深度模型（头部 + 腕部默认同型号）
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class D435iModel:
  hfov_deg: float = 87.0
  vfov_deg: float = 58.0
  min_z: float = 0.28        # m；Min-Z 随分辨率变化 —— TODO(calibrate) 实测
  max_z: float = 6.0         # m；超出 → 记为 max_z（"远"）
  noise_a: Tuple[float, float] = (0.001, 0.004)   # σ(z) = a·z²，a 每回合随机
  hole_frac: Tuple[float, float] = (0.0, 0.03)    # 随机空洞比例
  edge_hole_prob: float = 0.3                     # 深度跳变边缘置空概率
  edge_jump: float = 0.15                         # m，判定为边缘的相邻差


def apply_depth_noise(z: np.ndarray, rng: np.random.Generator, cam: D435iModel,
                      a: Optional[float] = None) -> np.ndarray:
  """z: (H,W) 真值深度（m，np.inf=无命中）。返回带噪深度，无效像素 = 0。"""
  z = np.asarray(z, np.float64).copy()
  far = ~np.isfinite(z) | (z > cam.max_z)
  a = rng.uniform(*cam.noise_a) if a is None else a
  zn = z + rng.normal(0.0, 1.0, z.shape) * a * np.square(np.where(far, 0.0, z))
  zn[far] = cam.max_z
  invalid = (z < cam.min_z)
  # Prepend from the far-filled image: prepending raw z put inf - max_z = inf
  # at the first row/column of no-hit pixels, marking them as edges.
  zf = np.where(far, cam.max_z, z)
  dzx = np.abs(np.diff(zf, axis=1, prepend=zf[:, :1]))
  dzy = np.abs(np.diff(zf, axis=0, prepend=zf[:1, :]))
  edge = (np.maximum(dzx, dzy) > cam.edge_jump) & (rng.random(z.shape) < cam.edge_hole_prob)
  holes = rng.random(z.shape) < rng.uniform(*cam.hole_frac)
  zn[invalid | edge | holes] = 0.0
  return zn.astype(np.float32)


def depth_to_uint8(z: np.ndarray, cam: D435iModel) -> np.ndarray:
  """0 = 无效；1..255 线性映射 (0, max_z]。replay 体积比 float32 小 4 倍。"""
  q = np.clip(np.round(np.asarray(z) / cam.max_z * 255.0), 0, 255).astype(np.uint8)
  q[(np.asarray(z) > 0) & (q == 0)] = 1
  return q[..., None] if q.ndim == 2 else q


# ---------------------------------------------------------------------------
# 4. 奖励（5 Hz，势函数塑形；切换代价不在 env 奖励里，而在 imagination 中解析施加）
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RewardCfg:
  w_nav: float = 2.0         # Δ(−d_xy(base, 当前目标))
  w_ee: float = 3.0          # Δ(−‖p_ee − p_target‖)
  r_grasp: float = 10.0
  r_place: float = 20.0
  r_failed_grasp: float = -1.0
  r_drop: float = -10.0
  r_collision: float = -2.0
  r_time: float = -0.01
  w_smooth: float = 0.01
  w_tilt: float = 0.5        # roll² 惩罚
  switch_cost: float = 0.2   # 由 agent 在 imagination 中施加（agent.switch_cost）


@dataclasses.dataclass
class StepState:
  """env 每个高层步计算一次，供 reward / 标签 / 日志共用。"""
  phase: int                 # Phase
  d_xy_goal: float           # 底盘到当前阶段目标（抓取点或放置点）的水平距离
  d_ee_goal: float           # 末端到当前阶段操作点的距离
  roll: float
  grasp_success: bool = False
  place_success: bool = False
  failed_grasp: bool = False
  dropped: bool = False
  collision: bool = False
  active_action_delta_sq: float = 0.0


def compute_reward(prev: StepState, cur: StepState, cfg: RewardCfg) -> float:
  r = cfg.r_time
  if prev.phase == cur.phase:   # 阶段切换那一步不计势差，避免目标跳变造成的伪奖励
    r += cfg.w_nav * (prev.d_xy_goal - cur.d_xy_goal)
    r += cfg.w_ee * (prev.d_ee_goal - cur.d_ee_goal)
  r += cfg.r_grasp * cur.grasp_success
  r += cfg.r_place * cur.place_success
  r += cfg.r_failed_grasp * cur.failed_grasp
  r += cfg.r_drop * cur.dropped
  r += cfg.r_collision * cur.collision
  r -= cfg.w_smooth * cur.active_action_delta_sq
  r -= cfg.w_tilt * cur.roll ** 2
  return float(r)


TERM_REASONS = ('none', 'success', 'collision', 'fall', 'drop', 'out_of_bounds',
                'too_many_failed_grasps', 'timeout', 'nan')


# ---------------------------------------------------------------------------
# 5. 场景类型与数据划分（含"受限接近方向"测试集）
# ---------------------------------------------------------------------------
class SceneType(enum.IntEnum):
  OPEN = 0          # 物体至少 270° 可接近
  WALL_BACK = 1     # 桌靠墙 → 只能正面
  SIDE_BLOCKED = 2  # 2–3 侧被障碍物挡住
  LOW_FLOOR = 3     # 地面物体 → 需要下蹲/低头
  CORNER = 4        # 房间角落桌角（held-out 类型）
  CUBBY = 5         # 低矮格子柜，只能正面且需下蹲（held-out 类型）


SPLITS = {
    'train':             dict(types=(0, 1, 2, 3), seeds=(0, 10_000)),
    'val':               dict(types=(0, 1, 2, 3), seeds=(10_000, 11_000)),
    'test_iid':          dict(types=(0, 1, 2, 3), seeds=(20_000, 20_500)),
    'test_constrained':  dict(types=(1, 2, 4, 5), seeds=(30_000, 30_500)),  # 4,5 训练中从未出现
    'test_unseen_layout': dict(types=(0, 1, 2, 3), seeds=(40_000, 40_500), layout='large_clutter'),
}


# ---------------------------------------------------------------------------
# 6. 技能先验（由 world model 的 aff / phase 头构造），供 KL[π(skill|z) ‖ prior] 使用
# ---------------------------------------------------------------------------
def skill_prior(aff_pick, aff_place, p_holding, eps: float = 0.05):
  """返回 (..., 3) 概率：未持物 → [1-a_pick, a_pick, 0]；持物 → [1-a_place, 0, a_place]；再与均匀分布混合。
  jax 版本逐字照抄（jnp 替换 np）。"""
  a1 = np.asarray(aff_pick); a2 = np.asarray(aff_place); h = np.asarray(p_holding)
  not_hold = np.stack([1 - a1, a1, np.zeros_like(a1)], -1)
  hold = np.stack([1 - a2, np.zeros_like(a2), a2], -1)
  p = (1 - h)[..., None] * not_hold + h[..., None] * hold
  return (1 - eps) * p + eps / NUM_SKILLS


# ---------------------------------------------------------------------------
# 7. 外部组件接口（Protocol）
# ---------------------------------------------------------------------------
class GoalProvider(Protocol):
  """目标坐标来源。sim：'gt_noisy'（模拟 RGB 教师：σ_pos、延迟、丢失）；real：'rgb_detector'。"""
  def __call__(self, world_state) -> np.ndarray: ...   # -> goal (16,)


class ScriptedExpert(Protocol):
  def reset(self, scene, rng: np.random.Generator) -> None: ...
  def act(self, obs: Dict[str, np.ndarray], priv) -> Dict[str, np.ndarray]:
    """返回已 mask 的动作字典（skill + nav/pick/place，[-1,1]）。priv = 特权仿真状态。"""
    ...


class AffordanceLabeler(Protocol):
  def label(self, sim_snapshot, phase: int, k: int = 4) -> Tuple[float, float]:
    """从快照出发跑 k 次脚本操作（不允许大范围重新导航，只允许 base_adj），返回 (success_rate, 0/1 mask)。"""
    ...


# ---------------------------------------------------------------------------
# 8. 评测指标（纯 numpy）
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class EpisodeMetrics:
  split: str
  scene_type: int
  seed: int
  success: bool
  pick_success: bool
  steps: int
  path_length_m: float
  n_skill_switches: int
  n_grasp_attempts: int
  n_false_triggers: int      # 进入 PICK/PLACE 时 rollout/几何可供性 < 0.2 的次数
  term_reason: str
  deliberation_used: int = 0


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
  s = np.asarray(scores, float); y = np.asarray(labels).astype(bool)
  pos, neg = s[y], s[~y]
  if len(pos) == 0 or len(neg) == 0:
    return float('nan')
  order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
  ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(order) + 1)
  allv = np.concatenate([pos, neg])
  for v in np.unique(allv):          # 平均秩处理并列
    m = allv == v
    ranks[m] = ranks[m].mean()
  return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
  """targets 可为软标签（rollout 成功率）。"""
  p = np.asarray(probs, float); t = np.asarray(targets, float)
  bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
  err = 0.0
  for b in range(n_bins):
    m = bins == b
    if m.any():
      err += m.mean() * abs(p[m].mean() - t[m].mean())
  return float(err)


def reward_head_event_error(r_true: np.ndarray, r_pred: np.ndarray, event_idx: Sequence[int],
                            window: int = 5) -> np.ndarray:
  """抓取刚成功（或放置成功）事件前后 ±window 步的 |r̂ − r| 平均，返回形状 (2*window+1,)。"""
  r_true = np.asarray(r_true, float); r_pred = np.asarray(r_pred, float)
  rows = []
  for t0 in event_idx:
    lo, hi = t0 - window, t0 + window + 1
    if lo >= 0 and hi <= len(r_true):
      rows.append(np.abs(r_pred[lo:hi] - r_true[lo:hi]))
  return np.mean(rows, 0) if rows else np.full(2 * window + 1, np.nan)


def false_trigger(skill_seq: np.ndarray, aff_seq: np.ndarray, thr: float = 0.2) -> int:
  """skill 从 NAV 切到 PICK/PLACE 的时刻，若对应真值可供性 < thr 计一次误触发。"""
  s = np.asarray(skill_seq); a = np.asarray(aff_seq)
  enter = np.where((s[1:] != int(Skill.NAV)) & (s[:-1] == int(Skill.NAV)))[0] + 1
  return int(np.sum(a[enter] < thr))
