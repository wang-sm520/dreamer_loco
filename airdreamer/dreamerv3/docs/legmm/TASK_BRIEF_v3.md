# 任务书：在 AirDreamer 上实现 A2 Pro + AIRBOT Play 深度输入移动操作（v3）

> 本文件交给代码代理（Opus）直接执行。配套文件：`legmm_interfaces.py`（常量、接口与已自测的 numpy 工具）、`legmm.yaml`（配置片段）。
> 背景与文献见 `report_zh.pdf`；模块级改造的原始分析见 `code_roadmap.md`。若与本文冲突，**以本文为准**。
> 仓库中的实际实现与实测数字见 `docs/legmm/PROGRESS.md`；v2 原文备份为 `OPUS_TASK_BRIEF_v2.md`。

## v3 变更记录（2026-11-04，均为用户已确认的决定）

| # | 变更 | 原 v2 | v3 |
|---|---|---|---|
| A | **训练方式** | 离线三阶段（wm → bc → rl）+ 在线微调 | **与 AirDreamer 相同的在线训练**：从零开始，智能体在环境中边交互边学，世界模型与策略同时训练；**不用离线数据、不用预训练世界模型、不用 BC**（§1 第 3 条、§5.5、§5.7、§8） |
| B | 专家数据的用途 | 训练数据 | 不参与训练；用于验证环境可解、构造基线 B1/B2、可供性标签与强裁判的审计（§7） |
| C | 标签审核 | 多模态大模型审核 | **模拟器强裁判**（`tools/legmm_judge_oracle.py`）；gpt-6-astra 在 40 个送审样本上与强裁判一致率仅 44%/58%，不参与可行性判断与剔除（§7.3） |
| D | 机身姿态 | `height` + `pitch` 指令 | 只控制 `height`（0.08–0.385 m，下限为腹部贴地，上限为 UniLab A2 策略零速度实测站高），**不控制俯仰** |
| E | 相机 | 两路 D435i | 头部 D435i（与原装相机重合，base_link (0.35, 0, 0.08)，深度流 848×480，min_z≈0.195 m）；腕部 **D405**（AIRBOT 相机支架，min_z≈0.07 m） |
| F | 机械臂 | 占位参数 | AIRBOT Play 运动学与关节限位（肩高 0.117、上臂 0.270、前臂 0.290、腕到 TCP 0.241 m，夹爪开度 8 cm），安装在背上脖子后方 (0.03, 0, 0.095)（待实测） |
| G | 物体 | 小方块 / 圆柱 | **70% 立起的扁盒**（厚 3–5 cm、宽 10–14 cm、高 12–15 cm，只能从两个窄面水平夹取）+ **30% 圆柱**作对照；**禁止俯抓** |
| H | CUBBY | 高 0.30–0.40 m，物体在柜内 8–18 cm | 高 0.45–0.55 m，物体在柜内 3–8 cm（原尺寸所有布局都不可完成） |
| I | 训练/验证布局 | 不筛选 | 只保留专家规划器能规划出抓取的布局（保持场景类型与物体形状比例）；测试划分保持原始布局 |

---

## 0. 一句话目标与工作规则

**目标**：在 `airdreamer-main/airdreamer/dreamerv3` 中实现一个多技能（NAV / PICK / PLACE）、仅用深度输入的腿足移动操作智能体。流程为：**与 AirDreamer 相同的在线训练**（DreamerV3：世界模型与想象中的 actor-critic 同时在线训练，从零开始）→ 在多个测试划分上评测，其中包括"受限接近方向"划分。脚本专家及其数据只用于验证、基线和标签审计，不参与训练。

**工作规则（必须遵守）**
1. **先在纯 numpy 的运动学替身后端（`backend: kin`）上跑通全链路**（M1–M5），之后再接 Isaac Lab（M6–M7）。在替身上没有通过验收之前，不要动 Isaac Lab。
2. 所有新增的 agent 开关默认关闭；`drone` / `dronenav` 旧任务的行为和参数量必须完全不变（有专门的回归测试）。
3. 常量与接口只从 `embodied/envs/legmm/interfaces.py`（即 `legmm_interfaces.py`）导入，不要另写一份。
4. 每个里程碑结束时提交一次 git commit，并在 `docs/legmm/PROGRESS.md` 里追加：完成项、验收数字（来自日志或评测 JSON，**不得估算或编造**）、未解决问题。
5. 遇到 §11 列出的"需要用户确认"的事项，停下来报告，不要自行猜测硬件参数。
6. 只要不确定 DreamerV3/elements 的 API，就先读源码（`embodied/jax/outs.py`、`heads.py`、`elements`），不要凭记忆写。

---

## 1. 方案决策表（用户已确认；v3 修改的行已标注）

| # | 质询问题 | 用户决定 | 对设计的影响 |
|---|---|---|---|
| 1 | 多阶段任务怎么表示 | 用"头"实现不同阶段任务 | 离散 `skill` 选择头加三个技能动作头 `nav/pick/place`，共享 actor 主干；world model 新增 `aff`（可供性）头和 `phase`（阶段）头（§4.1、§5） |
| 2 | world model 的作用 | 更长的历史记忆，用于想象比较"现在切还是稍后切" | `batch_length 128`（25.6 s）；评测时加入**想象审议**模块（§5.8），并对其做开/关消融；另设"近距离遮挡记忆"测试（§9） |
| 3 | 离线还是在线 | **v3：与 AirDreamer 相同的在线训练，不用 BC**（v2 为离线预训练加在线微调） | 只有一个训练阶段：`train_phase: online`，从零开始在环境中交互训练（§8）。可供性标签在训练时由环境在线生成 |
| 4 | 是否加 LiDAR | 不加，仅用深度本身就有价值 | 观测中不出现 LiDAR |
| 5 | 够不着低处物体 | 需要下蹲、趴下 | 每个技能动作都带机身 `height` 指令（**v3：不控制 `pitch`**）；新增 `LOW_FLOOR`、`CUBBY` 场景（§4） |
| 6 | 底盘低层 | 在 Isaac Lab 训练 A2 速度跟踪策略，并随机化负载 | M7 单独任务；M1–M6 先用一阶滞后的运动学底盘（§6.3） |
| 7 | 目标坐标来源 | 仿真中暂用 RGB 作为先验教师信号 | `GoalProvider: gt_noisy`，模拟 RGB 教师的噪声、延迟和丢失，策略本身不看 RGB（§4.3） |
| 8 | 可供性标签偏差 | 借助 AI 监管和修正（**v3：定稿为模拟器强裁判**） | 标注器采用多方案搜索；分歧样本由强裁判（同一模拟器、同一规则、更全面的搜索）复核；**审核只能剔除样本，不能直接覆写标签**（§7.3） |
| 9 | 想象步长太短 | 高层降到 5 Hz，`imag_length` 改为 30 | 30 步 = 6 s |
| 10 | 持物状态的奖励歧义 | 夹爪开度必须可观测；检查"抓取刚成功"前后奖励头的误差 | `proprio` 中包含 `grip_width` 和 `grip_effort`；评测时输出事件对齐误差曲线（§9） |
| 11 | 深度相机型号 | **v3：头部 D435i，腕部 D405** | 头部 D435i 深度流 848×480，`min_z`≈0.195 m；腕部 D405 装在 AIRBOT 相机支架上，`min_z`≈0.07 m；视场均 87°×58°，带噪声模型（数值待实测，§4.3） |
| 12 | 测试集设计 | 单独划分"受限接近方向"测试集 | `test_constrained` 包含两种训练中从未出现的场景类型 CORNER、CUBBY（§4.5） |

---

## 2. 运行环境与前置检查（M0）

- 假设运行在 Linux + NVIDIA GPU 上（≥24 GB 显存；有两张卡时按 AirDreamer 的方式拆分 policy 和 train 设备）。替身后端用 CPU 多进程跑环境。
- JAX 与 PyTorch（Isaac Lab）同进程时：设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，必要时再设 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`。
- M0 需要完成：
  1. 在现有仓库跑通 `--configs defaults debug --task dummy_...`（或仓库自带的最小冒烟任务），记录 JAX、CUDA、Isaac Lab 版本到 `PROGRESS.md`；
  2. 把 `legmm.yaml` 的片段 A/B/C/D 合并进 `dreamerv3/dreamerv3/configs.yaml`。片段 C 必须放在 `size50m` 等锚点之后（文件末尾）；
  3. 新建目录 `embodied/envs/legmm/`，把 `legmm_interfaces.py` 放进去，命名为 `interfaces.py`，并运行其中的自测逻辑（迁移成 `tests/legmm/test_interfaces.py`）；
  4. 在 configs.yaml 的 defaults 顶层增加片段 D：
     ```yaml
     offline: {dir: '', copy_to_replay: True, eval_every: 5000}
     eval: {episodes: 200, video_episodes: 6, online_aff_rollouts: 4}
     ```

---

## 3. 仓库地图（已核实的集成点）

| 位置 | 现状 | 本任务要做的事 |
|---|---|---|
| `dreamerv3/main.py` L271–L315 `make_env` | ctor 表 + `env_key_alias` + 无人机特判 | 增加 `'legmm': 'embodied.envs.legmm.env:LegMMEnv'`；alias 增加 `'legmm': 'legmm'`；**不要**进入 `suite in ('drone','dronenav')` 的分支 |
| `dreamerv3/main.py` L113–L166 `config.script` 分支 | train / train_eval / eval_only / parallel… | 新增 `train_offline`、`eval_legmm` |
| `dreamerv3/main.py` L174–L200 `make_agent` | 去掉 `log/` 观测与 `reset` 动作 | 无需改动；但它会调用 `make_env(config, 0)` 构造空间，所以 `spaces_only=True` 时 env 必须不启动仿真 |
| `dreamerv3/main.py` `wrap_env` | 对连续动作做 `NormalizeAction` | 动作已在 [-1,1]，保持不变 |
| `dreamerv3/dreamerv3/agent.py` L39 `exclude` | `('is_first','is_last','is_terminal','reward')` | 追加 `config.label_keys` |
| `agent.py` L66–L70 | 每个动作 key 按离散/连续选择分布；MLPHead 为每个 key 建独立输出层 | 天然对应"技能头"，无需改动 |
| `agent.py` L90–L93 `scales` | `rec` 统一作用于所有 dec key | 用 `config.rec_scales` 逐 key 覆盖 |
| `agent.py` L186–L193 | con 损失、逐 key 重建损失 | 加入 aff、phase、bc 损失；`rec_per_pixel_norm` |
| `agent.py` L199–L227 Imagination | `policyfn`、`imagine`、`imag_loss` | 动作屏蔽、切换代价、技能先验 KL；按 train_phase 开关 |
| `agent.py` L421 `imag_loss` | `logpi = sum(v.logp(...) for k in policy)` | 按 skill 屏蔽 logp、熵和平滑项 |
| `embodied/run/train.py` | 在线 driver 循环；L110 无人机任务禁用 parallel | legmm 的 kin 后端使用 `parallel=True` |
| `embodied/core/replay.py` L77/L295/L312 | `add/save/load(directory=…)` | 离线数据集直接存成 replay chunk |
| `embodied/run/train.py` L141–L143 | `from_checkpoint` 加 regex | 用于 offline → online 加载 |

---

## 4. 系统规格（数值全部以 `interfaces.py` 为准）

### 4.1 技能与动作（每个技能对应一个策略输出头）

```
act_space = {reset, skill ∈ {NAV=0, PICK=1, PLACE=2}, nav: (4,), pick: (11,), place: (11,)}   # v3：去掉 pitch
nav   = [vx, vy, wz, height]
pick  = place = [Δee_pos(3), Δee_rot(3), grip, base_adj(vx, vy, wz), height]
```

- 环境只执行当前 `skill` 对应的那个 key。NAV 时机械臂保持：未持物时处于收拢位，持物时处于携物位。PICK/PLACE 时底盘速度限制在 `manip_base_speed_cap`（0.15 m/s）以内。
- **非当前技能的 key 一律置零**。需要置零的位置有：agent 输出给环境之前、imagination 的 `policyfn` 中、scripted expert 写入 replay 之前。三处语义必须一致（`mask_inactive_np` 及其 jax 版本）。这样 world model 看到的动作嵌入与真实执行的动作一致。
- 下蹲和趴下通过 `height` 指令实现（v3：0.08–0.385 m，不控制俯仰；见 `interfaces.BASE_HEIGHT_RANGE`）。

### 4.2 观测

| key | dtype/shape | 说明 |
|---|---|---|
| `depth_head` | uint8 (48,80,1) | D435i 头部深度；0 表示无效，1..255 线性映射到 (0, max_z] |
| `depth_wrist` | uint8 (48,80,1) | 腕部深度（v3：D405）。**必须与 `depth_head` 同分辨率、同 dtype**（Decoder 按通道拼接） |
| `proprio` | f32 (28,) | 见 `PROPRIO_LAYOUT`，包含 `grip_width` 和 `grip_effort` |
| `goal` | f32 (16,) | 抓取点和放置点在机体系下的方向、距离、高度差、朝向和有效位 |
| `label_aff` / `label_aff_mask` / `label_phase` | f32 (2,) / f32 (2,) / i32 | **仅训练用**（v3：在线训练时由环境实时生成）；加入 `agent.label_keys`，不进入 encoder/decoder；部署时填 0 |

注：uint8 的 3 维观测在 Encoder 中会被归为 `imgkeys`，并除以 255 归一化，这是预期行为。

### 4.3 深度与目标教师

- `apply_depth_noise`：σ(z)=a·z²，每回合随机采样 a；深度跳变的边缘随机置空；随机空洞；z<min_z 记为无效（0）。v3：头部 D435i 深度流 848×480，`min_z`≈0.195 m；腕部 D405，`min_z`≈0.07 m（数值均按 datasheet，**需实测标定**）。
- v3：腕部 D405 装在 AIRBOT 相机支架上（TCP 系 (-0.127, 0, 0.1115)，朝夹爪轴线下俯 31.6°），预抓取和抓取时物体都在 min_z 以外。头部相机在机器人贴近桌边时大面积低于 min_z，近距离阶段仍依赖腕部深度和 world model 的记忆。
- `GoalProvider(gt_noisy)`：位置噪声 σ=3 cm，朝向噪声 σ=0.1 rad，延迟 0–2 步，丢失率 5%（丢失时 `*_valid=0`，沿用上一帧值）。

### 4.4 奖励与终止

- 奖励使用 `compute_reward(prev, cur, RewardCfg)`：导航和末端两项势函数塑形，加上事件奖惩。**阶段切换那一步不计势差**，避免目标跳变带来的伪奖励。
- **切换代价不在环境奖励里**，由 agent 在 imagination 中解析施加（§5.6），环境只在日志中记录切换次数。
- 终止原因见 `TERM_REASONS`：success / collision / fall / drop / out_of_bounds / too_many_failed_grasps（3 次）/ timeout（450 步）/ nan。

### 4.5 场景与划分

`SceneType`：OPEN、WALL_BACK、SIDE_BLOCKED、LOW_FLOOR、CORNER、CUBBY。`SPLITS` 中，`test_constrained` 包含训练里从未出现的 CORNER 和 CUBBY，以及 WALL_BACK、SIDE_BLOCKED 的未见种子。场景生成必须由 `(split, seed)` 完全决定，便于复现。

---

## 5. Agent 改造（`dreamerv3/dreamerv3/agent.py`）

所有改动受配置开关控制，开关默认关闭。下面的 jax 伪代码只说明语义，具体 API 以 `embodied/jax/outs.py` 和 `heads.py` 为准。

### 5.1 标签 key 排除
`exclude = ('is_first','is_last','is_terminal','reward') + tuple(config.label_keys)`，enc_space 和 dec_space 都要排除。

### 5.2 逐 key 重建权重与按像素归一化
```python
scales.update({k: config.rec_scales.get(k, rec) for k in dec_space})
# loss() 中：
if config.rec_per_pixel_norm and len(space.shape) == 3:
    losses[key] = recon.loss(sg(target)) / (space.shape[0] * space.shape[1])
```

### 5.3 技能屏蔽（jax 版）
```python
def mask_inactive(act):            # act: dict of (B,[T,]...) 
    if not cfg.skill.enabled: return act
    sk = act[cfg.skill.key]
    for i, k in enumerate(cfg.skill.action_keys):
        act[k] = act[k] * (sk == i)[..., None].astype(act[k].dtype)
    return act
```
在 `policy()` 采样之后调用，并在 imagination 的 `policyfn = lambda f: mask_inactive(sample(self.pol(...)))` 中调用。

### 5.4 world model 辅助头
- `self.aff`：MLP，以 `feat2tensor(repfeat)` 为输入（**允许梯度回传进 world model**，与 SkyDreamer 用特权信息监督 world model 的思路一致），输出 2 个 logit。损失为 `mask * BCE(sigmoid(logit), soft_label)`，按 mask 求均值，乘以 `aff_head.scale`。软标签 BCE 如果 `outs.Binary` 不支持，就自己实现。
- `self.phase`：MLPHead，3 类 categorical，损失为 `phase.loss(label_phase)`。

### 5.5 BC 损失（**v3 不使用**；代码已实现并保留，`agent.bc.scale` 恒为 0，仅供可选对照实验）
在时刻 t 实际执行的动作是 `prevact[:, t+1]`（`loss(carry, obs, prevact, …)` 中 prevact[t] 是到达 obs[t] 之前执行的动作）：
```python
pol = self.pol(sg(self.feat2tensor(repfeat)), 2)
a = tree_map(lambda x: x[:, 1:], prevact); P = tree_map(lambda d: d[:, :-1] ...)  # 对齐到 t ∈ [0,T-1)
lp = P[skill].logp(a[skill]) + Σ_i (a[skill]==i) * P[key_i].logp(a[key_i])
losses['bc'] = -lp * bc_scale(step)
```
**这里的对齐必须写单元测试**：构造合成序列，验证 BC 拟合的是"在 obs[t] 处执行的动作"。

### 5.6 `imag_loss` 修改
1. **屏蔽 logp、熵和平滑项**：
   ```python
   sk = act[skill_key][:, :-1]
   logpi = policy[skill_key].logp(sg(act[skill_key]))[:, :-1]
   ent   = policy[skill_key].entropy()[:, :-1]
   for i, k in enumerate(action_keys):
       on = (sk == i).astype(f32)
       logpi += on * policy[k].logp(sg(act[k]))[:, :-1]
       ent   += on * policy[k].entropy()[:, :-1]
   # actsmooth 只对当前技能的 key 计算
   ```
2. **切换代价**：`imgact[:, t]` 是在想象状态 t 执行的动作（见 L204–L212 的拼接方式），`rew[:, t]` 是到达状态 t 时的奖励。在状态 t 由 skill(t−1) 切换到 skill(t) 的代价应记到 `rew[:, t+1]`：
   ```python
   sw = (act[skill][:, 1:] != act[skill][:, :-1]).astype(f32)    # (N, H)，下标 j 对应 t=j+1
   rew = rew.at[:, 2:].add(-cfg.switch_cost * sw[:, :-1])
   ```
   **必须写对齐单元测试**，用手工构造的 3 步序列验证。
3. **技能先验 KL**：
   ```python
   aff = sigmoid(self.aff(inp))            # (N,H+1,2)，sg
   hold = softmax(self.phase(inp))[..., 1] # p(HOLDING)，sg
   prior = skill_prior(aff[...,0], aff[...,1], hold, eps)   # 与 interfaces.skill_prior 同式
   kl = Σ_c π_c (log π_c − log prior_c)    # π = policy[skill] 的类别概率
   losses['skill_prior'] = sg(weight[:, :-1]) * kl[:, :-1] * prior_scale(step)
   ```
4. 退火系数 `bc_scale(step)`、`prior_scale(step)` 在 `[scale, scale_end]` 之间按 `anneal_steps` 线性退火。step 用 agent 内部计数器（nj.Variable），每次 `train()` 自增。

### 5.7 `train_phase` 门控
**v3 只使用 `online`**（与 AirDreamer 相同：世界模型、辅助头、想象中的 actor-critic 同时训练，从随机初始化开始）。`wm` / `bc` / `rl` 三个离线阶段的代码保留，仅供可选对照实验。

| phase | world model 损失 | aff/phase | bc | imagination（policy/value/prior） | repval |
|---|---|---|---|---|---|
| **online（v3 主流程）** | ✔ | ✔ | ✘（scale 0） | ✔ | ✔ |
| wm（仅对照） | ✔ | ✔ | ✘ | ✘ | ✘ |
| bc（仅对照） | ✔ | ✔ | ✔ | ✘ | ✘ |
| rl（仅对照） | ✔ | ✔ | ✔（作正则） | ✔ | ✔ |

v3 中技能先验 KL（§5.6 第 3 条）的系数：初值 1.0，按 agent 更新次数在前 30 万次内线性退火到 0.1（初值，可调）。从零开始时可供性头还不准，先验的作用随它变准而增强，退火只是防止先验压过回报。

### 5.8 想象审议（仅评测时，`deliberation.enabled`）
在 `policy(mode='eval')` 的 jit 内部：当提议的 skill 与上一步不同，或技能概率中最大两项的差小于 `margin` 时，从当前后验 `dyn_carry` 出发，对每个延迟 d∈{0,5,10}（即 0/1/2 s）复制 M=8 份，想象 H=30 步：前 d 步强制保持旧技能，之后使用新技能，其余动作由策略采样。评分为 Σγ^t r̂ + γ^H V̂，取均值最高的方案；若最优 d>0，本步保持旧技能。用 `jnp.where` 选择输出（始终计算，避免在 jit 中写分支）。把每次审议的 (d*, 各方案得分) 写入 `log/` 键。

---

## 6. 环境（`embodied/envs/legmm/`）

### 6.1 `env.py: LegMMEnv(embodied.Env)`
- 签名为 `LegMMEnv(task, **kwargs)`：`--task legmm_mm` 会被 `make_env` 拆成 suite=`legmm`、task=`mm`，task 作为第一个位置参数传入（可预留给不同任务变体，如 `mm` / `navonly` / `pickonly`）。构造参数取自 `config.env.legmm`；`backend` 取 `kin` 或 `isaac`；`spaces_only=True` 时只返回空间，不创建仿真。
- `obs_space` 和 `act_space` 直接调用 `interfaces.make_obs_space()`、`make_act_space()`。
- `step(action)`：处理 reset → 对动作做 mask（防御性，再做一次）→ 缩放到物理量 → 后端推进 `DECIMATION` 个物理步 → 计算 `StepState` 和奖励 → 标签（若启用）→ 组装观测。

### 6.2 `kin_backend.py`（M1，纯 numpy）
- **世界**：8×8 m 房间，墙体；障碍物和桌子用 AABB 盒；CUBBY 由 5 块板组合而成；物体为竖直的盒体或圆柱；放置区是桌面或地面上的矩形。
- **底盘**：位姿 (x, y, yaw) 加姿态 (h, pitch)，速度和姿态都按一阶滞后跟踪指令（τ 取自 config）；足迹矩形与 AABB 相交即判碰撞。
- **机械臂**：末端目标在机体系内裁剪到可达壳层 `[min_reach, reach]`（以肩部为中心，肩部高度随机身 h 和 pitch 变化），且不低于地面 2 cm。用两连杆竖直平面 IK 近似肘部位置（连杆长度为占位值，标 TODO(calibrate)）。连杆上取若干采样点，与 AABB 做碰撞检测。
- **深度渲染**：向量化 ray–AABB slab 求交，加上 ray–圆柱、地面；**把机械臂连杆近似为盒体，一并渲染进头部深度**，这样才会出现自遮挡，记忆测试才有意义。每路相机 3840 条射线，之后接 `apply_depth_noise` 和 `depth_to_uint8`。
- **可接近方向**：以物体为中心，按 8 个水平方向 bin 外加俯视方向检查；若 0.25 m 内无障碍，则该方向可接近。
- **抓取判定**：夹爪由开转合时，若末端距抓取点 < 3 cm（加上半尺寸），末端接近轴与某个可接近方向的夹角 < 35°，且无碰撞，则以 0.92 的概率成功；否则计为 `failed_grasp`。
- **持物**：物体附着在末端上，发生碰撞即掉落。**放置**：张开夹爪时物体底部距支撑面不超过 3 cm，且 xy 落在放置区内，则成功；否则掉落。
- **性能目标**：单进程每秒 ≥200 个高层步（M1 验收时给出实测值）。

### 6.3 `isaac_backend.py`（M6，Isaac Lab）
- 采用 DirectRLEnv；在一个场景中批量放置 N 个环境，对外通过 `BatchedLegMM` 暴露 N 个 `embodied.Env` 视图（可参考 `omni/arena_manager.py`）。
- **资产**：A2 和 AIRBOT Play 的 URDF/USD 需由用户确认来源（§11）。拿不到时，先用尺寸和质量接近 A2 的占位四足（重量约 37–42 kg、站立尺寸 820×440×570 mm），并在 PROGRESS 中标注。
- **底盘**：阶段 A 为"运动学底盘"，每个物理步按指令积分并写回 root pose，腿保持固定姿态；阶段 B（M7 之后）换成学到的 locomotion 策略，频率 50 Hz。
- **机械臂**：Isaac Lab 的 `DifferentialIKController`，频率 50 Hz，输出关节位置目标；夹爪为平行夹爪的关节位置控制。
- **深度**：TiledCamera 的 `distance_to_image_plane`，**只在 5 Hz 的高层步上渲染**；噪声模型从 `apply_depth_noise` 移植成 torch 版本。
- **标签快照**：在单独划出的一组标注用环境上，用 `write_root_state_to_sim` / `write_joint_state_to_sim` 写入快照，再并行跑 K 次 rollout。

### 6.4 M7：A2 locomotion 子任务（独立的 Isaac Lab 任务）
- 指令为 (vx, vy, wz, height, pitch)。机背上的臂按随机关节轨迹运动，作为扰动；末端负载在 0–1.5 kg 之间随机，臂质量 ±10%，质心随机偏移。观测为本体状态，动作为 12 个关节的 PD 目标，频率 50 Hz，用 Isaac Lab 自带的 PPO 训练。
- 验收：速度跟踪 RMS < 0.1 m/s，高度误差 < 2 cm，俯仰误差 < 3°，臂运动时不摔倒；在下蹲和趴下极限姿态下站稳 10 s。

---

## 7. 专家、标注与审核（`embodied/envs/legmm/`）

> **v3**：专家与离线数据**不参与训练**，用途为：①验证环境与布局可解（专家验收、训练/验证布局筛选）；②构造基线 B1/B2（专家的导航与操作控制器 + 启发式切换）；③可供性标签与强裁判的审计、aff 头校准评测的参考集。在线训练时，可供性标签由环境在交互中实时生成：`LegMMEnv` 在 `labels.enabled` 时自己挂上 `AffordanceLabeler`（目前只有采集器会挂，需补上），按 §7.2 的规则每 2 步、距离 2 m 内标注一次。标注会使环境变慢约 20 倍，但 32 个环境进程时瓶颈仍在 GPU 训练。

### 7.1 `expert.py: ScriptedExpert`
- **NAV**：在 5 cm 栅格、按 0.3 m 膨胀的占据图上做 A*，用 pure pursuit 跟踪，目标是"预操作位姿"。预操作位姿的候选：沿可接近方向、半径 0.35–0.55 m、朝向物体、无碰撞；按可达性评分选择最优。
- **多样性注入**（这部分是数据集的核心价值）：
  - 以概率 0.3 选择**坏位姿**：错误侧面、过远、过近或朝向偏差大；
  - 以概率 0.2 **提前**切到 PICK；
  - 每回合以概率 0.5 加入 DART 风格的动作噪声（σ=0.1，归一化单位）；
  - 10% 的随机探索片段；
  - 抓取失败后后退并重新规划，部分情况下会换到另一侧。
- **PICK/PLACE**：预抓取点（沿接近方向后退 10 cm）→ 推进 → 闭合 → 抬起 10 cm → 收回到携物位。物体高度低于阈值时，按查表结果设定下蹲和俯仰姿态。
- 输出必须是已经过 mask 的、[-1,1] 区间内的动作字典（用 `unscale_action`）。

### 7.2 `labeler.py: AffordanceLabeler`
- 在 TO_PICK 阶段（或 HOLDING 阶段）且 d<2 m 的状态上，每 2 步做一次标注：从快照出发跑 K=4 次脚本操作（只允许 `base_adj`，不允许重新导航），统计成功率，写入 `label_aff[0]`（或 `[1]`），并把对应的 mask 置 1。
- 同时计算几何启发式可供性 `aff_geom`（可达壳层、接近角 < 60°、直线路径无碰撞），只写日志，不作为训练标签，供 QA 和基线使用。

### 7.3 `tools/label_qa.py`：标签审核（**v3 定稿：判定者为模拟器强裁判** `tools/legmm_judge_oracle.py`，与标注器同一模拟器、同一动作约束、搜索更全面；多模态大模型不参与可行性判断与剔除）
1. **筛选**：满足 `|aff_rollout − aff_geom| > 0.5`，或 0.25 < aff_rollout < 0.75（高方差）的样本。
2. **审核包**：俯视示意 PNG（画出机器人、物体、障碍物和可接近方向）、两路深度 PNG，以及状态 JSON。
3. **判定**：调用可插拔的 `judge_fn(packet) -> {verdict: feasible|infeasible|uncertain, confidence, reason}`；v3 使用 `legmm_judge_oracle:judge`（从旁路表精确复原状态后，用强裁判重试）。
4. **规则**：判定与 rollout 标签矛盾且 confidence ≥ 0.8 时，把该样本的 `label_aff_mask` 置 0，即剔除；**禁止用判定结果覆写标签值**。全部决定写入 `qa_log.jsonl`。
5. 输出 QA 报告：筛出的比例、剔除的比例、20 个随机示例。
6. 在线微调之后再做一轮：以 critic 估计（强制 PICK 与强制 NAV 的价值差）作为第三方信号，重复步骤 1–5。

### 7.4 `collect.py`
- 多进程并行；直接调用 `embodied.replay.Replay(..., directory=out).add(step, worker)`，最后 `save()`。
- **transition 布局必须与 Driver 一致**：第 t 条 = `{**obs_t, **act_t, 'reset': False}`，其中 act_t 是**看到 obs_t 之后**执行的动作。agent 内部用 `prevact = prepend(carry, data[:, :-1])` 构造上一步动作（已核实：`agent.py` L356–L357）。如果写错一位，world model 的动作条件和 BC 会同时错位。
- 第一版数据：train 划分 **30 小时 = 54 万高层步**。uint8 深度下每步约 15 KB，总量约 8 GB。
- 同时输出 `dataset_stats.json` 和图表：各技能占比；含失败的回合比例（目标 25–40%）；各场景类型的专家成功率；物体周围底盘位姿的覆盖图（按 aff 着色）；标签分布。

---

## 8. 训练（v3：与 AirDreamer 相同的在线训练）

使用仓库原有的在线训练脚本 `embodied/run/train.py`（`--script train`），与 AirDreamer 训练无人机的方式相同：多个环境进程并行交互，数据进 replay，按 `train_ratio` 训练世界模型、辅助头和想象中的 actor-critic，全部从随机初始化开始。不读取离线数据，不加载预训练检查点，没有 BC 损失。

```bash
# 0. 冒烟（几分钟，CPU 也可）
python dreamerv3/main.py --configs legmm size1m legmm_debug --script train --logdir ~/logdir/legmm/debug
# 1. 在线训练（主方法）
python dreamerv3/main.py --configs legmm --script train --logdir ~/logdir/legmm/online \
  --jax.policy_devices 0 --jax.train_devices 0
# 2. 基线 B4：普通 DreamerV3（同样在线；关闭技能先验、切换代价和辅助头，见 §9）
python dreamerv3/main.py --configs legmm legmm_plain --script train --logdir ~/logdir/legmm/b4_plain
# 3. 评测
python dreamerv3/main.py --configs legmm --script eval_legmm --run.from_checkpoint <ckpt> \
  --env.legmm.split test_constrained --eval.episodes 200 --agent.deliberation.enabled True
```

配置要点（初值，可调）：`run.envs 32`、`run.train_ratio 32`（AirDreamer 无人机为 64）、`run.steps 3e6`；`agent.train_phase online`、`agent.bc.scale 0`；技能先验 KL 按 §5.7；`env.legmm.labels.enabled True`（在线生成可供性标签）；训练和验证布局使用规划器筛选（`scene_filter: solvable`）。`legmm_plain` 预设需新增。

（实际 CLI 参数名以仓库的 elements 解析规则为准；命令写好后加入 `scripts/legmm/*.sh`。）

---

## 9. 评测（`eval_legmm`）

**每个回合**写一条 `EpisodeMetrics` 到 JSONL；**每个划分**汇总为表格（均值 ± 95% bootstrap 置信区间），回合数 ≥200。

| 指标 | 计算方式 |
|---|---|
| 全任务成功率、抓取成功率 | `log/success`、`log/pick_success` |
| 误触发率 | `false_trigger(skill_seq, aff_rollout_seq)`；评测时在线对进入 PICK/PLACE 的状态做 K=4 次 rollout 得到真值 |
| 切换次数（抖动） | 每回合技能切换次数 |
| 时间与路径长度 | — |
| aff 头校准 | AUROC 和 ECE（相对 rollout 真值），分划分报告 |
| **奖励头事件误差** | `reward_head_event_error`，在抓取成功和放置成功前后 ±5 步，画 \|r̂−r\| 曲线（用户第 10 条要求） |
| phase 头准确率 | 同样做事件对齐 |
| world model 开环误差 | 深度 1/5/15/30 步预测的 MAE（只算有效像素），以及跨划分对比 |
| 近距离记忆测试 | 在 d<0.3 m 时把 `depth_wrist` 和 `depth_head` 置为全无效 1 s，比较抓取成功率的下降幅度：本方法 vs 无记忆对照（v3：同一模型在评测时每步清零 RSSM 的确定性状态，只依赖当前帧；原 v2 为帧堆叠 MLP-BC） |

**基线**（都在同一个环境中运行）：
- B1：距离阈值状态机（d<0.75 m 即切换，不重新调整站位）；
- B2：B1 加朝向条件；
- E（参考上限）：脚本专家（使用特权状态，非公平对照，只用于标定任务难度）；
- B4：**普通 DreamerV3 在线训练**，即 AirDreamer 原样做法：同一环境与动作空间，关闭 aff/phase 头、技能先验与切换代价；
- （v2 的 B3“仅 BC”在 v3 中取消）；
- B5（可选）：PPO，观测和动作与本方法相同。

**消融**：无 aff 头；无技能先验；无切换代价；关闭审议；无腕部深度；size50m vs size100m。可选对照：离线预训练路线（v2 的 wm → bc 检查点已训练完成，位于 `~/logdir/legmm/m3/{wm,bc}`），用于回答“离线预训练是否有帮助”。

**汇报格式**：每个划分 × 每种方法一行，列为上表指标；另附图：成功率柱状图、误触发率柱状图、奖励头事件误差曲线、aff 校准曲线，以及每个划分的 6 个典型回合视频（头部深度、腕部深度、俯视示意，并叠加技能和 aff 曲线）。

---

## 10. 里程碑与验收标准

| M | 内容 | 验收（全部用实测数字写入 PROGRESS.md） |
|---|---|---|
| M0 | 环境、配置合并、接口自测 | 冒烟任务通过；`drone_nav` 的 agent 参数量与改动前一致（回归测试） |
| M1 | kin 后端加 LegMMEnv | 单测通过（见下方清单）；吞吐 ≥200 步/s/进程；随机策略不崩溃 |
| M2 | 专家、标签、QA、采集（v3：数据不参与训练，见 §7） | 专家成功率：OPEN ≥85%，WALL_BACK / SIDE_BLOCKED / LOW_FLOOR ≥70%；aff_geom 对 rollout 标签的 AUROC ≥0.8；30 h 数据和统计图齐全 |
| M3 | **在线训练（v3 主流程）** | 训练过程无 NaN、无性能崩塌；val 上：aff AUROC ≥0.85，ECE ≤0.08（相对在线 rollout 真值）；phase 准确率 ≥0.95；深度 15 步开环 MAE 有报告；奖励头事件误差曲线有报告；在训练预算内 val OPEN 成功率 ≥50%（初定，可调） |
| M4 | 基线与消融 | 在 `test_constrained` 上误触发率低于 B1；与 B4（普通 DreamerV3 在线）相比成功率不低、误触发率更低；无 aff 头 / 无技能先验 / 无切换代价三组消融有报告 |
| M5 | 完整评测 | §9 的全部表格和图；审议开/关消融；近距离记忆测试 |
| M6 | Isaac Lab 后端（运动学底盘） | 同一接口通过 M1 的单测；在 Isaac Lab 中在线训练，复现 M3–M4 的趋势 |
| M7 | A2 locomotion 策略加接入 | §6.4 的验收；替换运动学底盘后重新评测 |

**单元测试清单**（放在 `tests/legmm/`，用 pytest）：
- `test_interfaces`：迁移自 interfaces 的自测；
- `test_depth_render`：已知距离的盒体对应正确的像素值；min_z 以内为无效；uint8 往返误差 ≤ max_z/255；
- `test_mask_roundtrip`：专家动作、agent 动作、replay 中存储的动作，三者 mask 一致；
- `test_reward_potential`：专家回合的累计奖励大于随机回合；阶段切换那一步没有势差跳变；
- `test_switch_cost_alignment`：用合成序列验证 §5.6 的对齐（`test_bc_alignment` 保留，供对照实验使用）；
- `test_label_exclusion`：encoder 和 decoder 的 key 中不含任何 label key；
- `test_offline_replay_roundtrip`：写入 chunk → `load` → 取 batch，形状和 dtype 正确；
- `test_backward_compat`：`drone_nav` 配置构造出的 agent 参数量和损失 key 集合与改动前一致；
- `test_split_determinism`：同一 (split, seed) 生成的场景完全相同；CORNER 和 CUBBY 不出现在 train 中。

---

## 11. 必须停下来问用户的事项

1. A2 Pro 和 AIRBOT Play 的 URDF/USD 来源，以及机械臂在机背上的安装位姿；
2. A2 的 `height` 和 `pitch` 实际可达范围，即趴下和低头的极限；
3. 两路 D435i 的实际安装位姿，以及所用分辨率下实测的 min_z；
4. 可用 GPU 的数量和型号，Isaac Lab 和 Isaac Sim 的版本；
5. AI 审核所用的多模态模型及其调用方式（`judge_fn`）；
6. 如果 M2 的专家成功率达不到验收线，先报告失败原因的分布，再决定是调整专家还是放宽场景。

## 12. 每次汇报的格式

```
[Mx 完成情况] 已完成 / 未完成项
[验收数字] 表格（每个数字注明来源：日志路径或 JSON 路径）
[与任务书的偏离] 每条一句，并说明原因
[阻塞与需要用户决定的问题]
[下一步]
```
