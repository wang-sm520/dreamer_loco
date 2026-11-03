# legmm 交接文档（2026-11-02 更新）

任务书：`/Users/amiao/Desktop/ADY/OPUS_TASK_BRIEF.md`（本机）。实测数字以 `docs/legmm/PROGRESS.md` 为准；本文只说明当前状态、怎么接着做、有哪些坑。

## 1. 代码在哪里

- **唯一权威副本**：air-5090 上的 `~/wang-sm/airdreamer-main`（git 仓库，分支 `main`，提交身份为 `legmm-agent`）。工作目录是 `airdreamer/dreamerv3/`，下文路径都相对于它。
- Python 环境：`/home/user/miniconda3/envs/airdreamer/bin/python`（jax 0.6.2，torch 2.7，Isaac Sim 5.1，Isaac Lab 2.3.2）。
- 之前的会话在本机临时目录里放了一份镜像，用 rsync 推到远端；那个目录会被清掉，**不要依赖它**。接手时直接在远端改，或者先把远端拉下来再改。
- 提交历史：

| commit | 内容 |
|---|---|
| a57b89f | 原样基线 |
| 9558eea | M0 |
| 8ec7e9e | M1 |
| aef2f3f / b2d279b | M1.1 / M1.2 校准 |
| 9dff6c0 / b36e367 / ab1fa64 | M2 |

## 2. 里程碑状态

| M | 状态 | 说明 |
|---|---|---|
| M0 | ✅ | 配置合并、`interfaces.py`、回归测试；drone 参数量与基线一致 |
| M1 | ✅ | kin 后端 + LegMMEnv；吞吐约 590 步/s/进程；单测通过 |
| M1.1 / 1.2 | ✅ | 按 A2 与 AIRBOT 实际参数校准（见第 4 节） |
| M2 | 🟡 进行中 | 专家、标注器、QA 工具、采集器都已完成，专家验收已过；**30 h 数据正在采集**，还差统计、QA 干跑、汇报 |
| M3–M7 | ⬜ | 未开始 |

### M2 已有的实测数字
- 专家验收（关闭多样性，每类 100 回合，`docs/legmm/m2_expert_eval.json`）：

| 场景 | 成功率 | 验收线 |
|---|---|---|
| OPEN | 0.96 | ≥0.85 ✅ |
| WALL_BACK | 0.91 | ≥0.70 ✅ |
| SIDE_BLOCKED | 0.86 | ≥0.70 ✅ |
| LOW_FLOOR | 0.83 | ≥0.70 ✅ |
| CORNER（测试集独有，无验收线） | 0.92 | — |
| CUBBY（测试集独有，无验收线） | **0.00** | — |

- 开启多样性（每类 30 回合，`m2_expert_eval_diversity.json`）：成功率 0.60–0.83。
- aff_geom 对 rollout 标签的 AUROC：在约 900 步的小样本上，pick 0.93、place 0.87。**正式数字要以 30 h 数据集的 `dataset_stats.json` 为准。**

## 3. 正在跑的任务，以及接下来要做的事

- 数据已经重新采完（专家调过参数），位于 `~/data/legmm_v1/{train,val}`。调参前的旧数据在 `~/data/legmm_v1_pretune`，可以删除。
- **QA 试点正在运行**（Codex gpt-6-astra，200 个样本，8 路并行，约 40 分钟）：
  ```bash
  wc -l ~/data/legmm_v1/train/qa/verdicts.jsonl          # 进度
  python tools/label_qa.py --stage apply --data ~/data/legmm_v1/train \
      --verdicts ~/data/legmm_v1/train/qa/verdicts.jsonl   # 只出报告，不改数据（--apply 默认 False）
  cat ~/data/legmm_v1/train/qa/qa_report.json             # 看 agree_with_rollout、rejected 等字段
  ```
  先看 Codex 和 rollout 的一致率，再和用户确定剔除阈值（目前 0.8）以及审核范围（全量 24.5k 条在 8 路下约 3 天），然后加 `--apply True` 正式应用。
- 然后按 §12 汇报 M2，进入 M3。
- 场景图和视频：`docs/legmm/scenes/`；本机副本在 `~/Downloads/legmm_scenes/`。

## 4. 已经定下来的硬件参数

| 参数 | 取值 | 依据 |
|---|---|---|
| 机身高度 | 0.08–0.385 m | 0.385 m 由 UniLab `A2JoystickFlat` 策略在零指令下实测（`tools/measure_a2_stand_height.py`，在本机 UniLab 用 `uv run` 运行）；0.08 m 为趴下时腹部贴地 |
| 俯仰 | **不作为指令** | NAV_DIM=4，MANIP_DIM=11 |
| 头部 D435i | base_link (0.35, 0, 0.08)，水平朝前 | 与原装相机重合；min_z 暂取 0.28（**分辨率未定**） |
| AIRBOT Play 运动学 | 肩 0.1172，上臂 0.270，前臂 0.290，腕到 TCP 0.241，夹爪开度 0.08 | `~/wang-sm/depth-mani/airbot_rl/assets/airbot_play.xml` |
| 机械臂安装 | (0.03, 0, 0.095) | 由用户描述“背上、脖子后面”结合网格推算，**待实测** |
| 腕部 D405 | TCP 系 (-0.127, 0, 0.1115)，下俯 31.6°；min_z 0.07，max_z 1.0 | `camera_stand.obj` |

## 5. 需要找用户确认的事项

1. ✅ 审核模型：用户先指定 Sonnet 5（API 在该地区被拒绝，返回 403），后改为 **air-5090 上的 Codex gpt-6-astra**，judge 为 `tools/legmm_judge_codex.py`。待定的是阈值和审核范围（见第 3 节）。
2. ✅ 失败回合比例：由我调整。改了统计口径并放宽了两处控制容差；任务书原定的概率保持不变。结果 train 0.336、val 0.362，在目标区间内。
3. ⬜ CUBBY 专家成功率为 0：用户要先看场景，图和视频已提供。待决定：改格子柜尺寸（例如高 0.40–0.50 m，物体放在柜内 0.03–0.10 m），或者保留为“不可完成”的测试场景。
4. ✅ 头部相机为 1920。D435i 的 1920 是 RGB 流，深度流最高 1280×720，min_z 取 0.28。
5. ⬜ 机械臂安装位置的实测值。

## 6. 坑与约定（务必先看）

- **配置正则**：size 预设里的 `.*\.depth` 用 `re.match`，只锚开头，会改写所有以 `depth` 开头的 key。新增配置项的任何一段都不要以 depth、units、layers、rssm、bins 开头。为此 `rec_scales` 用字符串形式 `'k=w,...'`，相机相关 key 也已改名。
- elements.Config 不允许空列表，会丢掉空 dict，并且拒绝未知 key。
- 本机只有 1 张卡，命令行要加 `--jax.policy_devices 0 --jax.train_devices 0`。GPU 和 UniDR 等任务共用，空闲显存约 19 GB。
- legmm 预设设了 `run.debug: False`，Driver 会多进程运行。
- `Replay.load(directory=...)` 要求 replay 本身也有 directory，否则会报缺少 `saved` 属性。
- 离线 chunk 的特点：
  - 多一个 `sample_id`（int64），QA 用它定位样本；
  - 不含 `dyn/deter`、`dyn/stoch` 等 replay_context 条目，因为它们的形状取决于模型尺寸；
  - **M3 的 `train_offline` 必须丢掉 `sample_id`，并用零补齐上下文条目**，因为 `agent.train` 要求 batch 的 key 与 spaces 完全一致；
  - M5 把离线数据并入在线 replay 时也要这样处理。
- 环境规则的几处改动：
  - 持物时物体底部压到支撑面，只会阻挡这一步手臂运动，不算掉落；
  - 机械臂、物体中心碰撞，或者释放时不在窗口内，才算 drop；
  - 掉落原因记录在 `backend.drop_reason`。
- 专家里比较关键的参数：
  - `ARM_GAIN` 0.6，用来抗 1 步机械臂延迟；
  - `PLAN_MARGIN` 0.01（自碰撞时按 ×3 计）；
  - 抓取后的收回分两种方式：先抬起再边收边转，或者先抬起、原地转、再收回；
  - 导航有预测足迹安全过滤 `_safe`；
  - 噪声和探索不作用于高度维和夹爪维。
- 运行 `tests/legmm`：`python -m pytest tests/legmm -q -p no:warnings`，48 项全部通过，约 50 s，其中包括 drone 回归测试。

## 7. M3 准备工作（任务书 §5）

agent 侧的改动都还没做：
- 用 label_keys 做 exclude；
- 每个 key 单独的 rec_scales 和按像素归一化；
- jax 版 mask_inactive；
- aff 头和 phase 头；
- BC 损失，以及对齐单测；
- `imag_loss` 中的屏蔽、切换代价（以及对齐单测）、技能先验 KL；
- train_phase 门控；
- 想象审议；
- `embodied/run/train_offline.py`，以及 `main.py` 中的 `train_offline` / `eval_legmm` 脚本。

目前 `--configs legmm size1m legmm_debug --script train` 能端到端跑通，但 label_* 还会被当作观测重建，这是 M3 第一步要修的。注意 `imag_loss` 里 actsmooth 对所有 policy key 调用 `.pred()`；离散的 skill 头要在屏蔽逻辑里单独处理。

## 2026-11-03 决定

- 标签审核定稿为强裁判（`legmm_judge_oracle:judge`，与原标注器同一模拟器、同一规则，搜索更全）。gpt-6-astra 在 40 个送审样本上与强裁判一致率仅 44%（原提示词）/ 58%（加几何信息），不用于可行性判断和剔除。详见 PROGRESS「M2 收尾」。
