# AGENTS.md — LegMM v4（Codex 执行规则）

## 项目

项目是 LegMM v4 主动夹取：A2 四足机器人加 AIRBOT 机械臂，在室内走到桌边，自己选择从哪一面接近、站在哪里，然后触发一个冻结的夹取技能把物体抬起来。不做放置。

- **上层**：DreamerV3 在线训练。动作是 `nav=(vx, vy, wz)` 加离散的 `trigger`，预设为 `legmm_ag`，环境版本 `legmm_ag_v1`。
- **下层**：冻结的夹取技能，在 Isaac Lab 里训练，改造自 `depth-mani/airbot_rl`。kin 阶段先用代替模型。
- 两层网络独立。v3 的 skill 头、放置、BC 先验全部关闭。

## 开工前先读

1. 本次任务文件。Opus 每次派活都会单独给一份，它和本文件冲突时以任务文件为准。
2. `OPUS_TASK_BRIEF.md`：v4 任务书，是设计决定的唯一依据。
3. 最新的 `HANDOFF_*.md`（目前是 `HANDOFF_v4_20261113.md`，它接在 `HANDOFF_v4_20261109.md`、`HANDOFF_v4_20261105.md` 后面，三份都要读）：当前状态和踩过的坑。

**已作废，不要照做**：`OPUS_TASK_BRIEF_v2.md`、`OPUS_TASK_BRIEF_v3.md`、`HANDOFF_legmm.md`、`PROGRESS_legmm.md`，以及 air-5090 仓库里的 `docs/legmm/*`（停在 v3）。只有任务书里写明“见 v3”的部分可以查。

## 执行方式：Mac 本地 Codex，经 `ssh bdi` 操作（用户 2026-11-12 定）

你在用户的 Mac 上运行。从 2026-11-12 起，代码、训练和评测都在 **bdi** 上做，所有远程操作都通过 `ssh bdi`。air-5090 只作为旧数据的只读来源。

**禁止**
- 在 bdi 或 air-5090 上启动 `codex` 或任何 agent CLI；不运行 `~/logdir/legmm/codex/v4_dispatch.sh`。
- 把仓库拷到本地改完再整体同步回去。bdi 上的仓库是唯一事实来源。
- **bdi 是共用账号**：
  - 不改全局 git 配置（全局用户是别人的）、`~/.bashrc`、conda 的 base 环境；
  - 不碰 `~/wang-sm/` 以外的文件；
  - 不碰别人的进程。

**推荐做法**
- 读文件：`ssh bdi "sed -n '1,200p' <path>"`。
- 改文件：在本地生成 patch，`scp` 过去后在仓库里 `git apply`；整文件就在本地写好再 `scp` 覆盖。不要在 ssh heredoc 里嵌套单引号。
- 改完用 `ssh bdi "cd <仓库> && git diff --stat"` 确认改对了。
- 长任务（训练、全量测试）在 bdi 上用 `nohup … > <log> 2>&1 &` 或 `tmux` 跑，本地只轮询日志，确保 ssh 断开不会杀掉任务。
- 查进程用 `pgrep -f "patter[n]"`；不要用 `pkill -f`，它会把自己的 ssh 命令也杀掉。
- Python 用 `/home/bdi/miniconda3/envs/dreamer_loco/bin/python`，不要依赖 `conda activate`。
- **GPU 是共用的**：bdi 有 4 张 RTX 4090 D（24 GB），常被别人占着一部分。
  - 用之前先 `nvidia-smi` 看每张卡还剩多少显存，用 `CUDA_VISIBLE_DEVICES` 指定一张卡；
  - JAX 要设 `XLA_PYTHON_CLIENT_MEM_FRACTION`（或 `XLA_PYTHON_CLIENT_PREALLOCATE=false`），别把整张卡预占掉；
  - 实际占用写进报告。
- **磁盘紧张**：bdi 根分区约 96% 满，2026-11-12 剩约 237 GB。上次一个训练的 replay 就有 38 GB。新训练前先查余量，在报告里写出预计占用。
- air-5090 和 bdi 之间不能直连，要传数据只能经 Mac 中转（`ssh air-5090 'tar …' | ssh bdi 'tar …'`）。

## 机器与路径

| 用途 | 位置 | 权限 |
| --- | --- | --- |
| 仓库 | bdi `~/wang-sm/dreamer_loco`，GitHub 上的 `wang-sm520/dreamer_loco`（公开），当前分支 `legmm-v4-env` | 只在任务指定的分支或 worktree 里改，不在 `main` 上改 |
| 工作目录 | `airdreamer/dreamerv3/`（相对仓库或 worktree） | — |
| 新 worktree | bdi `~/wang-sm/dreamer_loco_worktrees/<名称>` | 新建前要 Opus 同意 |
| 实验输出 | bdi `~/wang-sm/logdir/legmm/v4/<工作包或实验名>/` | 可写 |
| 已迁移的旧结果 | bdi `~/wang-sm/logdir/legmm/v4/` 下现有的目录，如 `pilot/ckpt_s1_4m`、`contract_true_state_20261110` | 只读，新实验用新目录 |
| 任务与报告 | Mac `codex_tasks/`（本目录下） | 可写 |
| air-5090（旧机器） | `~/wang-sm/airdreamer-main` 及其 worktree、`~/logdir/legmm/`、`~/data`、`~/wang-sm/depth-mani` | 全部只读，只用来取没迁移的数据 |
| 规划文档与本地评测结果 | Mac `/Users/amiao/ADY`，评测结果在 `outputs/` | 只在 `outputs/` 下新建 |

新 worktree 和分支同名，用 `legmm-v4-<主题>`，基于任务指定的提交创建。**新建 worktree 前必须先得到 Opus 同意**：在报告里写明名称、基线提交和理由，停下等确认；任务文件里已明确指定的 worktree 视为已同意。

注意：GitHub 和 bdi 上的提交号与 air-5090 上的不一样，因为上传时改写过历史。对应关系是：`legmm-v4-env` 的 `75a848f` 对应 air-5090 的 `721a821`，`main` 的 `fd3ca61` 对应 air-5090 的 `4a8207e`。air-5090 上旧输出里记录的 `git_head` 用的是 air-5090 的提交号。

## 硬规则

1. **用中文**写报告和回复。
2. **数字只能来自实测**，并注明日志或 JSON 的完整路径。不能估算或编造；手算结果要标“手算”。没跑完就写“未实测”。
3. **不提交、不合并、不碰 main**。用户没有明确要求时，不执行 `git commit`、`push`、`merge`、`rebase`。
4. **只读目录不写**：`~/data`、`~/logdir/legmm/m3`、`~/wang-sm/depth-mani`。不删除、不覆盖已有实验输出，新实验用新目录。
5. **数值由用户定**：安全门、奖励、场景范围、成功判据的数值不自行修改。觉得要改时先停下，在报告里写出依据。
6. **不自行换方案**：站立控制器撑不住时停下来问，不自行换成固定底盘；不自行加 BC、先验或信息通道。
7. **回归不破坏**：新增开关默认关闭；`drone` 和 `dronenav` 的行为和参数量不变。
8. **观测里不能有特权信息**：不含 yaw、place 块、q\* 或反事实标签。q\* 只用于奖励。
9. **超出任务范围就停**：发现任务文件和任务书冲突，或需要改任务以外的文件时，停下来写进报告，不要自己判断后继续。

## 分工与工作流

Opus 负责规划和验收，Codex 负责实现和实测，用户负责拍板。你只做任务文件里写的工作包。

1. **接任务**：读任务文件，确认目标 worktree、基线提交、验收标准。有歧义就先问，不要先猜。
2. **实现**：只在指定 worktree 里改。改动要小，每个行为改动都配测试。
3. **自测**：先跑相关测试，再跑全量回归；影响训练路径时，再跑 CPU 和 GPU 冒烟。
4. **交付**：最终回复就是报告，`codex exec -o` 会把它写到 Mac 上的 `codex_tasks/<任务名>_report.md`，报告尾部列出下面的交付清单。不要自己宣布“已验收”。

**报告必须包含**
- 改了哪些文件（`git diff --stat`），每处改动一句话说明为什么。
- 跑了哪些命令，结果是什么（通过数、失败数，失败时贴输出）。
- 每个数字的来源路径。
- 和任务要求不一致的地方、没做完的部分、已知风险。说“无偏离”时必须有对应的测试。

**Opus 验收时会重点查**
- 恒真测试：不要写 `rate >= 0` 这类断言，也不要故意构造一定能过的状态。
- 动作在采样、执行、replay、想象中语义是否一致。
- 势函数 shaping 总和是否恒等于 −Φ(s₀)，改 q\* 权重后是否仍成立。
- 冻结技能是否靠隐藏的底盘或高度调整拿到成功。

## 测试与输出

- 全量回归：在仓库或 worktree 的 `airdreamer/dreamerv3/` 下运行。在 air-5090 上大约要 7–10 分钟，在 bdi 上的耗时还没实测。

  ```
  /home/bdi/miniconda3/envs/dreamer_loco/bin/python -m pytest tests/legmm -q -p no:warnings
  ```

- `configs.yaml` 的正则预设按前缀匹配，新键不要以 `depth`、`units`、`layers`、`rssm`、`bins` 开头。`elements.Config` 不接受空列表，遇到未知键会报错。
- v3 的 replay 和 checkpoint 与 v4 不兼容，不要混用。
- GPU 冒烟时留意 XLA 报错 `Contracting dimension is too fragmented`，出现就完整记录。
- 训练和评测的输出放在 bdi 的 `~/wang-sm/logdir/legmm/v4/<实验名>/` 下。每个输出目录里都要有一个 `status.json`，写明配置、种子、步数、checkpoint 路径和是否跑完；在 bdi 上跑的，还要写明用的是哪张 GPU（`CUDA_VISIBLE_DEVICES`）。
- 拉回 Mac 的评测结果放在 `outputs/eval_<实验名>_<YYYYMMDD>/`。
