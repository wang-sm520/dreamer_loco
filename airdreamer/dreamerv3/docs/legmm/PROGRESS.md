# legmm 进度记录

所有数字均来自日志或 JSON 文件，来源写在每个数字旁边。仓库根目录为 `~/wang-sm/airdreamer-main`（air-5090），以下路径都相对 `airdreamer/dreamerv3/`。

## M0：环境、配置合并、接口自测（2026-11-01）

### 完成项
- 从本地 `airdreamer-main` 同步到 air-5090，执行 `git init`。基线提交为 `a57b89f`，即未做任何改动的原始仓库。
- `legmm_interfaces.py` 放到 `embodied/envs/legmm/interfaces.py`，并新增 `embodied/envs/legmm/__init__.py`。
- `legmm.yaml` 合并进 `dreamerv3/configs.yaml`：
  - 片段 A 放在 `defaults.env.legmm`；
  - 片段 B 放在 `defaults.agent` 末尾；
  - 片段 C 放在文件末尾，位于 size 锚点之后；
  - 片段 D（`offline`、`eval`）放在 defaults 顶层。
  - 合并时做了改动，见“与任务书的偏离”。
- 测试放在 `tests/legmm/`：
  - `test_interfaces.py`：10 项，覆盖布局、空间、动作缩放往返、mask、深度噪声与量化、奖励势函数、skill_prior、各指标函数、数据划分；
  - `test_configs.py`：9 项，覆盖全部 legmm 预设组合；drone 预设的新开关均为关闭；
  - `test_backward_compat.py`：2 项，drone_nav 和 drone_multi。
- 基线签名用 `tests/legmm/drone_signature.py` 在**改动前**的仓库（a57b89f）上生成，写入 `tests/legmm/drone_baseline.json`。
- 冒烟脚本 `scripts/legmm/m0_smoke.sh`。

### 运行环境（实测）
| 项 | 值 | 来源 |
|---|---|---|
| GPU | 1 × RTX 5090，32607 MiB；与其他任务（UniDR、isaacsim-atec51）共用，冒烟时已被占用约 13 GB | `nvidia-smi` |
| 驱动 / CUDA | 580.178.04 / CUDA 13.0（驱动）；JAX 使用 cuda12 插件 | `nvidia-smi` |
| Python | 3.11.15（conda env `airdreamer`） | `python -c "import sys"` |
| JAX | jax 0.6.2，jaxlib 0.6.2，jax-cuda12-plugin 0.6.2，cudnn 9.8.0.87 | `pip list` |
| elements / ninjax / portal | 3.21.0 / 3.6.2 / 3.7.4 | `pip list` |
| PyTorch | 2.7.0+cu128 | `torch.__version__` |
| Isaac Sim | 5.1.0.0（pip，装在同一 env） | `pip list` |
| Isaac Lab | v2.3.2（isaaclab 0.54.2，`~/wang-sm/IsaacLab-v2.3.2`） | `git describe` |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-38，32 核，62 GB 内存 | `lsb_release`、`nproc`、`free` |

### 验收数字
| 项 | 结果 | 来源 |
|---|---|---|
| 冒烟任务：`--configs defaults debug --task dummy_disc`，CPU，`run.steps 3000` | 退出码 0，耗时 43.3 s；记录到 2828 步，5 条训练指标，28 个回合，保存了 1 个 checkpoint | `/tmp/legmm_m0_smoke.log`；`~/logdir/legmm/m0_smoke/metrics.jsonl`、`scores.jsonl` |
| 冒烟最后一次训练损失 | dyn 1.0，rep 1.0，rew 1.60，con 0.66，policy 0.017，value 0.23 | `~/logdir/legmm/m0_smoke/metrics.jsonl` 最后一条 `train/loss/*` |
| drone_nav agent 参数量（改动前） | 312,515,857（354 个张量） | `tests/legmm/drone_baseline.json` |
| drone_multi agent 参数量（改动前） | 576,964,234（366 个张量） | `tests/legmm/drone_baseline.json` |
| 回归测试（改动后与基线比较参数量、参数名哈希、损失 key、损失权重、enc/dec key） | 2/2 通过 | `pytest tests/legmm` |
| `tests/legmm` 全部 | 21 passed | `pytest tests/legmm -q` |
| 原有 `embodied/tests`（test_driver、test_replay） | 基线 a57b89f：61 failed / 10 passed；M0 之后：57 failed / 14 passed。失败原本就存在，因为测试引用了已删除的 `Replay.dataset` 等旧 API；多通过的 4 项来自 Driver 修复 | 在 detached worktree 中分别运行 pytest |

### 与任务书的偏离
1. **片段 A 有 3 个 key 改名**：`depth_dtype` → `obs_depth_dtype`，`domain_rand.depth_noise_a` → `cam_noise_a`，`domain_rand.depth_hole_frac` → `cam_hole_frac`。
   - 原因：size 预设中的 `.*\.depth` 是用 `re.match` 匹配的，只锚定开头，会把所有以 `depth` 开头的 key 一起覆盖。原名在 `size50m` 下会变成 `'32'`，或者抛出 TypeError。
   - 旁证：在现有 drone_multi 训练中，`agent.enc.simple.depth_symlog` 已经因此从 False 被改成了 True，见 `~/wang-sm/airdreamer_runs/drone_multi_full_training2/config.yaml:32`。为了保持 drone 的行为不变，这个现象本身我没有修改。
2. **片段 B 的 `label_keys` 默认值**改为 `[label_aff, label_aff_mask, label_phase]`，原为 `[]`。原因是 elements.Config 禁止空列表。对没有这些 key 的环境，排除操作不起作用，回归测试已验证这一点。
3. **片段 B 的 `rec_scales`** 改成字符串 `'key=w,...'`，默认 `''`。原因有两个：elements 会丢弃空 dict，并拒绝未知 key；而且 dict 写法会产生 `rec_scales.depth_head` 这样的 key，被 `.*\.depth` 覆盖。legmm 预设中的值为 `'depth_head=1.0,depth_wrist=1.0,proprio=2.0,goal=2.0'`。
4. **`interfaces.py` 修了一处 bug**：`apply_depth_noise` 原来对原始 z 做 `prepend`。当第一行或第一列是无命中像素（inf）时，差分为 inf，被当作边缘，以 `edge_hole_prob` 的概率置为无效。现在改为对已填充 max_z 的图像做差分。测试为 `test_depth_noise_far_pixels_stay_far`。
5. **`interfaces.py` 没有自带自测代码**，所以 `test_interfaces.py` 是按各函数的约定新写的，不是迁移来的。
6. **恢复了上游的 `embodied/envs/dummy.py`**，用作冒烟任务。原因是仓库里只剩 ctor 表中的条目，文件本身被删掉了。另外加了 `**kwargs`，因为 `make_env` 总会传入 `logdir`。
7. **修复 `embodied/core/driver.py`**：用于探测无人机共享竞技场的 `getattr(env, '_arena', None)` 在 `embodied.Wrapper` 下会抛 `ValueError`（不是 AttributeError），导致任何非无人机任务在非并行 Driver 下都会崩溃。改为用 `_probe()` 同时捕获这两种异常。无人机任务上有这个属性，所以行为不变。
8. 冒烟命令需要额外传 `--jax.policy_devices 0 --jax.train_devices 0`，因为 defaults 中是 [3] 和 [2]，而本机只有 1 张卡。

### 未解决问题
- **§11 第 4 项**：本机只有 1 张 RTX 5090，并且与其他任务共用，可用显存约 19 GB，低于任务书假设的 ≥24 GB。Isaac Sim 5.1.0 和 Isaac Lab 2.3.2 已装在 `airdreamer` env 里。这些与 M6、M7 相关，需要用户确认是否就用这套配置。
- `run.debug` 的默认值是 True，此时 `train.py` 的 Driver 走非并行路径。任务书要求 kin 后端 `parallel=True`，M1 需要在 legmm 预设里设 `run.debug: False`，或者在 train.py 中按任务判断。

## M1.1：按用户提供的资料校准（2026-11-02）

用户给出的资料：
- 腕部相机照片；
- A2 的 MJCF：`/Users/amiao/unitree_rl_mjlab/src/assets/robots/unitree_a2`；
- AIRBOT Play 的 MJCF：DISCOVERSE。GitHub 在本机和 air-5090 上都下载失败（SSL 拦截和超时），所以改用 air-5090 上已有的 `~/wang-sm/depth-mani/airbot_rl/assets/airbot_play.xml`。它的运动学与 DISCOVERSE 版本相同，另外带了腕部相机支架 `camera_stand.obj`，与照片一致。
- 用户决定：机身高度范围为“初始站立 → 肚子贴地”，不控制俯仰；头部 D435 与原装相机位置重合；M6/M7 先用当前这张 5090。

| 参数 | 取值 | 依据 |
|---|---|---|
| 机身高度（base_link 原点离地） | 0.08–0.374 m | 腹部网格底面在原点下方 0.078 m（`base_link.STL`、`base1_collision`）；站立姿态取 `a2_constants.INIT_STATE`（thigh 0.9、calf -1.8，两段腿长 0.275 m，足端半径 0.032 m）算得 0.374 m |
| 俯仰 | 不再作为指令：`NAV_DIM` 由 5 改为 4，`MANIP_DIM` 由 12 改为 11；proprio 中仍保留 `base_pitch` 观测（kin 中恒为 0） | 用户决定 |
| 头部 D435i | base_link 系 (0.35, 0, 0.08)，水平朝前 | 原装前置相机的圆形模组：面法向加权中心 (0.357, 0, 0.082)，正中法向俯仰约 0° |
| AIRBOT 运动学 | 肩高 0.1172；上臂 0.27009；前臂 0.29015；腕关节到 TCP 0.241；夹爪行程 2 × 0.04 m | `airbot_play.xml`（link1–link6、手指 slide range） |
| 腕部相机（TCP 系，x 为接近轴，z 为支架一侧） | (-0.127, 0, 0.1115)，向夹爪轴线俯 31.6°；光轴在指尖前约 5 cm 处与夹爪轴相交 | `camera_stand.obj` 中相机块做 PCA：中心 (-0.1115, 0, -0.1221)，法向 (0.523, 0, 0.852)（link6 系） |
| 机械臂模型 | TCP 位姿指令 → 腕心 = TCP − 0.241·接近轴 → 腕心裁剪到肩部可达壳层 [0.08, 0.559] → 两连杆 IK（肘部朝上）。碰撞点包括上臂、前臂、腕到 TCP、指尖、腕部相机；与 A2 机身包围盒（来自网格）做自碰撞检测 | — |
| 物体宽度 | 0.04–0.07 m，原为 0.04–0.08 | 夹爪行程 0.08 m；抓取判定新增条件“宽度 < 0.075 m” |

### 校准后的实测
| 项 | 结果 | 来源 |
|---|---|---|
| 单测 | 42 passed | `pytest tests/legmm` |
| 吞吐 | train 577.5 步/s，test_constrained 594.8 步/s；非有限值 0 | `docs/legmm/m1_bench_calib_{train,test_constrained}.json` |
| 预抓取位姿（50 个场景） | 头部被手臂遮挡 1.8%（原为 12.5%）；头部 min_z（0.28）以内的像素 94.7%，原因是机器人站在桌边时，桌子前立面距头部相机不到 0.28 m | `docs/legmm/m1_mount_check_calib.json` |
| 腕部相机到物体的 z 深度 | 预抓取时中位数 0.252 m，抓取时 0.167 m（由几何决定） | 同上 |
| 物体低于 min_z 的比例（预抓取 / 抓取） | min_z 0.28：100% / 100%；0.195：0% / 100%；0.105：0% / 0%；0.07：0% / 0% | 同上 |

### 仍待用户确认
- AIRBOT 底座在 A2 背上的安装位姿。目前 `arm.mount_xyz` 用占位值 (0.10, 0, 0.13)，即背板顶面附近。
- 两路相机的型号、分辨率和实测 min_z。照片中的腕部相机外形（方形、银色、约 4–5 cm）和支架上的相机块尺寸（53 × 53 × 22 mm）更像 D405，不像 D435i（90 × 25 × 25 mm）。按上表，若用 D435i 1280×720（min_z 约 0.28 m），腕部相机在预抓取和抓取时都看不到物体。
- 站立高度 0.374 m 取自 mjlab 的初始关节姿态。如果实机的初始站立更高，需要更正。

## M1.2：用户第二轮答复后的定稿（2026-11-02）

| 参数 | 取值 | 依据 |
|---|---|---|
| 站立高度 | **0.385 m**（原为按关节姿态算出的 0.374 m） | 用 UniLab 中训练好的 `A2JoystickFlat` 策略（`logs/rsl_rl_ppo/A2JoystickFlat/2026-08-20_17-29-21_mujoco/policy.onnx`）实测：零速度指令，关闭推扰、加质量和质心偏移，16 个环境各 10 s，丢弃前 2 s。均值 0.3853 m，各环境均值在 0.3839–0.3867 m 之间，时间方向标准差 1.9e-5 m，没有摔倒。训练目标 `base_height_target` 为 0.40。来源：`docs/legmm/m1_a2_stand_height.json`、`tools/measure_a2_stand_height.py`。 |
| 机身高度范围 | 0.08–0.385 m | 下限为腹部贴地（M1.1） |
| 机械臂安装 | joint1 轴位于 base_link 系 (0.03, 0, 0.095) | 用户描述为“狗背上、脖子后面”。网格中线显示，平背板顶面 z 约 0.09，范围 x ∈ [-0.10, 0.10]；从 x 约 0.10 开始向前升高、收窄成脖子。AIRBOT 底座外壳半长约 0.07 m，所以 x 取 0.03；z 为 0.09 加 5 mm 底板。属于推算值，标 TODO(calibrate)。 |
| 腕部相机 | **D405**：min_z 0.07 m（datasheet），max_z 1.0 m（uint8 量化步长 3.9 mm），视场 87°×58° | 用户确认 |
| 头部相机 | D435i，min_z 0.28 m（1280×720 时的值） | 用户确认型号；所用分辨率尚未给出，按最保守的值处理 |

### 定稿后的实测
- 单测：42 passed。
- 吞吐：train 594.5 步/s（`docs/legmm/m1_bench_final_train.json`）。
- 预抓取位姿，50 个场景（`docs/legmm/m1_mount_check_final.json`）：
  - 头部被手臂遮挡 2.0%；
  - 头部 min_z 以内的像素 94.2%（站在桌边时桌子前立面距相机不到 0.28 m）；
  - 腕部 D405 到物体的 z 深度：预抓取时 0.252 m，抓取时 0.167 m。两者都远大于 0.07，所以物体在**整个接近过程**中都可见。

## M2：专家、标注、QA、采集（2026-11-02；差 QA 收尾）

### 完成项
- `planning.py`：栅格 Dijkstra（等价于从起点对所有候选终点同时做 A*）、视线捷径化、基于投影点的纯追踪。
- `expert.py`：ScriptedExpert，用状态机执行导航、对准、直线接近、操作控制器。操作控制器的流程：航点 → 预抓取 → 推进 → 闭合 → 抬起 → 收回。
  - 规划时对整条航点路径插值检查碰撞（含 1 cm 规划余量）。
  - 抓取后收回有两种方式：边收边转、先原地转再收。
  - 导航带预测足迹安全过滤。
  - 多样性：坏站位 0.3，提前切换 0.2，DART（σ=0.1，约 50% 回合），约 10% 步数为探索片段。噪声不作用于高度维和夹爪维。
- `labeler.py`：从快照出发、只允许 base_adj、K=4 次 rollout；几何启发式 aff_geom（解析臂展加点碰撞，留 0.10 m 余量）。
- `collect.py`、`dataset_stats.py`：replay chunk 加旁路表 `meta/`，生成统计 JSON 和图。
- `tools/label_qa.py`：分三段 packets / judge / apply，判定可续跑、可并行；只剔除样本，不改标签值。
  - judge 有两个实现：`legmm_judge_codex:judge`（air-5090 上的 Codex，gpt-6-astra，按用户要求）和 `legmm_judge_sonnet:judge`（Sonnet 5；air-5090 所在地区被 API 拒绝，返回 403，暂未使用）。
- `tools/legmm_scene_viewer.py`：每类场景一张静态图（3D、俯视、两路深度）和一段专家执行视频，输出在 `docs/legmm/scenes/`。
- 环境规则修订：
  - 持物时物体底部压到支撑面，只阻挡手臂运动，不算掉落；
  - 增加 `drop_reason`；
  - 计入“含失败回合”的事件包括：以失败告终、抓取失败、抓取或放置控制器失败；不包括专家导航层面的重试。
- 测试：`tests/legmm` 共 49 passed。

### 验收数字
| 项 | 结果 | 验收线 | 来源 |
|---|---|---|---|
| 专家成功率（关闭多样性，每类 100 回合） | OPEN **0.96**，WALL_BACK **0.91**，SIDE_BLOCKED **0.86**，LOW_FLOOR **0.83**（CORNER 0.92，CUBBY 0.00） | ≥0.85 / ≥0.70 | `docs/legmm/m2_expert_eval.json` |
| aff_geom 对 rollout 标签（阈值 >0.5）的 AUROC，train | pick **0.862**，place **0.897** | ≥0.8 | `docs/legmm/m2_dataset_stats_train.json` |
| 同上，val | pick 0.843，place 0.903 | — | `docs/legmm/m2_dataset_stats_val.json` |
| 数据量 | train 540,000 步 = 30.00 h，2182 个回合，1.15 GB，用时 968 s（20 进程）；val 54,000 步 = 3.00 h，199 个回合 | 30 h | 同上 |
| 含失败的回合比例 | train **0.336**，val 0.362；其中以失败告终的为 0.186 / 0.226 | 0.25–0.40 | 同上 |
| 技能占比（train） | NAV 0.713，PICK 0.179，PLACE 0.108 | — | 同上 |
| 各场景专家成功率（train，开启多样性） | OPEN 0.889，WALL_BACK 0.795，SIDE_BLOCKED 0.795，LOW_FLOOR 0.782 | — | 同上 |
| 标签数量（train） | pick 100,736 条（34.3% 为正），place 66,757 条（51.5% 为正） | — | 同上 |
| 审核筛选 | 24,511 / 167,493 条标签被选中，占 14.6% | — | `label_qa.py --stage packets` |
| Codex 判定速度 | 约 89 s 每次（3 次顺序调用共 4 分 27 秒） | — | 手动计时 |

图：`docs/legmm/figs/{summary,coverage_pick,episode_length}.png`；场景图和视频在 `docs/legmm/scenes/`。

### 与任务书的偏离
1. 预操作站位的半径范围为 0.45–0.96 m，加 ±0.1/0.2 m 横向偏移和两种朝向；任务书为 0.35–0.55 m。原因是 A2 机身长 0.82 m，站在桌前时机身中心离物体至少约 0.5–0.7 m。
2. A* 用一次 Dijkstra 代替，效果等价；规划先用 0.45 m 膨胀，找不到路径再退回任务书的 0.3 m。
3. 标注器的局部规划允许 ±0.15 m / ±0.12 m / ±0.3 rad 的底盘偏移，即任务书说的“只允许 base_adj”；rollout 最长 30 步，带 σ=0.05 的动作噪声。
4. “含失败回合”的口径如上（不计导航重试），否则会是 0.47。
5. 审核判定改用 Codex gpt-6-astra（用户指定），不用 Sonnet 5（API 在该地区被拒绝）。
6. 环境规则修订见上文。

### 未解决
- **QA**：200 个样本的 Codex 试点正在运行（`~/data/legmm_v1/train/qa/verdicts.jsonl`，日志 `~/logdir/legmm/qa_pilot.log`）。全量 24.5k 条在 8 路并行下约需 3 天。前 3 条判定都和 rollout 相反（rollout 为 0，Codex 判可行，置信度 0.71–0.84），可能需要提高阈值或缩小范围。等试点的一致率出来后由用户决定。
- **CUBBY 专家成功率为 0**：原因见 `docs/legmm/scenes/5_CUBBY.png`。多数候选够不着；完全趴下时前臂会碰机头；腕部相机支架会撞顶板。加入夹爪 ±90° 滚转后仍为 0，已撤回该改动。需要用户决定是否改格子柜的尺寸。
- 机械臂安装位置仍是推算值。
