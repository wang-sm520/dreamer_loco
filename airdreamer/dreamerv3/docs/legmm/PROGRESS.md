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
