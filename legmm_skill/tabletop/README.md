# W8 实施状态（2026-11-15，未完成交付）

当前门槛：固定验证 seed10000–10255 全256计数，老师至少231成功（90%），学生至少218成功（85%）。自由站立、全场景/适用actor求解器4/0、15s与连续保持2s、禁止碰撞及窄面约束不变。

旧236/256使用桌/物体默认16/1，不是指定4/0资格。真正4/0下，经实际指垫长度/物体投影尺寸推导插入深度与预抓取距离后，原固定256生产闭环老师为231/256，无执行中禁止碰撞失败。保留一个已成功终止后继续随批运行而出现的接触；不能将原始累计接触写成零。证据：W8输出根 `teacher_true4_pregrasp_fixed256_retry1_01/summary.json`，执行记录根 `w8_true4_contact_retry1_20261115/post_terminal_contact_audit.json`。

相机的USD/Fabric位姿及不推进物理的渲染已修复。8环境1800步有/无渲染配对一致；多阶段真实RGB/DA2/腕米制深度和语义诊断通过。诊断语义/物体状态不进入学生。完整双路视觉、DA2、监督缓冲、学生实际优化更新/恢复已测32、64、128档；峰值分别10154、11792、16139MiB。256档外推24833MiB超过自有18GiB，故未启动256–4096，不能称4096实测或达成。

正式DAgger首次启动于 `dagger_true4_visual128_retry1_01`，实际128环境，6轮×32000环境决策，总192000；每轮2000更新，有效batch64、微批8、lr1e-4、有界10000帧。第3/6轮保存并验证固定256；实际进度与PID以该目录status/training_status和逐回合日志为准。尚未学生达线；PPO、独立test、最终冻结均未完成。

入口：`learning_launch.py` / `train_dagger.py`；独立PPO实现 `ppo_launch.py` / `train_ppo.py` 要求已选学生>=218/256和相同老师源码/物理审计，不能绕过门槛启动。残差0.05rad/0.01m、TimePPO/M6、零均值/std0.2；有效1024样本、4个256小批、5epochs。PPO每100更新在固定前32个val作smoke并选内存权重，预算末对所选权重全256验证；常规checkpoint500/1000。PPO代码现仅CPU验证，尚未实际GPU smoke/正式训练。

`frozen_skill.py` 提供W10调用边界（待合格权重及manifest生成）：`reset`后每0.16s调用 `act_depth(images,proprio)` 或 `act_raw(head_rgb,wrist_depth_metres,proprio)`，返回4个25Hz native目标。只有双路深度/本体输入；物理100ms延迟、PD和自由站立仍由调用环境执行。没有修改W9。

执行记录根的协调指令须在阶段边界重读并ack；不写启动器生命周期文件。GPU0优先/2备用、单Isaac、自有18GiB、宿主可用内存8GiB保护、磁盘至少十进制10GB；旧证据不覆盖、不提交推送。


## 自身指令历史修复（显式启用）

`tabletop.train_dagger`和`tabletop.learning_launch`新增`--command-history`，默认关闭。启用时网络原生状态为14维：实测joint1..6与夹爪总开口7维，随后是真实上一条已发出的同单位指令7维；网络看见的是尚未完全经过100ms延迟的自身指令，不是固定环形队列末槽。图像、标签、8预测4执行、限幅、延迟和物理成功判据不变。

外部`Student.act(images, measured_proprio7, last_command7)`接口保留，训练缓存自动保存两者拼接的14维状态。模型文件使用`w8_own_last_command_student_v1`及明确input_contract；完整会话恢复也验证契约和缓冲维度。旧7维缓存缺少这段历史，不得补猜后混入新训练。`Student.from_checkpoint`为PPO自动选择相应契约；冻结接口在reset后维护真实上一指令，manifest记录内部14维状态。

2026-11-16修复控制根：`/home/bdi/wang-sm/logdir/legmm/v4/w8_command_history_repair_20261116`。旧128000/8000完整状态单独保留，新链从头使用原训练初态；短视觉更新/恢复探针不等于学生物理成功率已提升。
