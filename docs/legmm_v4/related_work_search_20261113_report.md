# LegMM v4 相关工作检索报告

检索日期：2026-11-13。任务来源：`/Users/amiao/ADY/codex_tasks/related_work_search_20261113.md`。设计依据：`/Users/amiao/ADY/OPUS_TASK_BRIEF.md`（重点阅读第 0、1、4、5、9 节），并读取三份指定交接文档。本次只新建本报告，没有连接 bdi、修改代码、运行训练或实验。

## 1. 结论先行

**存在高度接近的单项主张和任务设定，不能把“学会站位”“足式移动夹取”“学习技能切换”或“世界模型用于移动操作”单独作为新贡献。尚未在本次已核验文献中确认有工作完整复现“未知朝向与受阻桌边下，通过移动和记忆选站位 + 世界模型在线学习 + 单次触发冻结夹取技能 + 成功率驱动的特权站位势函数退火并撤除验证”这一组合。** 后一句是有限检索结论，不是查无先例的证明。

最接近的五篇，按对当前研究主张的威胁列出：

1. **LMGN: A Unified Policy Learning Model for Last-Mile Grasp-Navigation**：四足带臂、接近方向、站位、桌面抓取均直接重合，是任务设定上最直接的风险。正文明确把目标朝向作为输入，抓取与导航联合优化；并非未知朝向下调用冻结技能。[arXiv 与摘要](https://arxiv.org/abs/2611.07515)、[正文 §3–4](https://arxiv.org/html/2611.07515v1)。
2. **Active-Perceptive Motion Generation for Mobile Manipulation（ActPerMoMa）**：直接针对通过移动观察未知场景并选择可执行抓取的站位，最接近“主动感知”主张。它采用显式地图、信息增益和可达性规划，不是 Dreamer 在线学习。[ICRA 2024 版本](https://arxiv.org/abs/2310.00433)、[正文 §II](https://arxiv.org/html/2310.00433v2)。
3. **N2M: Bridging Navigation and Manipulation by Learning Pose Preference from Rollout**：从既有操作策略的执行结果学习适合调用它的底座位姿，直接挑战“为固定技能学习站位”的新颖性。它用当前点云预测位姿分布，正文明确不依赖历史信息；不是主动收集信息的循环策略。[ICML 2026 官方页](https://proceedings.mlr.press/v306/chai26a.html)、[正文 §3–4](https://arxiv.org/html/2509.18671v2)。
4. **Predictive Reachability for Embodiment Selection in Mobile Manipulation Behaviors（MMDirector）**：世界模型预测可达性并学习何时由底座运动转向机械臂动作，最接近算法架构。只验证 reaching，夹爪不动作，三层策略共同训练且使用示范。[预印本](https://arxiv.org/abs/2410.21059)、[正文 §III–IV](https://arxiv.org/html/2410.21059v1)。
5. **ASC: Adaptive Skill Coordination for Robotic Mobile Manipulation**：Spot 上通过学习协调冻结的导航、抓取和放置技能，最接近足式平台与技能切换架构；另有可学习纠错策略，不是触发后完全固定的一次夹取。[摘要](https://arxiv.org/abs/2304.00410)、[最终 PDF](https://arxiv.org/pdf/2304.00410)。

另一个必须关注的近期工作是 **DreamMimic**：已经将世界模型、部分可观测足式操作和性能驱动的教师引导调度结合起来；但它是人形全身模仿与蒸馏，不能等同于特权势函数退火。[IROS 2026 作者声明及摘要](https://arxiv.org/abs/2608.22278)。

### 核验和阅读口径

下列文献均实际打开了作者原文、arXiv 摘要或官方出版页；重点近邻另读正文。未从可访问来源确认的会议不补写，标“arXiv 预印本，会议未核实”。“未报告/未核实”不等于“确定没有”。表中“世界模型”指学习到的状态转移预测模型，不把 IK、TSDF、运动学模型或新视角渲染一概算作 Dreamer 类世界模型。重合等级是本报告的定性判断，不是实验数字；“高”可以指某一核心主张高度重合，不意味着整套方法相同。

## 2. 按方向列文献

### 方向一：移动操作站位与底座选择

**[R1] N2M: Bridging Navigation and Manipulation by Learning Pose Preference from Rollout**  
Kaixin Chai et al.；2026，ICML（2025 首次预印本）。[官方出版页](https://proceedings.mlr.press/v306/chai26a.html)，[正文](https://arxiv.org/html/2509.18671v2)。  
从操作策略成功 rollout 学习初始底座位姿偏好，用当前 RGB 点云预测多模态位姿分布，再选无碰撞位姿执行预训练策略。重合在于依据下游技能实际表现选站位；区别是监督式位姿预测、无历史推理和在线主动观察，未见独立触发学习或特权势函数撤除实验。**高**。不依据检索页旧“ICLR 投稿”标记推断会议，正式来源明确为 ICML。

**[R2] N²M²: Learning Navigation for Arbitrary Mobile Manipulation Motions in Unseen and Dynamic Environments**  
Daniel Honerkamp et al.；2023，IEEE T-RO（2022 预印本）。[arXiv，含 T-RO 接收声明与 DOI](https://arxiv.org/abs/2206.08737)。  
把末端任务空间运动生成与学习的底座控制分开，使底座运动满足操作的运动学可行性，并处理未知或动态障碍。重合在于为操作学习导航；区别是随末端运动协调底座，不是先选静止站位再调用终止式夹取技能。摘要未证实朝向歧义、学习触发、世界模型或 q* 退火。**中**。

**[R3] BaSeNet: A Learning-based Mobile Manipulator Base Pose Sequence Planning for Pickup Tasks**  
Lakshadeep Naik et al.；2024，IROS（arXiv 的 Journal reference 与 DOI 已确认）。[原文入口](https://arxiv.org/abs/2406.08653)。  
通过分层学习和图神经网络学习抓取多个物体的底座位姿与抓取顺序，优化导航及抓取成本。重合是学习站位；区别是多目标组合规划，未在已读摘要中证明主动消除朝向歧义、记忆或触发冻结技能。**中**。

### 方向二：足式移动操作

**[R4] LMGN: A Unified Policy Learning Model for Last-Mile Grasp-Navigation**  
Xuenmin Hu et al.（按原文作者拼写）；2026，arXiv 预印本，会议未核实。[摘要](https://arxiv.org/abs/2611.07515)，[正文](https://arxiv.org/html/2611.07515v1)。  
在 B1 + Z1 上学习最后一段接近、底座调整、抓取和抬起，用方向可达性表示及分阶段混合专家协调动作。正文 §3 将目标朝向输入策略，使用 PPO 联合优化导航与抓取，奖励权重固定；未见世界模型、冻结抓取或 q* 成功率退火。与本项目平台类别、站位和接近方向高度重合，但核心信息条件不同。**高**。只确认仿真评测，不据此声称已有真机复现。

**[R5] Learning Multi-Stage Pick-and-Place with a Legged Mobile Manipulator（SLIM 正式版本）**  
Haichao Zhang et al.；2025，IEEE RA-L（作者 arXiv 注释确认接收）。[正式版本入口](https://arxiv.org/abs/2509.03859)，[正文](https://arxiv.org/html/2509.03859v3)。  
Go1 + WidowX 上学习视觉搜索、接近、抓取、搬运和投放；通过特权教师和学生蒸馏训练视觉策略。冻结的是行走控制器，抓取属于高层策略，不能当成“冻结夹取技能”；使用 SAC 路线，未见世界模型或 q* 势差退火。学生无需特权物体状态，但这不证明专门解决未知朝向选桌边。**中**。其技术报告 [SLIM: Sim-to-Real Legged Instructive Manipulation via Long-Horizon Visuomotor Learning](https://arxiv.org/abs/2501.09905) 已打开，作为同一研究合并，不重复计算独立证据。

**[R6] UMI-on-Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers**  
Huy Ha et al.；第八届 CoRL，PMLR 出版年 2025。[官方页](https://proceedings.mlr.press/v270/ha25a.html)。  
将现实手持夹爪示范训练的操作策略与仿真训练的全身跟踪控制器组合，接口为任务坐标系末端轨迹，并展示预训练策略跨平台部署。与项目的技能复用和四足带臂重合；区别是全身轨迹跟踪与示范学习，摘要没有学习静止站位或离散触发的证据，也没有主动朝向推断或特权 shaping 撤除的证据。**中**。

### 方向三：导航—技能切换、启动集合与技能衔接

**[R7] ASC: Adaptive Skill Coordination for Robotic Mobile Manipulation**  
Naoki Yokoyama et al.；2023 预印本；RA-L 接收版本（最终 PDF 标注 2023 年接受，正式卷期年份未进一步核实）。[摘要](https://arxiv.org/abs/2304.00410)，[最终 PDF](https://arxiv.org/pdf/2304.00410)，[可读 HTML 的较早版本 §IV](https://arxiv.org/html/2304.00410v3)。  
在 Spot 上学习何时使用导航、抓取和放置技能，并用纠错策略应对技能交接后的分布偏移。正文明确技能预训练后冻结，但协调器可组合技能与纠错动作，甚至允许底座和手臂同时运动；与单次触发后固定技能、回合结束的语义不同。未见世界模型或未知朝向信息收集专项验证。**高**。

**[R8] Effectively Learning Initiation Sets in Hierarchical Reinforcement Learning**  
Akhil Bagaria et al.；2023，NeurIPS。[官方页](https://papers.neurips.cc/paper_files/paper/2023/hash/e8da56eb93676e8f60ed2b696e44e7dc-Abstract-Conference.html)。  
用离策略价值估计与分类处理启动集合学习中的非平稳性、信用分配和悲观性，并在机器人抓取等任务中验证。它直接覆盖“判断一个技能从当前状态能否成功”的概念；不同于四足导航选站位，且研究技能自身变化造成的启动集合问题，非完全冻结夹取。**中**。

**[R9] Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization**  
Youngwoon Lee et al.；第五届 CoRL，PMLR 出版年 2022。[官方页](https://proceedings.mlr.press/v164/lee22a.html)。  
通过对抗式终止状态分布正则化，使前一技能结束状态更适合后一技能启动，应用于家具组装。与导航末端状态应进入抓取技能起始集合有关；但不是移动站位、主动视觉或世界模型，也不是保持下游技能全程不变的特定方案。**低**。

### 方向四：世界模型与高层决策

**[R10] Predictive Reachability for Embodiment Selection in Mobile Manipulation Behaviors（MMDirector）**  
Xiaoxu Feng et al.；2024，arXiv 预印本，会议未核实。[摘要](https://arxiv.org/abs/2410.21059)，[正文](https://arxiv.org/html/2410.21059v1)。  
世界模型预测未来手臂轨迹及任务奖励，以预测可达性奖励训练底座/手臂选择器。HSR 仿真中夹爪保持不动作；使用示范，三层策略同时训练，不能当成冻结夹取技能或纯在线从零学习。它已经覆盖“世界模型决定何时切换到操作”的核心组合；未知物体朝向主动推断和特权势差撤除未证实。**高**。

**[R11] Spatially Generalizable Mobile Manipulation via Adaptive Experience Selection and Dynamic Imagination**  
Ping Zhong et al.；2026，IJCAI。[官方摘要与全文](https://ijcai.org/proceedings/2026/697)。  
用自适应经验筛选训练 RSSM，并以模型预测规划改善移动操作和布局泛化。正文使用 Husky + UR5，SAC 与 RSSM 同时训练，规划时搜索候选动作；RSSM 损失沿用 DreamerV3 并不等于整个算法就是 DreamerV3。重合是学习导航与模型想象，区别是底座—手臂协调而非终止式冻结抓取；未见未知朝向、学习单次触发及 q* 退火协议。**中**。

**[R12] DreamMimic: Learning Visuomotor Whole-Body Loco-Manipulation via World Model**  
Jie Yin et al.；2026，IROS（arXiv 作者注释标接受）。[原文入口](https://arxiv.org/abs/2608.22278)。  
用 RSSM 支持人形视觉全身控制的特权教师蒸馏，并按教师和学生表现自适应调整指导强度。RSSM 用于预测表示与监督，不承担 Dreamer 式规划；不是四足桌边站位或冻结抓取，也没有把成功率调度等同于 q* 势函数退火。与“世界模型 + 部分可观测 + 性能驱动减少特权指导”接近。**中**，同时归入方向七。

**[R13] DreamTrajectory: Trajectory-Guided Action Generation with World Model Alignment for Mobile Manipulation**  
Zheng Yang et al.；2026，arXiv 预印本，会议未核实。[原文入口](https://arxiv.org/abs/2608.01381)。  
联合生成末端意图轨迹与全身动作块，用轨迹世界模型在测试时筛选更符合计划的动作。重合是世界模型辅助移动操作决策；区别是语言条件全身动作生成和测试时优化，非独立冻结夹取调用，摘要未提供未知朝向主动站位或特权退火证据。**中**。

### 方向五：主动感知与抓取视角选择

**[R14] Active-Perceptive Motion Generation for Mobile Manipulation（ActPerMoMa）**  
Snehal Jauhri et al.；2024，ICRA（2023 预印本）。[出版信息及摘要](https://arxiv.org/abs/2310.00433)，[正文](https://arxiv.org/html/2310.00433v2)。  
在 TIAGo++ 上按路径累计信息增益与抓取可执行性规划移动，持续更新场景并接近可抓位置。它已经把“为了看清而移动”和“为了抓取而选站位”结合；底座位姿来自规划而非 RL，用 TSDF 和可达性地图，预训练抓取检测器不等于冻结闭环夹取技能。抓取转换由稳定性/质量条件驱动，非学习离散触发。**高**。正文有历史融合，但不是学习的循环记忆。

**[R15] Affordance-Driven Next-Best-View Planning for Robotic Grasping（ACE-NBV）**  
Xuechao Zhang et al.；2023，CoRL。[官方摘要](https://proceedings.mlr.press/v229/zhang23i.html)。  
通过预测未观察视角的抓取可供性，选择预期抓取质量更好的下一视角，以寻找遮挡目标的可行抓取。与主动观察服务抓取重合；已读摘要没有证明移动底座站位学习、独立触发、冻结技能或特权 shaping。新视角预测不应直接记成学习动力学世界模型。**中**。PDF 因工具大小限制未成功解析，本文结论只依据已打开的官方摘要。

### 方向六：可行性、可供性与成功预测

**[R16] Do As I Can, Not As I Say: Grounding Language in Robotic Affordances（SayCan）**  
Brian Ichter et al.（按该 PMLR 版本作者顺序）；第六届 CoRL，PMLR 出版年 2023。[官方页](https://proceedings.mlr.press/v205/ichter23a.html)。  
用预训练技能的价值函数约束语言模型的技能选择，使选出的行动兼顾语言相关性和执行可行性。与利用技能成功预测指导高层决策重合；不是世界模型在线选站位，也不是未知朝向感知任务。“选择哪个技能”不应直接等同于学习连续导航中的最佳触发时刻。**中**。

**[R17] Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation**  
Tzu-Jung Lin et al.；2026，AAAI（2025 预印本，2026 修订注释确认接受）。[原文入口](https://arxiv.org/abs/2511.06240)。  
将 VLM 语义先验和几何可行性结合，用从粗到细的迭代优化选底座位置。与可供性引导站位接近；是零样本规划，不是从抓取成败在线训练站位策略，摘要未证实独立触发、冻结技能、动力学世界模型或 q* 撤除。**中**。

R1、R8、R10 也是本方向关键证据，分别对应 rollout 的策略偏好、启动集合和世界模型预测可达性，避免重复列为新文献。

### 方向七：特权 shaping、势函数与课程退火

**[R18] Improving the Effectiveness of Potential-Based Reward Shaping in Reinforcement Learning**  
Henrik Müller et al.；2025，AAMAS extended abstract（链接为较完整 arXiv 稿）。[原文入口](https://arxiv.org/abs/2502.01307)。  
分析势函数 shaping 效果对初始 Q 值和外部奖励的依赖，并研究平移势函数改善探索。与 q* 势函数的训练作用有关，但实验为网格及经典控制，不涉及移动抓取、特权朝向或成功率退火撤除。**低**。

**[R19] Benchmarking Potential Based Rewards for Learning Humanoid Locomotion**  
Se Hwan Jeon et al.；2023，ICRA。[原文入口，含期刊/会议参考与 DOI](https://arxiv.org/abs/2307.10142)。  
比较人形行走中的普通 shaping 与势函数 shaping；摘要报告后者对奖励尺度更稳健，但收敛加速有限。与机器人 PBRS 方法学相关，不能推导 LegMM 的 q* 一定改善探索，也不能当作撤除后的保持性证明。**低**。

**[R20] Potential-Based Reward Shaping For Intrinsic Motivation**  
Grant C. Forbes et al.；2024，AAMAS。[官方 PDF，已打开正文及作者页](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p589.pdf)。  
扩展势函数 shaping 对更一般函数的策略不变性条件，并把内在动机奖励转换为势函数形式，在网格任务验证。与动态辅助奖励的正确性相关；没有机器人站位、冻结技能或特权成功率退火。**低**。

**[R21] HEADER: Hierarchical Robot Exploration via Attention-Based Deep Reinforcement Learning with Expert-Guided Reward**  
Yuhong Cao et al.；2025，arXiv 预印本，会议未核实。[原文入口](https://arxiv.org/abs/2510.15679)。  
以层次地图和注意力策略选择探索视点，使用无参数特权奖励促进探索。重合在于训练期特权奖励支持部署期决策；任务为探索，已读摘要未证明 PBRS 或按成功率撤到零，不能用它宣称 LegMM 的完整方案已被做过。**低**。

本方向还包括 R5 的特权教师、R12 的性能驱动指导调度。**没有检索确认直接匹配“特权站位质量严格势差 + 成功率驱动降到零 + 撤除后继续训练验证”的文献**；这不意味着 PBRS、课程学习或撤除指导这些构件本身新颖。

## 3. 高、中重合文献对比表

“未核”表示已读部分未提供足够证据，不作否定；“未见”限于所读摘要/相关正文。“冻结”专指调用的夹取/操作技能，而不是冻结行走器、视觉编码器或教师。平台细节没有直接核实的保留未知。

| 文献 | 重合 | 平台 | 站位是否学习 | 触发/切换是否学习 | 学习世界模型 | 朝向未知/主动感知 | 夹取技能冻结 | 特权 shaping/退火 |
|---|---|---|---|---|---|---|---|---|
| R1 N2M | 高 | 多种移动操作；足/轮细节未核 | 是，rollout 监督 | 未见独立触发 | 未见 | 当前点云；明确无历史 | 复用预训练策略，未见联训 | 未见 |
| R2 N²M² | 中 | 多种移动操作平台 | 是，底座 RL | 非终止式夹取设定 | 未核 | 未知障碍；朝向未核 | 末端生成器，非该语义 | 未核 |
| R3 BaSeNet | 中 | 移动机械臂，足/轮未核 | 是，位姿及顺序 | 顺序选择；触发未核 | 未核 | 未核 | 未核 | 未核 |
| R4 LMGN | 高 | 四足 B1 + Z1 | 是 | 阶段路由；非单次 trigger | 未见，PPO | 明确输入目标朝向 | 否，联合优化 | 固定奖励；非撤除方案 |
| R5 SLIM | 中 | 四足 Go1 + WidowX | 隐含于移动操作策略 | 隐含阶段控制 | 未见，SAC | 视觉学生；专项朝向未核 | 否，冻结的是行走器 | 特权教师蒸馏，非 q* |
| R6 UMI-on-Legs | 中 | 四足带臂 | 全身跟踪；站位决策未核 | 未核 | 未核 | 示范视觉；专项主动感知未核 | 有预训练策略迁移，完整冻结契约未核 | 未核 |
| R7 ASC | 高 | 四足 Spot | 导航技能及协调，非单独位姿目标 | 是 | 未见 | 机载视觉；朝向专项未见 | 是，但另加可学习纠错 | 未见 q* 退火 |
| R8 Initiation Sets | 中 | Robosuite 等；非移动足式任务 | 不适用 | 学启动集合，非同一触发协议 | 未报告 | 未报告 | 技能与启动集合随训练变化 | 未报告 |
| R10 MMDirector | 高 | 轮式 HSR 仿真 | 是，移动策略 | 是，底座/手臂掩码 | 是 | 图像输入；不研究抓取朝向 | 否，共同训练且不夹取 | 预测可达性奖励，非特权 q* 退火 |
| R11 Spatially Generalizable MM | 中 | 轮式 Husky + UR5 | 是，导航配合操作 | 单次触发未见 | RSSM + 规划 | 未知布局；朝向专项未见 | 未证实，技能与 RSSM 同训 | 未见 |
| R12 DreamMimic | 中 | 人形 | 无独立桌边站位主张 | 未报告 | 是，供蒸馏与预测表示 | 部分可观测视觉；抓取朝向未核 | 否，学生控制器学习 | 特权监督 + 性能调度；非势差 |
| R13 DreamTrajectory | 中 | 移动机械臂；足/轮未核 | 联合底座/手臂动作 | 未报告独立 trigger | 是，轨迹预测 | 变化视角；主动消歧未核 | 非冻结夹取宏动作 | 未报告 |
| R14 ActPerMoMa | 高 | 轮式 TIAGo++ | 否，规划选位姿 | 质量/稳定性规则 | 否，显式地图与几何 | 主动观察未知场景；非专门未知 yaw | 预训练检测器，非相同技能契约 | 不采用 RL shaping |
| R15 ACE-NBV | 中 | 抓取机器人；底座类别未核 | 选择视点；底座学习未证实 | 未核 | 新视角预测，非动力学 WM | 遮挡下主动抓取观察 | 未核 | 未核 |
| R16 SayCan | 中 | 移动机械臂；轮式细节未核 | 未报告 | 技能选择，是；时刻语义不同 | 未报告动力学 WM | 非朝向主动消歧主张 | 使用预训练技能 | 未报告 q* 退火 |
| R17 Affordance-Guided | 中 | 移动机械臂；足/轮未核 | 零样本优化，非在线 RL | 未核 | 未报告动力学 WM | 可供性搜索；朝向歧义未核 | 未核 | 未核 |

## 4. 审稿风险与可强调的区别

- **最直接任务风险：R4 LMGN。** “四足带臂自己选方向和站位抓物体”已被明确提出。可强调的信息条件是“不向策略提供目标朝向”，控制边界是“抓取技能固定且只能执行一次”。但只有在主动观察和记忆的消融成立时，这才是实验贡献。
- **最直接主动感知风险：R14 ActPerMoMa。** 它已有通过移动获取视野、兼顾抓取可执行性的完整系统；不能把换桌边、绕障和看清目标本身说成新颖。学习潜在记忆相对显式融合地图的作用需要证据，单纯换成 Dreamer 不自动产生贡献。
- **站位学习风险：R1 N2M、R2、R3、R17。** 应把“几何上够得着”和“固定技能实际上能成功”分开。N2M 已强调这种差别，LegMM 需要进一步展示未知信息下的序贯决策价值。
- **架构风险：R7 ASC、R10 MMDirector、R11。** 冻结技能协调、世界模型切换和模型想象均已有先例。学习 trigger 适合作为实现机制，与项目任务书定位一致。
- **课程主张风险：R12 DreamMimic、R5、R18–R21。** “训练有特权、部署没有”“根据表现减少教师影响”并非空白。可保留的具体问题是：特权只进入奖励，按真实任务成败退火，撤除后继续训练，在相同冻结技能及相同预算下与纯无 q* 比较；不能用 PBRS 的策略不变性理论替代有限预算下的保持性实测。

建议论文围绕受限信息下的序贯站位选择建立证据：初始好面不可见或朝向有歧义、桌边受阻、真正无历史模型、规则触发和同预算纯无 q* 的分层比较。以上是文献定位建议，不修改任务书或新增已批准实验。若坚持不加入可部署经典流水线基线，需要正面讨论与 ActPerMoMa 的区别；目前不能仅凭最近站位基线排除显式主动规划也能解决任务的可能性。

本次没有重跑项目训练、没有对旧 kin 成功率作新结论。尤其不能把历史 legacy 契约结果作为已验证的主动感知或撤除 q* 证据；项目状态以指定交接文档为准。

## 5. 检索记录与限制

使用 web 搜索工具进行开放网络检索，正式来源核验覆盖 arXiv、PMLR（ICML/CoRL）、NeurIPS、IJCAI、AAMAS 官方 PDF 和作者项目页。IEEE Xplore 的 ASC 页面返回 418，改读可打开的 arXiv 最终 PDF；ACE-NBV 官方 PDF 超出工具大小上限，官方摘要可读。未以聚合站或搜索摘要代替收录文献的直接打开核验。未对所有检索候选穷尽全文，也未对每个数据库完整遍历。

检索包含所有指定方向，每方向三个原始查询；初次按方向合并搜索，随后逐条复查以获得可追溯的返回计数。下表“返回条目”是工具本次实际显示的结果块数（可含重复、无关和非论文结果），不是搜索引擎总命中数。“已核验相关”只统计该次结果文本中出现且本报告收录核验的独立文献编号，是严格保守的相关计数；不把未经打开核实的候选算成相关文献。数字来源为本报告附带的查询记录表和对应编号链接，不是机器人实验数字。

| 查询编号/方向 | 实际查询 | 返回条目 | 已核验相关（去重） |
|---|---|---:|---|
| Q1 / 1 | mobile manipulation learned base placement N2M2 navigation manipulation | 23 | 4：R1、R2、R11、R17 |
| Q2 / 1 | mobile manipulation where to stand reinforcement learning grasp base placement | 17 | 0：无 |
| Q3 / 1 | navigation for manipulation active perception grasping base placement | 20 | 2：R4、R14 |
| Q4 / 2 | legged mobile manipulation quadruped arm learning grasping 2024 2025 | 18 | 2：R5、R6 |
| Q5 / 2 | quadruped Spot ANYmal B1 Z1 mobile manipulation reinforcement learning | 16 | 0：无 |
| Q6 / 2 | A2 AIRBOT Go2 arm active grasping robot | 26 | 0：无 |
| Q7 / 3 | mobile manipulation learning when to switch navigation grasping skills | 21 | 3：R3、R7、R11 |
| Q8 / 3 | hierarchical reinforcement learning frozen skills initiation termination robot manipulation | 22 | 1：R8 |
| Q9 / 3 | robot skill chaining initiation sets terminal conditions reinforcement learning 2022 | 18 | 1：R9 |
| Q10 / 4 | Dreamer world model mobile manipulation navigation grasping | 28 | 3：R11、R12、R13 |
| Q11 / 4 | world model legged mobile manipulation hierarchical reinforcement learning | 24 | 1：R5 |
| Q12 / 4 | model based reinforcement learning high level frozen skills robot manipulation | 19 | 0：无 |
| Q13 / 5 | active perception grasping mobile robot next best view 2023 2024 | 24 | 2：R14、R15 |
| Q14 / 5 | move to see grasping object pose ambiguity active perception | 21 | 0：无 |
| Q15 / 5 | learning mobile base viewpoint selection grasping partially observable | 22 | 1：R3 |
| Q16 / 6 | mobile manipulation affordance success predictor navigation grasping | 21 | 1：R17 |
| Q17 / 6 | navigation manipulation feasibility predictor learned reachability | 21 | 2：R10、R11 |
| Q18 / 6 | SayCan affordance value functions robot skill success | 22 | 1：R16 |
| Q19 / 7 | privileged reward shaping potential based annealing robotics | 25 | 3：R18、R19、R20 |
| Q20 / 7 | potential based reward shaping annealing curriculum reinforcement learning 2023 | 22 | 2：R18、R19 |
| Q21 / 7 | teacher guided curriculum privileged information reward shaping removal | 18 | 0：无 |

上述计数由返回文本的结果块和文献标识匹配获得，可复核口径为“本报告收录集合的命中数”，不声称其他候选不相关。特别是 teacher/curriculum 的泛化查询返回了教育学内容，相关数为零，因此又用下列定向查询补充，而没有据此把方向七记作没有相关研究。

补充查询（同样使用 web 搜索）：

| 查询 | 收录核验的相关文献 |
|---|---|
| "Adaptive Skill Coordination" mobile manipulation | R7 |
| "privileged" "reward shaping" annealing robotics | R21；其他候选未收录 |
| "mobile manipulation" "Dreamer" 2025 2026 | R11 |
| "N2M" "Chai" ICLR 2026 | R1；追到 ICML 正式出版页，更正旧投稿信息 |
| "mobile manipulation" "potential" "annealing" | 未获得直接匹配本报告收录项的证据 |

原始开放检索没有指定单一域名；上述“在哪些站点”指实际打开核验的来源，不意味着在每个会议网站分别完成全文数据库检索。检索时间范围按论文元数据筛选为 2018 年以后；本报告收录集中于 2022–2026 出版记录及相关较早预印本。没有因抓取页显示“最近发表”而把旧论文当作新论文，例如搜到的 ICML03 shaping 页面即未作为范围内新作收录。

已读重点原文位置：

| 文献 | 实际核验位置 | 关键判据 |
|---|---|---|
| R1 | arXiv HTML §3–4；PMLR 官方元数据 | 当前点云、位姿分布、rollout、无历史信息 |
| R4 | arXiv HTML §3–4 | 目标朝向输入、PPO、B1 + Z1、桌面与局部障碍 |
| R5 | arXiv HTML §IV 及硬件说明 | 冻结行走器、特权教师、SAC 蒸馏、Go1 |
| R7 | arXiv HTML §IV；最终 PDF | 冻结技能、协调与纠错动作；较早 HTML 与最终 PDF 区分 |
| R10 | arXiv HTML §III–IV | 世界模型、共同训练、示范、HSR、夹爪不动作 |
| R11 | IJCAI 官方页全文 §3–4 | Husky + UR5、RSSM 与 SAC、CEM 规划 |
| R14 | arXiv HTML §II | 信息增益、可达性、TSDF、稳定抓取条件 |
| R20 | AAMAS PDF 首页及摘要 | 作者、会议、一般化 PBRS 与网格实验 |

## 6. 交付与自查

- 新建文件：`/Users/amiao/ADY/codex_tasks/related_work_search_20261113_report.md`；用途是记录核实后的文献与撞题风险。
- 文件操作：用本地文件读取命令读取任务、任务书、交接及 orchestration 技能；用 web 的 search/open 检索核验；用 apply_patch 只创建本报告。未操作远程仓库，因此不生成或伪报 bdi 的 git diff --stat。
- 本地报告结构检查：通过 Python 断言核对文献编号、逐条查询编号、七方向覆盖及无内部引用标记；检查结果为通过。该检查只验证报告结构，不代表文献结论或项目已验收。
- 未运行代码测试、全量回归或 CPU/GPU 冒烟：本任务为纯文献检索，这些结果均为“不适用”，不报告通过数或性能估计。
- 局限：部分只读摘要，平台和训练细节在对比表中保留“未核”；预印本不冒称已发表；没有证明查全。文献之间及项目之间没有做数值性能比较。
- 尚待项目实测：主动感知因果证据、无记忆消融、冻结真实技能结果、同预算纯无 q* 及退火撤除后保持性；本检索不替代这些实验。
- 交付清单：中文结论、重点近邻链接、各方向文献、维度对比表、审稿风险、逐查询记录均在本报告；等待 Opus 审阅，不宣称已验收。


