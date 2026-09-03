---
type: concept
tags: [evaluation, benchmark, metrics, locomotion, whole-body-tracking, motion-tracking, humanoid, wbc, mpc, sim2real]
status: complete
updated: 2026-09-03
related:
  - ../overview/hub-embodied-eval-benchmark.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ./sim-vs-real-eval-gap.md
  - ./whole-body-tracking-pipeline.md
  - ./simulation-evaluation-infrastructure.md
  - ./humanoid-policy-reward-functions.md
  - ./control-inference-frequency-decoupling.md
  - ./hqp.md
  - ./sim2real.md
  - ../tasks/locomotion.md
  - ../entities/paper-humantracker.md
  - ../entities/humanoid-bench.md
  - ../entities/trackerlab.md
  - ../entities/paper-barkour-quadruped-agility-benchmark.md
  - ../entities/paper-notebook-benchmarking-humanoid-imitation-learning-with-mo.md
  - ../entities/paper-notebook-mimicking-bench-a-benchmark-for-generalizable-hu.md
  - ../queries/humanoid-motion-tracking-method-selection.md
  - ../methods/sonic-motion-tracking.md
  - ../formalizations/control-loop-latency-modeling.md
sources:
  - ../../sources/papers/humantracker_arxiv_2608_13555.md
  - ../../sources/papers/humanoid_pnb_benchmarking-humanoid-imitation-learning-with-mo.md
  - ../../sources/papers/barkour_arxiv_2305_14654.md
  - ../../sources/repos/humanoid-bench.md
  - ../../sources/repos/trackerlab.md
summary: "运控模型（locomotion / whole-body tracking / MPC-WBC 控制器）的评测指标体系：把跟踪误差类、命令跟随类、model-based 求解类、硬件部署类四组指标分开定义，说明为什么单报 MPJPE 会奖励「死得更早」的策略、为什么必须钉死参考表示/rollout 记账/终止准则/指标实现四项才有可比性，以及难度分层与仿真→半实物→真机三段验收怎么落地。"
---

# 运控模型评测指标（Motion-Control Policy Evaluation Metrics）

## 一句话定义

**运控模型评测指标** 指用于给 locomotion 策略、whole-body tracking 策略与 model-based 控制器（MPC / WBC）出具验收结论的那组指标——它回答的不是"任务做没做成"，而是"**跟得准不准、走得稳不稳、算得过来算不过来、敢不敢上真机**"，因此与任务侧的成功率评测是两套坐标系，互相套用会得出错误结论。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPJPE | Mean Per-Joint Position Error | 平均每关节位置误差；通常只在**已执行片段**上取平均 |
| Succ | Success / Completion Rate | 未触发终止准则的回合比例，必须与 MPJPE 联报 |
| CoT | Cost of Transport | 单位质量单位距离能耗（J/kg/m），步态经济性主指标 |
| MDS | Motion Difficulty Score | 与策略无关的动作难度分，用于把结果按难度分档 |
| MID | Maximum Imitable Difficulty | 策略能跟住的最高难度档 |
| DSJE | Difficulty-Stratified Joint Error | 按难度分层报的关节误差 |
| WBT | Whole-Body Tracking | 全身运动跟踪，本页跟踪误差类指标的主要对象 |
| WBC | Whole-Body Control | 全身控制器，model-based 求解类指标的对象 |
| MPC | Model Predictive Control | 模型预测控制，同上 |
| QP | Quadratic Programming | WBC/MPC 的标准求解形式，求解耗时的来源 |
| OOD | Out-of-Distribution | 分布外，评测集覆盖不到的地形/扰动/参考 |

## 为什么重要

- **任务侧指标测不到运控的失败模式。** 「成功率」这类结果指标假设任务有离散的完成判据；运控模型的坏法却是连续的——跟踪逐渐发散、步态越走越抖、力矩顶到限幅、求解偶发超时。把 [任务侧成功率评测](../queries/embodied-eval-benchmark-selection-loop.md)那套指标直接搬过来，这些失败全部不可见。
- **单报误差类指标会奖励错误行为。** MPJPE 一类误差通常只在**已执行片段**上取平均，一条早早触发终止的 rollout 反而误差更小——"更准"可能只是"死得更早"。这不是理论担忧，[HumanTracker](../entities/paper-humantracker.md) 正是因此把 Succ 与 MPJPE 强制联报。
- **数字好看 ≠ 视频里像人。** 解析误差与人的主观判断之间有系统性偏差，需要专门的偏好对齐指标补位。
- **它是训练路线的共用验收环节。** [RL 运动控制](./sim2real.md)、[全身跟踪流水线](./whole-body-tracking-pipeline.md)、行为基础模型都在"把控制器训出来"；本页负责给它出一份能拿去做发版决策的数字。

## 核心原理

### 四组指标，四种坐标系

| 组 | 被测对象 | 主指标 | 这一组单独看会漏掉什么 |
|----|----------|--------|------------------------|
| ① 跟踪误差类 | whole-body tracking / motion imitation | Succ（终止准则下跑完比例）、MPJPE / 关键点误差、末端与骨盆误差、人类偏好分 | 只看误差会奖励早失败；只看 Succ 看不出跟得糙 |
| ② 命令跟随类 | velocity-command locomotion | 命令跟踪误差（线速度/角速度）、行走速度、CoT、地形通过率、摔倒频率、运动自然性 | 平地高分掩盖地形与长尾；速度快但 CoT 爆炸 |
| ③ model-based 求解类 | MPC / WBC / TSID 控制器 | 求解耗时与实时率（deadline miss 比例）、约束违反量与可行性、模型/增益误差下的鲁棒裕度 | 纯 RL 指标表里完全不出现这一组 |
| ④ 硬件与部署类 | 真机上的同一策略 | 关节力矩 / 电流 / 温升 / 冲击峰值的安全裕度、长时间连续运行稳定性 | 仿真里全部为零成本，真机上它们才是发版红线 |

**①②** 的分野常被忽略：跟踪类问"和给定参考轨迹差多少"，命令跟随类问"和给定速度指令差多少"，参考的存在与否决定了误差怎么定义、终止怎么判。**③** 的量纲完全不同——它测的是控制器本身算不算得过来：人形 30+ DoF 在 500–1000 Hz 下的 [HQP 实时求解压力](./hqp.md)是真实约束，而策略侧推理常在 10–50 Hz、靠 [控制/推理频率解耦](./control-inference-frequency-decoupling.md)撑住，两条链路的时序预算见 [控制环延迟建模](../formalizations/control-loop-latency-modeling.md)。

### 可比性的前提：四项必须先钉死

不同工作报出的同名指标往往不可比。要横评运控模型，先把下面四项统一，否则**比的是后处理不是策略**：

1. **参考表示** — 参考动作统一转成同一套 `qpos`（或同一命令空间），而不是各家各自的中间格式；
2. **rollout 记账** — 运动列表、机器人模型、参考索引、每步记录的状态量一致；
3. **终止准则** — 什么算失败必须同源。[HumanTracker](../entities/paper-humantracker.md) 对齐 [SONIC](../methods/sonic-motion-tracking.md)：骨盆、双踝或双腕垂向误差超过 **0.25 m**，骨盆旋转超过 **1 rad**，或 `qpos`/`qvel` 非有限即判失败；
4. **指标实现** — 误差在哪些关节上取、在哪些帧上取（HumanTracker 的 MPJPE 只在**已执行片段**上对 29 个主动关节取平均绝对误差）。

只有策略接口保持各自原样、以上四项统一，表里比的才是 tracker 本身——这正是 [HumanTracker](../entities/paper-humantracker.md) 能把 GMT / TWIST2 / SONIC / Humanoid-GPT 放进同一张表的原因。

### 偏好对齐：让"数字好看"和"像人"重合

解析误差之外还需要一条**从专家成对偏好学出来的轨迹指标**。HumanTracker 的 **HumanScore** 把整条轨迹切成数秒窗、逐窗打分再按真实帧数加权，得到 0–100 分；其族均衡对齐率 **90.83%**，比最强的单条解析诊断高约 **6.8** 点，且消融显示优势来自**数秒的接触证据**而不是未来参考残差。含义是：接触历史比瞬时姿态差更能解释"人觉得这段动作对不对"。

### 难度分层：让"比 SOTA"重新有意义

聚合均值会被容易的动作稀释。[Motion Difficulty Score](../entities/paper-notebook-benchmarking-humanoid-imitation-learning-with-mo.md) 从刚体动力学出发，对参考姿态做小扰动、观察力矩变化空间，从体积 / 方差 / 时间变化率三维打分，得到**与策略无关**的难度尺子；据此把 AMASS 切成难度分层的 MD-AMASS，并配 **MID**（最大可模仿难度）与 **DSJE**（按难度分层的关节误差）。评测表因此从"比一个总分"变成"在每个难度档分别比"。

同理，动作族分层也是必要的：族按**暴露的失败机制**划分而非按活动频率均分，主表必须按族报，不能只给一个总分。

## 工程实践

### 指标读法速查

| 指标 | 怎么读 | 常见坑 |
|------|--------|--------|
| MPJPE / 关键点误差 | 必须与 Succ 同表出现 | 单报时早失败的策略反而"更准" |
| Succ | 看终止准则定义再看数值 | 换了终止阈值就不可比 |
| HumanScore 类偏好分 | 与解析误差**并列**报，不互相替代 | 用它代替末端轨迹精度 |
| CoT | 与速度、地形一起报 | 慢慢走当然省电，孤立看无意义 |
| 摔倒频率 | 分地形/分扰动报 | 平地零摔倒掩盖楼梯崩溃 |
| 敏捷分（如 Barkour 的 0–1 分） | 当作压缩后的综合分 | 拿它反推单项能力 |
| 求解耗时 | 看**分位数与 deadline miss 比例**，不看均值 | 均值达标但 p99 超时，真机上就是抖动 |
| 约束违反 | 报违反量与发生率 | 只报"可行"，不报违反幅度 |
| 力矩/电流/温升/冲击峰值 | 报裕度而非绝对值 | 短测通过、长时间跑温升越限 |

### 三段分层验收

按 [locomotion 任务页的工程落地检查](../tasks/locomotion.md)，把评测拆成成本递增的三段，不要一步跳到真机：

1. **仿真**：看成功率、跟踪误差与能耗；用统一套件（[HumanoidBench](../entities/humanoid-bench.md) 的 15 项全身操作 + 12 项运动任务、[Mimicking-Bench](../entities/paper-notebook-mimicking-bench-a-benchmark-for-generalizable-hu.md) 的人形模仿泛化、四足走 [Barkour](../entities/paper-barkour-quadruped-agility-benchmark.md) 的敏捷课）；
2. **半实物**：看延迟与电流——这一段专门暴露 ③ 组求解与时序问题；
3. **真机**：看温升、冲击与长时间稳定性，这一段的数字才是发版红线。

换本体后策略突然退化时，先用 [TrackerLab](../entities/trackerlab.md) 这类统一实验接口把**参考表示、奖励与机器人模型**的影响分开，排除环境差异，再去改策略。

### 失败 rollout 要归档，不要只报均值

把失败按机制分类保存（摔倒、打滑、膝盖反关节、脚底震荡、求解超时、力矩限幅），它们比平均 reward 更能指导下一轮改动。评测报告的最小形态应当是：**Succ + 误差类 + 偏好/敏捷分，按动作族或难度档分报，附失败模式分解**。

## 局限与风险

- **奖励项 ≠ 评测指标。** 能耗惩罚、抖动惩罚是训练信号；直接拿同一个式子当验收指标是自证其说。两者的边界见 [人形策略奖励函数](./humanoid-policy-reward-functions.md)。
- **跨基准直接比榜不成立。** 任务集、观测接口、重置协议、终止准则、指标实现不同源，数字放在一起没有意义。
- **难度尺子有边界。** MDS 建立在刚体动力学与力矩敏感度之上，衡量的是**动力学意义上的难**，不覆盖感知或任务语义层面的难度。
- **偏好指标继承标注者的偏好。** HumanScore 类指标对齐的是专家判断，不是物理正确性；它能说"人更喜欢哪条"，不能替代约束满足与安全裕度。
- **仿真里的绝对分不能直接外推。** 运控模型同样受 [sim↔real 评测 gap](./sim-vs-real-eval-gap.md) 约束：该看 sim↔real 排名相关性，而不是仿真绝对成功率；接触与物理层面的根因见 [物理保真度 sim2real gap](./physics-fidelity-sim2real-gap.md)。
- **基准数据可得性不等于协议可得性。** 例如 HumanTracker 的评测框架与 HumanScore 权重已开源，而 153 h 数据集截至入库日仍未发布——复现时要先核对拿得到哪一半。

## 关联页面

- [具身评测基准选型闭环（知识链汇总）](../overview/hub-embodied-eval-benchmark.md) — 本页是其四层评测链在**运控模型**这一被测对象上的展开
- [Query：具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 任务侧（③ 层）的姊妹页，本页与之互补而非替代
- [仿真 vs 真机评测 gap](./sim-vs-real-eval-gap.md) — 本页所有仿真指标能否外推的前提
- [仿真评测基础设施](./simulation-evaluation-infrastructure.md) — 把这些指标做成高频闭环的工程底座
- [全身跟踪流水线](./whole-body-tracking-pipeline.md) — ① 组指标的被测管线
- [Locomotion 任务页](../tasks/locomotion.md) — ② 组指标与三段验收流程的原始清单
- [人形策略奖励函数](./humanoid-policy-reward-functions.md) — 训练奖励项，与本页验收指标必须分开
- [HQP](./hqp.md) · [控制/推理频率解耦](./control-inference-frequency-decoupling.md) — ③ 组求解与时序指标的机制背景
- [HumanTracker](../entities/paper-humantracker.md) · [HumanoidBench](../entities/humanoid-bench.md) · [TrackerLab](../entities/trackerlab.md) · [Barkour](../entities/paper-barkour-quadruped-agility-benchmark.md) — 本页指标的具体承载基准
- [Query：人形动作跟踪方法选型](../queries/humanoid-motion-tracking-method-selection.md) — 选完指标之后的方法选型侧

## 参考来源

- [HumanTracker 论文](../../sources/papers/humantracker_arxiv_2608_13555.md) — 四族光学基准、终止准则对齐、MPJPE 记账口径与 HumanScore 偏好指标
- [Benchmarking Humanoid Imitation Learning with Motion Difficulty](../../sources/papers/humanoid_pnb_benchmarking-humanoid-imitation-learning-with-mo.md) — MDS / MD-AMASS / MID / DSJE 难度分层评测
- [Barkour 论文](../../sources/papers/barkour_arxiv_2305_14654.md) — 四足敏捷课与 0–1 综合敏捷分
- [HumanoidBench 仓库](../../sources/repos/humanoid-bench.md) · [TrackerLab 仓库](../../sources/repos/trackerlab.md) — 统一评测套件与统一跟踪实验接口
- 本页的 ②④ 组指标与三段验收流程归纳自 [locomotion 任务页](../tasks/locomotion.md) 的「评价指标」与「工程落地检查」小节

## 推荐继续阅读

- Liu et al., *Towards Comprehensive and Human-Aligned Motion Tracking Benchmark*（HumanTracker）— <https://arxiv.org/abs/2608.13555>
- *Benchmarking Humanoid Imitation Learning with Motion Difficulty* — <https://arxiv.org/abs/2512.07248>
- Caluwaerts et al., *Barkour: Benchmarking Animal-level Agility with Quadruped Robots* — <https://arxiv.org/abs/2305.14654>
