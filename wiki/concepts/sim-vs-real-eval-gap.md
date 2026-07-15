---
type: concept
tags: [evaluation, benchmark, sim2real, simulation, embodied-ai, reproducibility, world-model]
status: complete
updated: 2026-07-14
summary: "仿真评测可复现性 ↔ 真实世界代表性 取舍概念页：明示仿真基准在可复现性/吞吐/可控性上的优势，为何以牺牲真实接触、感知噪声、长尾分布的代表性为代价，并把这条 gap 讲成「评测结论能否外推到真机」的物理根因；配可复现性 vs 代表性代价表、缩小评测 gap 的三条工程路线与常见误判速查。"
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md
  - ../../sources/papers/robo_bench_arxiv_2510_17801.md
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../concepts/sim2real.md
  - ../concepts/physics-fidelity-sim2real-gap.md
  - ../entities/paper-gigaworld-1-policy-evaluation.md
---

# 概念：仿真评测可复现性 ↔ 真实世界代表性取舍（sim↔real 评测 gap）

> **一句话**：仿真评测的可复现性、吞吐与可控性几乎都是**用「牺牲真实接触/感知噪声/长尾分布的代表性」换来的**——所以一个仿真榜单越好复现、越干净，它的结论就越可能**外推不到真机**。这条 gap 不是「仿真做得不够真」的工程 bug，而是[评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)里「④ sim↔real 校准层」必须单独存在的物理根因。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| sim2real | Simulation-to-Real | 仿真到真机的迁移与外推 |
| real-to-sim | Real-to-Simulation | 用真机数据校准/构建仿真，锚定二者相关性 |
| OOD | Out-of-Distribution | 分布外，部署时相对评测集的分布漂移 |
| WM | World Model | 世界模型，也可当策略评估器，同样受 gap 约束 |

## 为什么单独立这一页

「仿真里测出来的结论能不能信」这个问题，横跨[评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)的每一层：认知评测、世界模型保真度评测、策略成功率评测最终都要回答「这个分数在真机上还成立吗」。把这条取舍从各评测基准实体页里抽出来独立成页，是为了讲清一件容易被榜单掩盖的事——**可复现性和真实代表性在评测里天然对立，提高其中一个通常压低另一个**，因此不存在「又干净又代表真机」的免费评测。

## 可复现性 ↔ 代表性：这道 gap 的代价表

仿真评测的每一项工程优势，都对应放弃了一部分真实世界的代表性：

| 仿真评测优势 | 换来它放弃了什么真实代表性 |
|--------------|--------------------------|
| 可复现（固定种子/初值，结果确定） | 真机的随机接触、传感器噪声、初始条件抖动 |
| 高吞吐（GPU 并行成千上万 rollout） | 真机每次 rollout 的物理磨损/标定漂移成本 |
| 可控（任意重置场景、注入干扰） | 真实长尾分布：罕见物体、极端光照、退化接触 |
| 干净观测（无标定误差、无遮挡噪声） | 真实感知链路的时延、抖动、缺失与错配 |
| 可自动打分（成功判据程序化） | 真实「做成」的语义模糊与人工判定成分 |

核心结论：**仿真评测越靠左列的优势极致，它的分数就越是「在一个被简化过的世界里」得到的**——[RoboBench](../entities/robo-bench.md) 之所以要专门验证「认知分与下游 VLA 成功率相关」、[GigaWorld-1](../entities/paper-gigaworld-1-policy-evaluation.md) 之所以强调「长时序动作忠实比短时视觉逼真更决定评估质量」，本质上都是在为「上层易测代理指标能否外推到真机结果」这件事补证据。

## 把 gap 讲成「评测结论能否外推」的物理根因

评测 gap 的物理根因和 [Sim2Real](../concepts/sim2real.md)、[物理保真度 sim2real gap](../concepts/physics-fidelity-sim2real-gap.md) 同源，但落点不同：

- **策略迁移 gap** 问的是「策略在真机上还能不能跑」；
- **评测 gap** 问的是「用仿真给策略打的分，在真机上排名还成不成立」。

后者对**相对排名**的要求，往往比对绝对成功率的要求更宽松也更严苛：宽松在于——即使仿真绝对成功率偏高，只要 sim↔real **排名相关**，仿真仍能当有效的选型/迭代信号（这正是[仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)成立的前提）；严苛在于——一旦训练管线与评测共享同一仿真分布（评测集泄漏），或基准被刷到饱和，排名相关性就会悄悄失效，而绝对分看起来依旧漂亮。

## 缩小评测 gap 的三条工程路线

在「不假装仿真等于真机」的前提下，让仿真评测结论更能外推：

1. **real-to-sim 相关性锚定**：用少量真机 rollout 校准仿真评测，报告 **sim↔real 排名相关性** 而非仿真绝对分；训练管线**刻意不与评测共享同一仿真分布**，堵住评测集泄漏。
2. **落到真实低层控制**：把「魔法抓取」等抽象化实现替换为真实接触/力控（如 [ManiSkill-HAB](../entities/paper-notebook-maniskill-hab-a-benchmark-for-low-level-manipula.md) 的低层控制），减少「简化换来的虚高」。
3. **分布外 + 长尾分层评测**：加入 OOD 测试集与按失败模式/物体分层的成功率，避免用均值成功率和饱和榜单掩盖长尾崩溃与分布漂移。

## 常见误判速查

| 误判 | 真相 |
|------|------|
| 仿真榜单接近满分 = 真实场景就绪 | 基准饱和 ≠ 场景就绪，多半是分布太窄或泄漏 |
| 仿真绝对成功率高 = 真机会成 | 该看 sim↔real 排名相关性，而非绝对分 |
| 仿真更逼真就一定评得更准 | 视觉逼真 ≠ 动作忠实，评测质量由后者主导 |
| 评测集和训练同源没关系 | 共享分布=评测集泄漏，排名相关性会虚高失效 |

## 关联页面

- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页是其「④ sim↔real 校准层」的取舍根因，双向回链
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) — 用可信仿真当闭环评测引擎的前提正是本页的 sim↔real 相关性
- [Sim2Real](../concepts/sim2real.md) — 策略迁移 gap，本页评测 gap 与其同源不同落点
- [物理保真度 sim2real gap](../concepts/physics-fidelity-sim2real-gap.md) — 接触/物理层面 gap 的物理根因
- [GigaWorld-1（世界模型策略评估器）](../entities/paper-gigaworld-1-policy-evaluation.md) — 「动作忠实 > 视觉逼真」是评测代表性的直接证据

## 参考来源

- [wechat_embodied_ai_lab_robot_training_stack_layers_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md) — 训练栈分层视角下「仿真作可扩展闭环评测、real-to-sim 相关性」的论述
- [robo_bench_arxiv_2510_17801.md](../../sources/papers/robo_bench_arxiv_2510_17801.md) — RoboBench 对「认知代理指标↔下游真机成功率相关」的专门验证，佐证上层易测指标外推需证据
