---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, safety, safe-control, collision-avoidance, quadratic-programming, cmu, unitree]
status: complete
updated: 2026-09-28
arxiv: "2502.02858"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../concepts/safety-filter.md
  - ../concepts/control-barrier-function.md
  - ../concepts/constrained-optimization.md
  - ../tasks/teleoperation.md
  - ./paper-crosssafe.md
sources:
  - ../../sources/papers/humanoid_pnb_dexterous-safe-control-for-humanoids-in-cluttere.md
summary: "在真实应用中确保人形安全且不牺牲性能至关重要。本文考虑灵巧安全（dexterous safety）问题，特点是肢体级（limb-level）几何约束，用于在杂乱环境中同时避免外部碰撞与自碰撞。为处理\"确保碰撞避免\"时产生的大量约束，提出投影安全集算法（Projected Safe Set Algorithm, p-SSA）；针对约束不可行（infeasibility）问题，以有原则的方式松弛冲突约束，最小化安全违例以保证可行的机器人控制。在仿真与 Unitree G1 真机上验证：p-SSA 能让人形在挑战性场景中稳健运行、最小违例，并能跨任务免调参泛化。"
---

# Dexterous Safe Control for Humanoids in Cluttered Environments via Projected Safe Set Algorithm

**Dexterous Safe Control for Humanoids in Cluttered Environments via Projected Safe Set Algorithm** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

在真实应用中确保人形安全且不牺牲性能至关重要。本文考虑灵巧安全（dexterous safety）问题，特点是肢体级（limb-level）几何约束，用于在杂乱环境中同时避免外部碰撞与自碰撞。为处理"确保碰撞避免"时产生的大量约束，提出投影安全集算法（Projected Safe Set Algorithm, p-SSA）；针对约束不可行（infeasibility）问题，以有原则的方式松弛冲突约束，最小化安全违例以保证可行的机器人控制。在仿真与 Unitree G1 真机上验证：p-SSA 能让人形在挑战性场景中稳健运行、最小违例，并能跨任务免调参泛化。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| p-SSA | Projected Safe Set Algorithm，投影安全集算法 |
| Dexterous Safety | 灵巧安全（肢体级几何约束） |
| Limb-level | 肢体级，按各肢体几何建约束 |
| Self-collision | 自碰撞（机体各部分相撞） |
| Infeasibility | 约束不可行（冲突） |
| Safety Violation | 安全违例 |

## 为什么重要

- **"约束太多会不可行"是安全控制的真问题**，有原则的松弛比硬失败更实用；
- **肢体级几何**对高自由度人形的自碰撞避免必不可少；
- **免调参跨任务**的安全层利于工程复用；
- 与学习类控制互补：安全控制作"护栏"，学习作"性能"。

## 解决什么问题

人形在**杂乱环境**操作要**安全**： - 需**肢体级**避**外部碰撞 + 自碰撞**； - 约束**数量巨大**且可能**互相冲突（不可行）**； - 安全不能太保守而**牺牲性能**。

论文要：一个能处理**大量、可能冲突**约束、**最小违例**且**保性能**的安全控制算法。

## 核心机制

1. **灵巧安全问题**：肢体级几何约束，避外部 + 自碰撞；
2. **p-SSA 算法**：投影安全集处理大量约束；
3. **有原则松弛冲突约束**：最小化违例、保证可行控制；
4. **真机验证 + 免调参泛化**：G1 杂乱场景稳健。

方法拆解（深读笔记小节）：灵巧安全：肢体级几何约束；p-SSA：投影安全集算法；有原则地松弛冲突约束；验证；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set.html> |
| arXiv | <https://arxiv.org/abs/2502.02858> |
| 源码 | **未开源**：论文与 arXiv 页未给出代码或项目页链接（截至 2026-09-28） |
| 作者 | Rui Chen、Yifan Sun、Changliu Liu（CMU） |
| 发表 | 2025 年 2 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：MuJoCo 中的 Unitree G1，两种动力学配置（固定基座 G1FixedBase / 全身 G1WholeBody）× 两类障碍（静态球 SO、布朗运动动态球 DO）× 两档障碍密度（V0 更多、V1 更少）= 8 个任务。机器人用手腕（全身配置下还有基座）跟踪不断刷新的 3D 目标点，同时与外部障碍保持 ≥ 0.05 m、自碰撞保持 ≥ 0.01 m。参考控制由不考虑安全的 PID 给出。

**比较方法**：朴素 SSA（QP 不可行时直接放行参考控制）、r-SSA（在目标里加松弛项，需调权重 Q_s）、p-SSA（把不可行约束投影成最小违例问题，无需调参）。每个任务 2000 步。指标：手臂跟踪得分 J、QP 不可行时的控制约束满足度 C、安全裕度侵入程度 S、QP 可行率 R_Feas（正文以柱状图报告，未列数值表）。

- **不可行有多常见**：障碍越多、固定基座（缺少移动性）、障碍在动，QP 越容易不可行。朴素 SSA 的可行率反而常更高——它在不可行时直接穿过障碍，活动约束变少。
- **违例最小化**：朴素 SSA 在不可行时违例显著；r-SSA 与 p-SSA 都能让肢体贴近障碍而不碰撞。
- **调参消融**：在 λ = a·10^b（a ∈ [1,9]，b ∈ [0,5]）范围内扫 r-SSA 的 Q_s，各任务的性能–安全帕累托前沿差异很大，调参繁琐且不可迁移；p-SSA 无需调参即落在各任务前沿的较优区域。
- **真机**：Apple Vision Pro 遥操作 G1 双臂伸进窄口柜子整理物品，操作者故意做危险动作，p-SSA 拒绝不安全的参考控制，仿真与真机均演示可行（定性）。

## 与其他工作对比

| 方法 | QP 不可行时的处理 | 与 p-SSA 的差异 |
|------|------|------|
| 朴素 SSA | 直接执行参考控制 | 安全完全失守 |
| r-SSA（本文） | 目标中加权松弛 | 可达到类似权衡，但需逐任务调 Q_s |
| [控制屏障函数 / 安全过滤器](../concepts/safety-filter.md) | 通常假设约束可行 | 多约束冲突时同样会不可行；本文正面处理这一情形 |
| 学习类安全策略 | 用数据学避障 | 无显式约束；p-SSA 可作为挂在任意上层控制器后的安全层 |

## 结论

**p-SSA 的贡献不是「更保守的安全层」，而是承认肢体级约束必然互相冲突，把「不可行」从硬失败改写成一个有原则的最小违例问题。**

- 真正起作用的是两件事：**肢体级几何约束**（外部碰撞与自碰撞一并建模）解决高自由度人形的自碰撞盲区，**投影安全集**把由此产生的海量约束压到可解。
- 核心取舍是**松弛而非放弃**：约束冲突时最小化安全违例以保证控制仍可行——代价是安全从硬保证降为「最小违例」，用在强安全等级场景前要认清这一点。
- 定位边界是「护栏」而非完整方案：性能仍由上层（学习类）控制器提供，本方法应被当成可挂载的安全层来复用。
- 工程价值集中在**免调参跨任务**：仿真与 Unitree G1 真机在杂乱场景验证，同一套参数迁移到不同任务，这是安全层能否被工程复用的现实门槛。
- 量化结果只以柱状图报告：结论是 r-SSA 调对参数也能达到类似权衡，但各任务帕累托前沿差别大、无法沿用同一参数，p-SSA 免调参即可落在较优区域。

## 局限与风险

- **松弛后没有安全保证**：只要需要松弛，违例就无法有界，硬安全保证不再成立（论文自述）。
- **结果以图呈现**：8 个任务的 J / C / S / R_Feas 只给柱状图，没有数值表。
- **几何简化**：障碍与目标用球体、柜子用平面近似；真机中人体与障碍位置由 Vision Pro 感知。
- **一阶动力学模型**：安全指数按一阶（速度级）动力学设计。
- **开源边界**：未见代码；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 安全过滤器：[safety-filter](../concepts/safety-filter.md)
- 控制屏障函数（同类安全约束）：[control-barrier-function](../concepts/control-barrier-function.md)
- 约束优化 / QP：[constrained-optimization](../concepts/constrained-optimization.md)
- 真机验证场景：安全遥操作：[teleoperation](../tasks/teleoperation.md)
- 人形安全控制的另一路线：[paper-crosssafe](./paper-crosssafe.md)

## 参考来源

- [humanoid_pnb_dexterous-safe-control-for-humanoids-in-cluttere.md](../../sources/papers/humanoid_pnb_dexterous-safe-control-for-humanoids-in-cluttere.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set.html>
- 论文：<https://arxiv.org/abs/2502.02858>
- 论文正文（实验设置、指标、消融与局限）：<https://arxiv.org/html/2502.02858>

## 推荐继续阅读

- [机器人论文阅读笔记：Dexterous Safe Control for Humanoids in Cluttered Environments via Projected Safe Set Algorithm](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set.html)
