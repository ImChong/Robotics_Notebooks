---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "1410.1465"
related:
  - ../overview/paper-notebook-category-09-state-estimation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_the-invariant-extended-kalman-filter-as-a-stable.md
summary: "标准 EKF 在非线性系统上把动力学「就地线性化」，线性化质量取决于当前估计，估计一偏、线性化就坏、滤波就可能发散。本文从确定性观测器的视角重新审视 不变 EKF（InEKF）：作者定义了一类「群仿射（group-affine）」系统，证明在这类系统上，建立在李群上的估计误差演化是自治的（不依赖具体轨迹/估计），并且满足精确的对数线性微分方程；由此推出 InEKF 在标准可观测性条件下局部渐近收敛——也就是说它是一个有理论保证的「稳定观测器」，而经典 EKF 没有这种保证。"
---

# The Invariant Extended Kalman Filter as a Stable Observer

**The Invariant Extended Kalman Filter as a Stable Observer** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：09_State_Estimation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

标准 EKF 在非线性系统上把动力学「就地线性化」，线性化质量取决于当前估计，估计一偏、线性化就坏、滤波就可能发散。本文从确定性观测器的视角重新审视 不变 EKF（InEKF）：作者定义了一类「群仿射（group-affine）」系统，证明在这类系统上，建立在李群上的估计误差演化是自治的（不依赖具体轨迹/估计），并且满足精确的对数线性微分方程；由此推出 InEKF 在标准可观测性条件下局部渐近收敛——也就是说它是一个有理论保证的「稳定观测器」，而经典 EKF 没有这种保证。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| EKF | Extended Kalman Filter | 扩展卡尔曼滤波，对非线性系统在估计点处一阶线性化后套用 KF |
| InEKF / IEKF | Invariant EKF | 不变扩展卡尔曼滤波，状态建模在李群上、用「不变误差」而非欧氏误差 |
| Lie Group | - | 李群，既是群（有乘法/逆）又是光滑流形（如 SO(3)、SE(3)、SE_2(3)） |
| Group-affine | 群仿射 | 一类动力学：其结构使得不变误差的演化与当前状态无关 |
| Log-linear | 对数线性 | 误差在李代数中（经对数映射）的演化满足精确线性微分方程 |
| Left/Right-invariant error | 左/右不变误差 | 用群乘法定义的误差 η = ĝ⁻¹g（左）或 gĝ⁻¹（右），而非 x − x̂ |

## 为什么重要

- **足式状态估计的理论根**：[Contact-Aided InEKF (1904.09251)](https://arxiv.org/abs/1904.09251) 把本文理论落到足式机器人——base 位姿/速度 + 接触点统一建模在 SE_{n+2}(3) 上，正是「群仿射 + 对数线性」的直接应用
- **为何 InEKF 成标配**：MIT Cheetah、Unitree 等的本体感知状态估计器普遍采用 InEKF，根源就是本文给出的「线性化不依赖估计 → 更大收敛域」
- **可观测性洞见**：本文框架下能清楚说明偏航角 / 绝对位置在仅本体感知时不可观，避免 EKF 产生「假修正」
- **学习型滤波的对照基线**：[InEKFormer (2511.16306)](https://arxiv.org/abs/2511.16306)、KalmanNet 等「保留几何骨架 + 网络学难调部分」的工作，保留的正是本文这套李群几何结构

## 解决什么问题

- **EKF 的一致性 / 收敛性没有保证**：经典 EKF 对非线性系统在「当前估计」处线性化。一旦估计有偏，雅可比就算错，滤波可能「过分自信」甚至发散。这在机器人定位、惯性导航这类大范围运动里尤其致命。 - **不变滤波此前缺乏严格的收敛分析**：人们已经观察到把状态放到李群上、用「不变误差」做 EKF（InEKF）经验上更稳，但**为什么稳、在什么条件下稳**，缺乏统一的理论说明。 - **本文的目标**：把 InEKF 当作**确定性非线性观测器**（先不谈噪声统计，只问「估计会不会收敛到真值」），刻画出一类能享受良好性质的系统，并给出可证明的局部稳定性结论。

## 核心机制

1. **群仿射系统的提出**：给出一个清晰的充分条件，刻画「哪些系统能让不变误差演化与状态无关」，把 InEKF 的优势从经验上升为可分析的结构性质。
2. **对数线性性质的证明**：非线性误差在李代数中满足精确线性 ODE，这是把线性收敛理论搬过来的关键桥梁。
3. **稳定观测器的收敛保证**：证明 InEKF 在标准可观测性条件下局部渐近收敛——经典 EKF 缺这一保证。
4. **统一视角**：把移动机器人定位、惯性导航等看似不同的问题，纳入同一套李群 + 不变误差的框架下处理。

方法拆解（深读笔记小节）：不变误差：用群结构定义「差」；群仿射（group-affine）系统：让误差「自治」；对数线性（log-linear）性质；作为稳定观测器的收敛证明；实验验证（示例层面）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 09_State_Estimation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer.html> |
| arXiv | <https://arxiv.org/abs/1410.1465> |
| 发表 | 2014-10-06 (arXiv v1)，2017 正式发表（IEEE TAC） |
| 源码 | 理论论文，无配套代码（后续 [RossHartley/invariant-ekf](https://github.com/RossHartley/invariant-ekf) 是工程实现参考） |
| 笔记阅读日期 | 2026-06-25 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-09-state-estimation](../overview/paper-notebook-category-09-state-estimation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_the-invariant-extended-kalman-filter-as-a-stable.md](../../sources/papers/humanoid_pnb_the-invariant-extended-kalman-filter-as-a-stable.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer.html>
- 论文：<https://arxiv.org/abs/1410.1465>

## 推荐继续阅读

- [机器人论文阅读笔记：The Invariant Extended Kalman Filter as a Stable Observer](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer.html)
