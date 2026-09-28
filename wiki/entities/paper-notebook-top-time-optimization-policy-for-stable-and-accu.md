---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, whole-body-control, reinforcement-learning, motion-prior, vae, zju]
status: complete
updated: 2026-09-28
arxiv: "2508.00355"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../tasks/loco-manipulation.md
  - ./paper-hrl-stack-08-omnih2o.md
  - ./paper-exbody-expressive-humanoid.md
  - ./paper-loco-manip-161-138-mobile-television.md
  - ../methods/centroidal-nmpc-wbc-stack.md
sources:
  - ../../sources/papers/humanoid_pnb_top.md
summary: "人形能做多样操作，前提是鲁棒精确的站立控制器。已有方法要么难精控高维上身关节、要么难同时保证鲁棒与精度——尤其当上身运动快时。本文提出一个新颖的时间优化策略（Time Optimization Policy, TOP），训练一个站立操作控制模型，同时保证平衡、精度与时间效率。核心思想是：调整上身动作的时间轨迹，而不只是一味强化下身的抗扰能力——让快速上身运动在时间上\"错峰\"，减轻对平衡的冲击。方法用 VAE 编码上身动作先验，并解耦全身控制（上身 PD 控制器 + 下身 RL 控制器）。仿真与真机实验表明，TOP 在站立操作上稳定且精确，优于已有方法。"
---

# TOP

**TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

人形能做多样操作，前提是鲁棒精确的站立控制器。已有方法要么难精控高维上身关节、要么难同时保证鲁棒与精度——尤其当上身运动快时。本文提出一个新颖的时间优化策略（Time Optimization Policy, TOP），训练一个站立操作控制模型，同时保证平衡、精度与时间效率。核心思想是：调整上身动作的时间轨迹，而不只是一味强化下身的抗扰能力——让快速上身运动在时间上"错峰"，减轻对平衡的冲击。方法用 VAE 编码上身动作先验，并解耦全身控制（上身 PD 控制器 + 下身 RL 控制器）。仿真与真机实验表明，TOP 在站立操作上稳定且精确，优于已有方法。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| TOP | Time Optimization Policy，时间优化策略 |
| Standing Manipulation | 站立操作 |
| VAE | 变分自编码器（编码上身动作先验） |
| Decoupled WBC | 解耦全身控制（上身 PD + 下身 RL） |
| Time Trajectory | 时间轨迹（动作的时间安排） |
| Disturbance Resistance | 抗扰能力 |

## 为什么重要

- **"调时间"是平衡-精度权衡的新维度**：不止调空间动作，还可调时间安排；
- **上身 PD + 下身 RL 解耦**契合"精确 vs 鲁棒"的不同需求，与 Mobile-TeleVision 思路相通；
- **VAE 动作先验**是常用的紧凑表示手段；
- 站立操作是人形干活的基础，稳准快都重要。

## 解决什么问题

站立操作要**平衡 + 精度 + 时间效率**三者兼顾： - 难**精控高维上身关节**； - **上身快速运动**时，扰动大，难同时稳与准； - 一味强化下身抗扰**治标不治本**。

TOP 要：通过**调上身动作时间轨迹**，从源头减轻平衡负担，兼顾稳、准、快。

## 核心机制

1. **时间优化策略 TOP**：调上身动作时间轨迹，同时保证稳/准/快；
2. **VAE 上身动作先验**：紧凑可优化表示；
3. **解耦全身控制**：上身 PD 精控 + 下身 RL 鲁棒；
4. **稳定精确站立操作**：仿真 + 真机优于已有方法。

方法拆解（深读笔记小节）：思想：调上身时间轨迹（而非只强化下身）；VAE 上身动作先验；解耦全身控制；时间优化策略训练；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2508.00355> |
| 源码 | **未开源**：论文附匿名链接 anonymous.4open.science/w/top-258F/，2026-09-28 访问返回 410（已失效），未见正式仓库 |
| 作者 | Zhenghan Chen、Haocheng Xu、Haodong Zhang、Zhongxiang Zhou、Rong Xiong 等（浙江大学） |
| 发表 | 2025 年 8 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：1.65 m / 60 kg / 41 DoF 全尺寸人形（双 7-DoF 臂，每臂负载 3 kg）；上身动作先验 VAE 在 GRAB 人体动作数据上训练（潜变量 64 维、窗口 30 帧 = 0.6 s），另构造 1.6 万条更大幅度、更快的测试动作集 T，均经重定向。下身为带域随机化的 RL 平衡策略；TOP 为三层 MLP actor-critic，输出每步时间间隔 Δt ∈ [0.01, 0.1] s。评测执行 1 万条以上动作片段，误差只在机器人原地站立时统计。

| 方法 | 用时 | 成功率 | 上身关节误差 | 末端位置误差 | 重力误差 E_g |
|------|---:|---:|---:|---:|---:|
| 固定基座（参考上限） | 15.0 s | 100% | 0.0130 | 0.0164 | 1.000 |
| ExBody | 15.0 s | 92.46% | 0.0376 | 0.0741 | 3.432 |
| OmniH2O | 15.0 s | 94.08% | 0.0361 | 0.0506 | 3.267 |
| Mobile-TeleVision（复现） | 15.0 s | 85.79% | 0.0354 | 0.0513 | 3.831 |
| NMPC + WBC（放慢动作） | 35.0 s | 89.60% | 0.0278 | 0.0437 | 2.938 |
| **TOP** | 40.5 s | **95.30%** | **0.0269** | **0.0270** | **2.729** |

- **消融**：去掉 TOP、固定 Δt = 0.01 / 0.03 / 0.05 s，成功率 82.43% / 87.27% / 92.41%，用时 15 / 45 / 75 s——单纯放慢也能变稳，但比 TOP 更慢、更不准；去掉动作先验 89.33%；去掉动作块 93.16%。
- **解耦策略鲁棒性**：未见的更快、更大幅度动作偶尔会摔（尤其双臂举过头顶的大动量变化），配合放慢后成功率超过 80%。
- **动作先验泛化**：已见与未见动作在潜空间 50% 线性插值后仍可重建合理动作。

## 与其他工作对比

| 方法 | 如何处理「稳 vs 准」 | 与 TOP 的差异 |
|------|------|------|
| [ExBody](./paper-exbody-expressive-humanoid.md) / [OmniH2O](./paper-hrl-stack-08-omnih2o.md) | 全身 RL 跟踪奖励 | 快（15 s）但上身精度低、平衡误差大 |
| [Mobile-TeleVision](./paper-loco-manip-161-138-mobile-television.md) | 上下身解耦 + 动作先验 | 相当于 TOP 去掉时间优化的版本，成功率 85.79% |
| [NMPC + WBC](../methods/centroidal-nmpc-wbc-stack.md) | 模型预测 + 全身控制 | 放慢后 89.60%；TOP 用学习的时间调度取得更高成功率与精度 |
| 固定放慢（消融） | 统一加大 Δt | 需 75 s 才到 92.41%，TOP 40.5 s 达 95.30% |

## 结论

**TOP 的核心主张是把「稳 vs 准」的权衡从空间维度挪到时间维度：与其不断加强下身抗扰，不如让上身快速动作在时间上错峰，从源头减少对平衡的冲击。**

- 真正起作用的机制是 **时间轨迹优化 + 解耦控制** 的组合：上身用 PD 换精度、下身用 RL 换鲁棒，VAE 上身动作先验则提供一个紧凑、可被优化的时间轨迹表示。
- 这条思路的隐含前提是 **上身动作的时间安排可以被调整**；若任务对动作时序有外部约束（必须按固定节拍完成），"错峰"这一自由度就不存在，方法收益随之下降。
- 适用边界是 **站立操作**：论文处理的是站立姿态下的平衡-精度-时效三角，本页未涉及行走中操作或移动底座场景。
- 与「一味强化下身抗扰」的路线相比，TOP 明确把后者判为治标不治本；与 Mobile-TeleVision 的相通之处在于同样承认上身与下身对"精确 vs 鲁棒"有不同需求，应分开设计。
- 量化上 TOP 以 95.30% 成功率与最低的末端误差（0.0270）领先，但代价是用时 40.5 s；固定放慢 Δt 虽也能提高成功率，却要 75 s 才到 92.41%，说明收益来自「按动作难度分配时间」而非单纯放慢。

## 局限与风险

- **时间换稳定**：TOP 用时 40.5 s，约为全身 RL 基线（15 s）的 2.7 倍；对节拍有硬约束的任务不适用。
- **动作偏僵硬**：机器人倾向于后仰而不是扭髋来平衡（论文结论节自述）。
- **只覆盖站立操作**：误差仅在原地站立时统计，不涉及行走中操作。
- **未见大幅动作仍会摔**：双臂过头、速度过快等大动量动作是主要失败源。
- **开源边界**：匿名链接已失效，未见代码；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 任务语境：站立 / 移动操作：[loco-manipulation](../tasks/loco-manipulation.md)
- 对比基线 OmniH2O：[paper-hrl-stack-08-omnih2o](./paper-hrl-stack-08-omnih2o.md)
- 对比基线 ExBody：[paper-exbody-expressive-humanoid](./paper-exbody-expressive-humanoid.md)
- 上下身解耦的对比基线 Mobile-TeleVision：[paper-loco-manip-161-138-mobile-television](./paper-loco-manip-161-138-mobile-television.md)
- NMPC + WBC 对照路线：[centroidal-nmpc-wbc-stack](../methods/centroidal-nmpc-wbc-stack.md)

## 参考来源

- [humanoid_pnb_top.md](../../sources/papers/humanoid_pnb_top.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html>
- 论文：<https://arxiv.org/abs/2508.00355>
- 论文正文（Table II–III、结论）：<https://arxiv.org/html/2508.00355>

## 推荐继续阅读

- [机器人论文阅读笔记：TOP](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html)
