---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2602.00678"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_toward-reliable-sim-to-real-predictability-for-m.md
summary: "RL 在四足敏捷运动上很有前景，即便仅本体感受也行。但实践中sim-to-real 差距与复杂地形上的奖励过拟合会让策略迁移失败，而物理验证又风险高、低效。本文提出一个统一框架：① 一个专家混合（MoE）运动策略，用门控的专家集合把隐式地形与指令建模分解，仅靠本体感受实现更优的部署鲁棒性与泛化；② RoboGauge ——一个预测性评估套件，量化 sim-to-real 可迁移性，通过跨地形、难度、域随机化的sim-to-sim 测试给出多维本体感受指标，使无需大量真机试验即可可靠地选 MoE 策略。在 Unitree Go2 上：雪、沙、楼梯、斜坡、30cm 障碍等未见地形稳健通行；高速测试达 4 m/s，并涌现与高速稳定相关的窄步态。"
---

# Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion

**Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：05_Locomotion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

RL 在四足敏捷运动上很有前景，即便仅本体感受也行。但实践中sim-to-real 差距与复杂地形上的奖励过拟合会让策略迁移失败，而物理验证又风险高、低效。本文提出一个统一框架：① 一个专家混合（MoE）运动策略，用门控的专家集合把隐式地形与指令建模分解，仅靠本体感受实现更优的部署鲁棒性与泛化；② RoboGauge ——一个预测性评估套件，量化 sim-to-real 可迁移性，通过跨地形、难度、域随机化的sim-to-sim 测试给出多维本体感受指标，使无需大量真机试验即可可靠地选 MoE 策略。在 Unitree Go2 上：雪、沙、楼梯、斜坡、30cm 障碍等未见地形稳健通行；高速测试达 4 m/s，并涌现与高速稳定相关的窄步态。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| MoE | Mixture-of-Experts，专家混合 |
| RoboGauge | 本文的 sim-to-real 可迁移性评估套件 |
| Proprioception | 本体感受，仅用内部状态感知 |
| Sim-to-Sim | 仿真到仿真测试，用于预测迁移性 |
| Gating | 门控，按情境选择/混合专家 |
| Emergent Gait | 涌现步态，训练中自发出现的步态 |

## 为什么重要

- **"预测可迁移性"是 sim-to-real 被忽视的一环**：与其试错真机，不如用 sim-to-sim 指标筛选——对人形（真机更贵更危险）尤其有价值；
- **MoE 按地形/指令分解**是鲁棒多地形的有效结构，呼应 EGM 的专家分设；
- 虽为四足，但**评估方法论与 MoE 思想可迁移到人形**；
- **涌现步态**提示 RL 能自发找到与稳定相关的运动模式。

## 解决什么问题

四足 RL 运动的痛点： - **sim-to-real 差距 + 奖励过拟合** → 迁移失败； - **物理验证风险高、低效** → 难以判断哪个策略能迁移。

论文要：① 更鲁棒的多地形策略（仅本体感受）；② 一个**无需大量真机**就能**预测迁移性**、可靠选策略的评估方法。

## 核心机制

1. **MoE 鲁棒运动策略**：门控专家分解地形/指令，仅本体感受多地形泛化；
2. **RoboGauge 可迁移性评估**：sim-to-sim 多维指标预测 sim-to-real，免大量真机；
3. **可靠选策略**：把"哪个策略能迁移"变成可预测问题；
4. **Go2 实测**：多难地形稳健、4 m/s、涌现窄步态。

方法拆解（深读笔记小节）：MoE 运动策略（仅本体感受）；RoboGauge：可迁移性预测评估；结果（Unitree Go2）；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion.html> |
| arXiv | <https://arxiv.org/abs/2602.00678> |
| 作者 | Tianyang Wu、Hanwei Guo、Yuhang Wang、Junshu Yang、Xinyang Sui 等（西安交大等） |
| 发表 | 2026 年 1 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_toward-reliable-sim-to-real-predictability-for-m.md](../../sources/papers/humanoid_pnb_toward-reliable-sim-to-real-predictability-for-m.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion.html>
- 论文：<https://arxiv.org/abs/2602.00678>

## 推荐继续阅读

- [机器人论文阅读笔记：Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion.html)
