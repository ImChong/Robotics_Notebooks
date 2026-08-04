---
type: entity
tags: [paper, humanoid, world-model, locomotion, denoising, hmi-papers, humanoid-paper-notebooks]
status: complete
updated: 2026-07-31
arxiv: "2408.14472"
venue: "HMI curated · Paper Notebooks"
related:
  - ../concepts/world-action-models.md
  - ../methods/dreamwaq.md
  - ../tasks/humanoid-locomotion.md
  - ./paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md
  - ../queries/hmi-papers-coverage.md
  - ../overview/paper-notebook-category-05-locomotion.md
sources:
  - ../../sources/papers/hmi_p017_denoising-world-model-locomotion.md
  - ../../sources/papers/humanoid_pnb_advancing-humanoid-locomotion-mastering-challeng.md
  - ../../sources/repos/humanoid-motion-intelligence.md
summary: "Denoising World Model Locomotion（arXiv:2408.14472，HMI P017）：用去噪世界模型在复杂地形上学习人形运动表征/策略，强调对噪声观测与地形不确定性的鲁棒性。"
---

# Denoising World Model Locomotion（HMI P017）

**Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning**（[arXiv:2408.14472](https://arxiv.org/abs/2408.14472)）收录于 HMI **P017**，并已在 Paper Notebooks locomotion 分类占位。本页为该 arXiv 的**唯一详情节点**。

## 一句话定义

用去噪世界模型在复杂地形上学习人形运动表征/策略，强调对噪声观测与地形不确定性的鲁棒性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 学习环境/身体动态用于想象或表征 |
| PPO | Proximal Policy Optimization | 策略优化常用算法 |
| Sim2Real | Simulation to Real | 仿真到真机 |
| RL | Reinforcement Learning | 策略学习主线 |

## 为什么重要

- 把「世界模型去噪」接到人形复杂地形 locomotion，而不是只做视频生成。
- 与 DreamPolicy / DreamWaQ 同属世界模型运动线，便于对照「生成状态 vs 隐式地形」。
- HMI 与 Paper Notebooks 双入口共用本页，避免重复节点。

## 核心原理

去噪世界模型从带噪本体/地形观测中恢复对控制有用的潜在动态，再支撑策略在挑战地形上决策。阅读时抓住：输入噪声模型、去噪目标、以及策略如何消费世界模型表征。

```mermaid
flowchart LR
  A["带噪观测"] --> B["去噪世界模型"]
  B --> C["潜在动态 / 表征"]
  C --> D["Locomotion 策略"]
  D --> E["关节目标 / 真机"]
```

## 方法栈与流程

把上面的核心原理拆成可核对的流水线（各步细节以 PDF 为准）：

1. **观测建模**：明确本体/地形观测的噪声与遮挡模型，划定训练分布内的扰动范围。
2. **去噪世界模型**：从带噪观测中恢复对控制有用的潜在动态/表征，去噪目标是把观测噪声与地形不确定性从表征里剥离。
3. **策略消费表征**：locomotion 策略以去噪后的潜在动态/表征为输入，在挑战地形上做决策（RL/PPO 主线）。
4. **动作执行**：输出关节目标经真机执行，形成闭环。

阅读顺序：先抓「输入噪声模型 → 去噪目标 → 策略如何消费世界模型表征」，再谈网络结构与数值。

## 工程实践

| 检查项 | 建议 |
|--------|------|
| 开源 | 以项目页/作者声明再核 |
| 对照 | 与 DreamPolicy、感知跑酷线分开记账 |
| 一手来源 | arXiv PDF + HMI P017 |

## 源码运行时序图

**不适用**（截至本库升格日未绑定单一可运行官方训练仓为复现入口）。

## 实验与评测读法

- 分清仿真地形通过率与真机证据。
- 关注噪声/遮挡设定是否匹配部署传感器。

## 结论

**P017 应作为「去噪世界模型 → 人形复杂地形」节点阅读，并与 DreamPolicy 对照问题接口。**

- 先核对噪声模型与观测契约，再谈策略结构。
- 本页同时承接 Paper Notebooks 占位，不再另建平行实体。
- 开源与数值以一手来源为准。

## 局限与风险

- 计划索引阶段遗留信息可能过时；以 PDF 为准。
- 世界模型误差会传导到策略；需看失败模式是否被报告。

## 与其他工作对比

| 维度 | 本工作（去噪 WM Loco / P017） | [DreamPolicy / P018](./paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md) | [DreamWaQ](../methods/dreamwaq.md) | [World Action Models](../concepts/world-action-models.md) |
|------|------------------------------|-------------------------------------------------------------------------------------------|------------------------------------|-----------------------------------------------------------|
| 方法族 | 去噪世界模型 + RL 策略 | 自回归扩散世界模型 + goal-conditioned RL | 隐式地形想象（本体编码器）单阶段 RL | 世界-动作模型族（模型视角综述） |
| 世界模型角色 | 从带噪观测**去噪**出潜在动态/表征 | **生成**未来身体状态轨迹作参考 | CENet 从本体历史**推断**隐式地形上下文 | 联合建模世界演化与动作 |
| 关键假设 | 观测噪声/地形不确定性可被显式建模并剥离 | 多地形专家数据可训出可生成的 WM | 盲走：仅本体历史即可支撑鲁棒行走 | 世界预测与动作可耦合学习 |
| 输入/输出 | 带噪本体/地形观测 → 关节目标 | 专家数据 → 未来状态 → 统一跟踪策略 | 本体历史 → 隐式上下文 + 体速 → 关节目标 | 观测/动作 → 未来预测 + 动作 |
| 关系/取舍 | 强调鲁棒表征而非生成 | 强调统一多地形、减少奖励工程 | 无外感知基线，最轻量 | 提供族谱坐标，非单一系统 |

## 关联页面

- [DreamPolicy / One Policy but Many Worlds](./paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md)
- [World Action Models](../concepts/world-action-models.md)
- [HMI 论文导读](../queries/hmi-papers-coverage.md)

## 参考来源

- [hmi_p017_denoising-world-model-locomotion.md](../../sources/papers/hmi_p017_denoising-world-model-locomotion.md)
- [humanoid_pnb_advancing-humanoid-locomotion-mastering-challeng.md](../../sources/papers/humanoid_pnb_advancing-humanoid-locomotion-mastering-challeng.md)
- [humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)

## 推荐继续阅读

- [arXiv:2408.14472](https://arxiv.org/abs/2408.14472)
- [HMI P017](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P017.md)
