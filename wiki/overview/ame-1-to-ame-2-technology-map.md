---
type: overview
tags: [overview, perceptive-locomotion, ame, quadruped, biped, technology-map, eth]
status: complete
updated: 2026-09-10
related:
  - ../entities/paper-ame-attention-based-map-encoding.md
  - ../entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md
  - ../concepts/terrain-adaptation.md
  - ../concepts/privileged-training.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
sources:
  - ../../sources/blogs/wechat_embodied_station_ame1_ame2_2026-09-10.md
  - ../../sources/raw/wechat_embodied_station_ame1_ame2_2026-09-10.md
summary: "依据具身智能之心 2026-09-10 对照文，把 AME-1 与 AME-2 读成「局部 query → 全局+感知闭环 → Teacher–Student 部署」两节点演进轴。"
---

# AME-1 → AME-2：感知运动系统演进坐标

> **本页定位**：为 [具身智能之心 · AME-1 到 AME-2 改变了什么](https://mp.weixin.qq.com/s/VU_JcNP2FZrITTUoA8IDuA)（2026-09-10）提供 **按两代论文组织的阅读坐标**；方法细节见各 `paper-*` 实体页。

## 一句话观点

**AME-1 用本体条件注意力解决稀疏落脚；AME-2 在同一编码思想上补齐全局语境、在线不确定性感知映射与 Teacher–Student 部署一致性，把能力从「精确踩过去」推进到「混合地形里选对技能并穿过去」。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AME-1 | Attention-based Map Encoding (v1) | 本体 query + 两阶段 PPO |
| AME-2 | Attention-based Map Encoding (v2) | 全局+本体 query + 神经映射 |
| MHA | Multi-Head Attention | 对逐点地图特征加权 |
| GT | Ground Truth | Teacher 用仿真特权高程 |

## 为什么单独做这张地图

- 文内同时对比 **编码器、感知栈、任务接口、训练范式与未见地形数字**，读者易把「加全局池化」与「系统级闭环」混为一谈。
- **2/2 独立详情节点**（均复用既有 `paper-*`）；**0 重复 arXiv**。

## 流程总览

```mermaid
flowchart LR
  A1["01 AME-1\nproprio query · 速度跟踪\n两阶段 PPO · 给定地图"]
  A2["02 AME-2\nglobal+proprio query · 目标到达\nTeacher→Student · 在线映射"]
  A1 -->|"全局语境 + 记忆 + 不确定性"| A2
  A2 --> DEP["实机：训练栈=部署栈"]
```

## 系统级对照（文内 + 论文）

| 维度 | AME-1 | AME-2 |
|------|-------|-------|
| **Query** | 本体 + 速度 → query local map | **global ∥ proprio** → query local map |
| **地图来源** | 经典 elevation / 给定高程 | **深度→局部高程+方差→全局融合** |
| **记忆** | 瞬时地图输入 | **显式全局地图**（可诊断、可更新） |
| **任务** | **速度跟踪** | **目标到达**（中间轨迹更自由） |
| **训练** | 理想感知 Stage1 → 噪声 Stage2 | **Teacher（GT 地图）→ Student（在线映射）** |
| **未见混合地形（文内均值）** | **51.2%** | teacher **95.2%** / student **82.4%** |

## 分组索引

| # | 资料 | 节点类型 | 开源（入库日） | 详情 |
|---|------|----------|---------------|------|
| 01 | Attention-Based Map Encoding | 论文 2506.09588 / *Sci. Rob.* | 官方未发布；Zenodo 数据；社区 G1 复现 | [paper-ame-attention-based-map-encoding](../entities/paper-ame-attention-based-map-encoding.md) |
| 02 | AME-2 | 论文 2601.08485 | 官方未发布；社区 ANYmal-D 复现 | [paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi](../entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md) |

## 关联页面

- [Terrain Adaptation](../concepts/terrain-adaptation.md)
- [Privileged Training](../concepts/privileged-training.md)
- [楼梯与障碍 Locomotion](../tasks/stair-obstacle-perceptive-locomotion.md)
- [Extreme Parkour](../entities/extreme-parkour.md) — 敏捷对照轴

## 参考来源

- [wechat_embodied_station_ame1_ame2_2026-09-10.md](../../sources/blogs/wechat_embodied_station_ame1_ame2_2026-09-10.md)
- [ame-2-leggedrobotics.md](../../sources/sites/ame-2-leggedrobotics.md)

## 推荐继续阅读

- [AME-2 项目页](https://sites.google.com/leggedrobotics.com/ame-2)
- [Kitjesen/ame2](https://github.com/Kitjesen/ame2) — 社区 ANYmal-D 复现（非官方）
