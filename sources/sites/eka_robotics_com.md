# Eka Robotics 官网（ekarobotics.com）

> 来源归档

- **标题：** Eka Robotics — Vision-Force-Action (VFA)
- **类型：** site（公司 / 项目页）
- **链接：** https://www.ekarobotics.com/
- **联系：** hello@ekarobotics.com
- **入库日期：** 2026-09-26
- **一句话说明：** 剑桥创业团队公开的 **Vision-Force-Action（VFA）** 操作基础模型叙事：以 **力** 为物理世界「母语」，宣称同时追求 **通用性、性能（速度×可靠性）与人机安全**；强调仿真规模化 RL 与自研触觉机械手，目标 **超越人类** 的操作吞吐而非仅模仿人类视频。
- **沉淀到 wiki：** 是 → [`wiki/entities/eka-robotics-vfa.md`](../../wiki/entities/eka-robotics-vfa.md)

## 开源核查（2026-09-26）

| 项 | 状态 |
|----|------|
| 项目页 Code / GitHub | **未列链接** |
| 模型权重 / 数据集 | **未发布** |
| 演示视频 | 站内含 **1/25× SPEED** 等演示片段（无下载） |

**结论：** **未开源**（截至入库日仅有品牌站与公开演讲/媒体报道；无官方代码仓或模型卡）。

## 核心摘录（官网文案）

### 定位

- 「We are building intelligence for the physical world in its native language: **force**.」
- 「Until now, robotics required choosing: **generality or performance**. Our **Vision-Force-Action (VFA)** model changes that.」
- 新基础模型联合 **generality, performance, and safety**，推动机器人 **beyond human limits**。

### 三支柱（Built to scale in the real world）

| 支柱 | 官网表述 |
|------|----------|
| **GENERAL** | 单一模型跨物体、任务、环境泛化 |
| **Performant** | 掌握物理：快速、可靠、适应意外 |
| **Interactive** | 与人协作：安全、环境无关 |

### 团队（官网 TEAM 段）

- 背景涵盖自监督学习、大规模机器人学习、世界模型、强化学习、sim-to-real；成员来自 **MIT, Berkeley, Harvard, CMU, BU, UPenn, DeepMind, Microsoft, Boston Dynamics** 等（官网列举，非完整名单）。

## 对 wiki 的映射

- 与 [VLA 方法页](../../wiki/methods/vla.md) 对照：**力为一等模态 + 仿真 RL** 的 **并行 foundation 路线**（非 VLA 扩展）。
- 交叉 [sim2real](../../wiki/concepts/sim2real.md)、[触觉操作 T-Rex](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md)、[Foundation Policy 概念](../../wiki/concepts/foundation-policy.md)。
