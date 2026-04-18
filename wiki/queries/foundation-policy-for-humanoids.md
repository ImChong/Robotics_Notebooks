---
type: query
tags: [foundation-policy, humanoid, vla, locomotion, manipulation, loco-manipulation, generalization]
status: stub
summary: "综合分析人形机器人 foundation policy 的当前适用边界：操作任务已有初步可用，运动控制尚不成熟，loco-manipulation 仍是前沿。"
sources:
  - ../../sources/papers/rl_foundation_models.md
related:
  - ../concepts/foundation-policy.md
  - ../methods/vla.md
  - ../tasks/loco-manipulation.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
---

> **Query 产物**：本页由以下问题触发：「人形机器人 foundation policy 现在适合什么，不适合什么？」
> 综合来源：[Foundation Policy](../concepts/foundation-policy.md)、[VLA](../methods/vla.md)、[Loco-Manipulation](../tasks/loco-manipulation.md)、[Manipulation](../tasks/manipulation.md)、[Locomotion](../tasks/locomotion.md)

# 人形机器人 Foundation Policy 适用边界分析

## TL;DR 决策路径

```
你的任务是什么？
│
├── 桌面/固定底盘操作（抓取、排列、简单装配）
│   └── 已有大量演示数据？
│       ├── 是 → Foundation Policy 可以用，推荐 ACT / π₀ / RT-2 微调
│       └── 否 → 先收集数据（ALOHA 遥操作），数据量 < 50 条谨慎使用
│
├── 人形全身行走 / Locomotion
│   ├── 无语义任务，只是走路？ → 用 RL（PPO/SAC），foundation policy 目前无优势
│   └── 需要语言指令导航？ → 可以用 VLA 做高层规划 + 底层 RL 控制器
│
├── Loco-Manipulation（边走边操作）
│   ├── 操作是主体，行走是辅助（固定位置后操作）→ Foundation Policy 有一定可用性
│   └── 真正的同步行走+操作 → 目前 foundation policy 效果有限，建议 WBC/RL 混合
│
└── 精细操作（螺丝、接线、精密装配）
    └── 当前所有 foundation policy 精度不足 → 专用策略 + 力控
```

## 详细分析

### 当前 Foundation Policy 的能力边界

| 任务类型 | 代表模型 | 当前状态 | 适合程度 |
|---------|---------|---------|---------|
| 桌面 pick-and-place | RT-1, RT-2, Octo | 成熟，成功率 70-90% | ✅ 适合 |
| 语言条件操作（开放词汇） | RT-2, π₀ | 可用，泛化有限 | ✅ 适合（受限） |
| 多步骤操作任务 | π₀, ACT | 初步可用，链式失败率高 | ⚠️ 谨慎使用 |
| 双臂协调操作 | ACT, π₀ | 有基线，但泛化差 | ⚠️ 有限适用 |
| 双足行走 / Locomotion | 无成熟 foundation | 研究阶段 | ❌ 暂不适合 |
| 全地形行走 | 无 | 无成熟工作 | ❌ 暂不适合 |
| Loco-Manipulation（同步） | π₀（部分） | 非常早期 | ❌ 不成熟 |
| 精细力控操作 | 无 | 未解决 | ❌ 暂不适合 |

### Foundation Policy 适合的场景（Why it works）

**1. 操作任务中的语义理解**

Foundation Policy 的核心优势来自预训练的视觉-语言知识：理解"把红色杯子放到托盘右边"不需要从零学习，而是调用预训练的语义知识。这在操作任务（固定底盘，主要是上肢运动）中效果显著。

**2. 跨任务泛化**

经过大规模多任务预训练后（RT-2 使用 Web 数据，Octo 使用 Open X-Embodiment），模型对新任务的零样本/少样本适应能力明显优于单任务策略。

**3. 多模态指令理解**

"把那个蓝色的东西（视觉指向）放到左边"——同时处理视觉指代和语言指令是 VLA 的强项，传统 WBC/MPC 框架无法直接处理。

### Foundation Policy 不适合的场景（Why it fails）

**1. Locomotion 的根本性挑战**

当前 foundation policy 主要基于 Transformer 自回归生成动作序列，对**实时物理反馈**的敏感性不足：
- 行走需要 100Hz+ 的控制频率，foundation policy 推理延迟通常 10-50ms
- 步态稳定性对状态误差极度敏感，"大模型决策-小模型执行"的分层才是正确范式
- 没有足够的 locomotion 真机数据：桌面操作数据量级是行走数据的 100 倍以上

**2. 接触丰富场景的精度不足**

Foundation Policy 通常输出末端位姿或关节位置目标，对力/力矩缺乏精细控制。拧螺丝、插接头等精细接触任务需要毫米级精度和牛顿级力控，这是当前 foundation policy 的盲区。

**3. 真正的 Loco-Manipulation（同步运行）**

边走边操作需要两件事同时发生：
- 行走控制：实时响应地面变化，频率高
- 操作控制：精细感知和动作，频率适中

当前 foundation policy 无法同时在两个频率尺度上可靠工作。π₀ 等工作在固定位置后的操作任务上有效，但不能在行走过程中稳定操作。

**4. 数据分布外的泛化**

Foundation Policy 的泛化能力仍受训练数据分布约束：
- 在家居环境操作的模型部署到工厂环境会显著退化
- 在 6-DOF 机械臂训练的模型无法直接用于人形机器人
- 形态差异（embodiment gap）是跨平台泛化的主要障碍

### 最佳实践：Foundation Policy 与经典控制的结合

**分层架构（推荐）**：

```
[Foundation Policy / VLA]  ← 语言指令 + 视觉输入
         ↓ 高层动作目标（末端轨迹 / 接触目标）
[中层运动规划]            ← 任务空间规划
         ↓ 关节加速度 / 力分配
[底层控制器（WBC / RL）]  ← 精确执行 + 安全保证
         ↓ 关节力矩
[真机]
```

这种分层设计让每一层专注自己擅长的事：
- Foundation Policy：语义理解、任务规划、跨场景泛化
- WBC/RL：高频控制、接触力管理、物理安全保证

### 何时考虑 Foundation Policy（检查清单）

使用 Foundation Policy 的前提条件：

- [ ] 任务是操作为主（而非 locomotion）
- [ ] 有 ≥ 30 条高质量人类演示（少于此量微调效果不稳定）
- [ ] 任务语义多样（需要语言条件指定目标）
- [ ] 控制频率要求 ≤ 30Hz（否则推理延迟成为瓶颈）
- [ ] 精度要求 ≥ 5mm（精细操作不适合）
- [ ] 底盘相对固定（或行走是简单导航）

## 未来趋势（2025-2027 预判）

**近期可能突破**：
1. **Locomotion Foundation Model**：随着 OpenAI、DeepMind 等机构投入，全地形行走的 foundation policy 可能在 2025-2026 出现早期成果
2. **更快推理**：流式 diffusion / flow matching 将推理延迟压缩到 5ms 以下，使高频控制成为可能
3. **力控感知**：集成触觉传感器的 foundation policy 可能改善精细操作

**长期挑战（仍未解决）**：
1. 真正的 loco-manipulation foundation policy（统一行走+操作）
2. 实时世界模型（5ms 内的物理预测）
3. 跨形态泛化（一个模型适配多种机器人）

## 参考来源

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (2023)
- Black et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control* (2024)
- [sources/papers/rl_foundation_models.md](../../sources/papers/rl_foundation_models.md) — foundation model 全面技术摘要

## 关联页面

- [Foundation Policy](../concepts/foundation-policy.md) — 技术详解
- [VLA](../methods/vla.md) — Vision-Language-Action 模型架构
- [Loco-Manipulation](../tasks/loco-manipulation.md) — foundation policy 的前沿挑战场景
- [Manipulation](../tasks/manipulation.md) — foundation policy 当前的主战场
- [Locomotion](../tasks/locomotion.md) — 尚未被 foundation policy 攻克的任务

## 一句话记忆

> 人形机器人 foundation policy 目前是"上肢好手、下肢菜鸟"——在固定底盘的操作任务中已可用，在行走控制和真正的 loco-manipulation 中仍是前沿研究，分层架构是桥接两者的实用范式。
