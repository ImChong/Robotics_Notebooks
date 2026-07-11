# ARDY — NVIDIA 官方项目页

- **来源**：https://research.nvidia.com/labs/sil/projects/ardy/
- **类型**：site（项目页 / 交互演示）
- **机构**：NVIDIA Research（SIL）· ETH Zürich
- **归档日期**：2026-07-11
- **论文**：ACM TOG · SIGGRAPH 2026 · DOI [10.1145/3811284](https://doi.org/10.1145/3811284)

## 一句话说明

**ARDY** 是面向 **交互应用** 的 **自回归扩散** 人体运动生成框架：支持 **流式文本提示** 与 **灵活长时域运动学约束**（根部轨迹/路点、全身关键帧、末端关节位姿/旋转及组合），在 **实时响应** 下生成高保真 3D 人体运动。

## 为什么值得保留

- 项目页完整呈现 **在线文本→运动**、**多类运动学约束** 与 **长时域目标** 的演示分类，并给出 **方法图**（Motion Tokenizer、两阶段自回归去噪器）。
- 明确与 [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) 的 **离线高质量 vs 交互实时** 分工，以及同生态组件（SOMA、BONES-SEED、ProtoMotions、GEM、MotionBricks、GEAR-SONIC）。
- 展示 **人形机器人应用**：ARDY + SONIC → Unitree G1 交互控制。

## 核心能力（项目页归纳）

| 能力 | 示例 |
|------|------|
| 在线文本 | Limp、Pick & Put、Stealthy Walk、Victory Dance、Prompt Streaming 等 |
| 根部约束 | 稠密轨迹、稀疏路点、长时域远期目标 |
| 全身/稀疏关节 | 全身关键帧、末端位置/朝向、约束链与组合 |
| 交互 locomotion | 鼠标路点编辑 + 键盘速度指令的实时行走控制 |

## 架构要点（项目页「Method」）

- **混合表示**：显式 **全局 root** + **潜空间 body embedding**（经 Motion Tokenizer 编解码）。
- **自回归两阶段去噪**：可变历史上下文；窗口内预测干净 motion token；**root 先于 body** 的交错设计。
- **约束条件化**：mask 化运动学约束，时间/关节可稀疏，可超出当前生成窗口。

## NVIDIA 人形运动生态（项目页互链）

| 组件 | 与 ARDY 关系 |
|------|----------------|
| [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) | 离线可控扩散姊妹；ARDY 补 **交互速度** |
| [GEM / GENMO](https://github.com/NVlabs/GEM-X) | 单目视频运动重建 |
| [MotionBricks](https://research.nvidia.com/labs/sil/projects/motionbricks/) | 模块化实时潜空间生成 |
| [GEAR SONIC](https://nvlabs.github.io/GEAR-SONIC/) | ARDY 生成轨迹 → G1 物理跟踪 |
| [ProtoMotions](https://protomotions.github.io/) | 物理策略训练框架 |

## 对 wiki 的映射

1. **[ARDY（实体页）](../../wiki/entities/ardy.md)** — 架构、约束类型与机器人下游
2. **[Kimodo](../../wiki/entities/kimodo.md)** — 离线高质量可控扩散对照
3. **[SONIC](../../wiki/methods/sonic-motion-tracking.md)** — ARDY→跟踪闭环
