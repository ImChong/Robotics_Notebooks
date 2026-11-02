---
type: entity
tags: [curated-index, physical-ai, awesome-physical-ai, paper, paper]
status: complete
updated: 2026-09-20
arxiv: "2011.01387"
summary: "Siekmann et al. Periodic reward composition that transfers a single bipedal policy across walking, hopping, and skipping gaits (ICRA 2021)."
related:
  - ../entities/awesome-physical-ai-natnew.md
  - ../overview/awesome-physical-ai-technology-map.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/pai_awesome_2011_01387_periodic-reward-composition-for-bipe.md
  - ../../sources/repos/awesome-physical-ai-union-catalog.md
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/repos/awesome-physical-ai-aichr.md
---

# Periodic Reward Composition for Bipedal Gaits

**Periodic Reward Composition for Bipedal Gaits** 收录于 awesome-physical-ai（natnew）**第 201/384** 条，分组 **Locomotion / Key Papers**。本页是 **清单索引**：给出它在清单中的位置与官方入口，细节以官方文档 / 原文为准。

## 一句话定义

Siekmann et al. Periodic reward composition that transfers a single bipedal policy across walking, hopping, and skipping gaits (ICRA 2021).

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PAI | Physical AI | 具身/物理智能策展主题 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RFM | Robotics Foundation Model | 机器人基础模型 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Siekmann et al. Periodic reward composition that transfers a single bipedal policy across walking, hopping, and skipping gaits (ICRA 2021).
- 想横向对照同一分组的其他工作，可以从 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md) 逐条展开。
- 两份清单合并去重：同一条目只有这一页，出处见下方「参考来源」。

## 流程总览

```mermaid
flowchart TD
    SPEC["步态规格<br/>每个脚的相位序列（swing / stance 等）<br/>相位起止时间 (a_i, b_i, κ)、左右脚周期偏移 θ_left, θ_right"] --> RC
    RC["周期奖励组合<br/>R(s, φ) = β + Σ c_i · I_i(φ) · q_i(s)<br/>相位系数 c_i × 相位指示 I_i（Von Mises 分布）× 相位量测 q_i（脚力 / 脚速范数）"] --> EXP["取期望奖励 E[R(s, φ)]<br/>叠加指令奖励（速度/朝向）与平滑项等辅助代价"]
    SPEC --> ST["状态编码：周期时钟 φ（含 θ 偏移）<br/>+ 各相位占比 r_j<br/>+ 骨盆姿态/角速度、关节位置/速度、速度指令"]
    ST --> POL["LSTM 策略（2 层 × 128）<br/>输出 30 维：10 个关节目标位置 + 10 组 PD 增益<br/>策略 40 Hz"]
    POL --> PD["PD 控制器（2000 Hz）"]
    PD --> SIM["cassie-mujoco-sim 仿真 + 动力学随机化<br/>（执行频率 / 质量 / 阻尼 / 摩擦 / 坡度 / 编码器噪声）"]
    SIM --> ST
    EXP --> PPO["PPO（循环版，轨迹批采样）+ mirror loss"]
    SIM --> PPO
    PPO --> POL
    PPO --> HW["sim-to-real：部署到 Cassie 真机"]
    EXP -.-> TP["多步态训练：变化相位占比，另加 transition penalties"]
```

图依据：论文 §III（RL 框架、周期奖励组合、步态描述）与 §IV Method（网络与动作空间、状态空间、奖励、动力学随机化、PPO 训练）及 Fig.1 / Fig.3 / Fig.4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 201/384 |
| 分组 | Locomotion / Key Papers |
| 来源清单 | natnew |
| 主链接 | <https://arxiv.org/abs/2011.01387> |
| arXiv | 2011.01387 |

## 核心机制（归纳）

### 策展导读要点

Siekmann et al. Periodic reward composition that transfers a single bipedal policy across walking, hopping, and skipping gaits (ICRA 2021).

本页不复述论文公式与完整实验表；工程落地请回到原文 / 项目页，并对照站内 VLA、Sim2Real 或任务页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2011.01387) 为准。
- 横向对照请回到 [技术地图](../overview/awesome-physical-ai-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比，只给出清单坐标。
- 与站内 **深度论文实体** 的分界：深度页承载机构、实验表与源码运行时序；本页只承载清单导读锚点。同一 arXiv 若已存在深度页，应以深度页为准。
- 两份同名清单可能对同一工作给出不同链接（项目页 / arXiv / GitHub）；以本页主链接与技术地图为准。

## 结论

**这一页能给你的是「Periodic Reward Composition for Bipedal Gaits」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

1. **策展坐标** — 分组 **Locomotion / Key Papers**，来源 natnew。
2. **适用边界** — 本页不能替代原文 PDF / 官方文档；开源状态以项目页实际链接为准。
3. **以深读页为准** — 若站内已有这篇的深读页，以那一页为准；本页只是清单入口。
4. **怎么深读** — 要深读这篇，建议直接从原文入手。

## 源码运行时序图

**不适用**（本页未逐仓核对 README 里的训练 / 推理入口；清单标注的仓库链接可能滞后，复现前请先打开项目页核对）。

## 局限与风险

- 不要把 Awesome 摘要当成完整方法证明或合规结论。
- 同名 GitHub 仓（natnew vs aichr）条目链接可能不同；以本页主链接与技术地图为准。
- 清单中的实验室 / 硬件 / 人物条目偶发链到错误 org，复现或引用前先打开官方页核对。

## 关联页面

- [awesome-physical-ai（natnew）](../entities/awesome-physical-ai-natnew.md)
- [awesome-physical-ai（aichr）](../entities/awesome-physical-ai-aichr.md)
- [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)

## 参考来源

- [`sources/papers/pai_awesome_2011_01387_periodic-reward-composition-for-bipe.md`](../../sources/papers/pai_awesome_2011_01387_periodic-reward-composition-for-bipe.md) — 本条目策展摘录
- [`sources/repos/awesome-physical-ai-union-catalog.md`](../../sources/repos/awesome-physical-ai-union-catalog.md) — 双清单并集目录
- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)
- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)
- 主链接：<https://arxiv.org/abs/2011.01387>
- [arXiv PDF](https://arxiv.org/pdf/2011.01387)：已通读方法部分（实验与附录以略读为准）：§III 周期奖励组合与步态描述、§IV 方法（网络/状态/奖励/随机化/PPO）、§V 实验结果开头（单步态与多步态）

## 推荐继续阅读

- [natnew/awesome-physical-ai](https://github.com/natnew/awesome-physical-ai)
- [aichr/awesome-physical-ai](https://github.com/aichr/awesome-physical-ai)
- [原文 / 官方入口](https://arxiv.org/abs/2011.01387)
