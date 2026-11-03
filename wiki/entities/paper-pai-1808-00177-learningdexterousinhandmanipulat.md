---
type: entity
tags: [curated-index, physical-ai, awesome-physical-ai, paper, paper]
status: complete
updated: 2026-10-06
arxiv: "1808.00177"
summary: "Sim-to-real dexterous manipulation via automatic domain randomization."
related:
  - ../entities/awesome-physical-ai-natnew.md
  - ../overview/awesome-physical-ai-technology-map.md
  - ../concepts/sim2real.md
  - ../tasks/locomotion.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
sources:
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/papers/pai_awesome_1808_00177_learning-dexterous-in-hand-manipulat.md
  - ../../sources/repos/awesome-physical-ai-union-catalog.md
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/repos/awesome-physical-ai-aichr.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
---

# Learning Dexterous In-Hand Manipulation (OpenAI)

**Learning Dexterous In-Hand Manipulation (OpenAI)** 收录于 awesome-physical-ai（natnew）**第 334/384** 条，分组 **Sim-to-Real**。本页是 **清单索引**：给出它在清单中的位置与官方入口，细节以官方文档 / 原文为准。

## 一句话定义

Sim-to-real dexterous manipulation via automatic domain randomization.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PAI | Physical AI | 具身/物理智能策展主题 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RFM | Robotics Foundation Model | 机器人基础模型 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Sim-to-real dexterous manipulation via automatic domain randomization.
- 想横向对照同一分组的其他工作，可以从 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md) 逐条展开。
- 两份清单合并去重：同一条目只有这一页，出处见下方「参考来源」。

## 流程总览

```mermaid
flowchart TD
    R["随机化仿真分布：物理参数 / 观测噪声 / 未建模效应 / 视觉外观"] --> W["Rapid：384 台 worker 并行 rollout"]
    W -->|"经验写入 Redis"| OPT["优化器（8 GPU）：分布式 PPO"]
    OPT -->|"新策略参数"| W
    OPT --> POL["策略：带 LSTM 的循环网络，离散动作 11 档"]
    R --> VIS["视觉位姿估计器：3 路 RGB 相机 → CNN → 物体位置 + 朝向"]
    SIMDATA["用训练好的策略在仿真中采集约 100 万状态，随机化外观渲染"] --> VIS
    R --> SIMDATA
    MC["指尖位置：PhaseSpace 动捕"] --> POL
    VIS -->|"物体位姿"| POL
    POL -->|"相对关节角目标"| HAND["Shadow Dexterous Hand 实机"]
```

图依据：论文 §3（Transferable Simulations）、§4.1-4.3（策略、动作与奖励、Rapid）、§5（视觉状态估计），Fig.5 / Fig.6。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 334/384 |
| 分组 | Sim-to-Real |
| 来源清单 | natnew |
| 主链接 | <https://arxiv.org/abs/1808.00177> |
| arXiv | 1808.00177 |

## 核心机制（归纳）

### 策展导读要点

Sim-to-real dexterous manipulation via automatic domain randomization.

本页不复述论文公式与完整实验表；工程落地请回到原文 / 项目页，并对照站内 VLA、Sim2Real 或任务页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/1808.00177) 为准。
- 横向对照请回到 [技术地图](../overview/awesome-physical-ai-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比，只给出清单坐标。
- 与站内 **深度论文实体** 的分界：深度页承载机构、实验表与源码运行时序；本页只承载清单导读锚点。同一 arXiv 若已存在深度页，应以深度页为准。
- 两份同名清单可能对同一工作给出不同链接（项目页 / arXiv / GitHub）；以本页主链接与技术地图为准。

## 结论

**这一页能给你的是「Learning Dexterous In-Hand Manipulation (OpenAI)」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

1. **策展坐标** — 分组 **Sim-to-Real**，来源 natnew。
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

- [`sources/papers/pai_awesome_1808_00177_learning-dexterous-in-hand-manipulat.md`](../../sources/papers/pai_awesome_1808_00177_learning-dexterous-in-hand-manipulat.md) — 本条目策展摘录
- [`sources/repos/awesome-physical-ai-union-catalog.md`](../../sources/repos/awesome-physical-ai-union-catalog.md) — 双清单并集目录
- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)
- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)
- 主链接：<https://arxiv.org/abs/1808.00177>
- [arXiv PDF](https://arxiv.org/pdf/1808.00177)：已通读方法部分（实验与附录以略读为准）：§3 可迁移仿真、§4 从状态学习控制策略（含 Rapid 分布式训练）、§5 视觉状态估计、§6 开头

## 推荐继续阅读

- [natnew/awesome-physical-ai](https://github.com/natnew/awesome-physical-ai)
- [aichr/awesome-physical-ai](https://github.com/aichr/awesome-physical-ai)
- [原文 / 官方入口](https://arxiv.org/abs/1808.00177)
