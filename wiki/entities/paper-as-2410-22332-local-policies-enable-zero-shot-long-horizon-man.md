---
type: entity
tags: [paper, curated-index, awesome-sim2real, longchao-sim2real]
status: complete
updated: 2026-10-06
arxiv: "2410.22332"
summary: "LongchaoDa AwesomeSim2Real 收录；分组 Action / Foundation Models。 本页是清单索引，细节以原文为准。"
related:
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_2410_22332_local-policies-enable-zero-shot-long-hor.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# Local Policies Enable Zero-shot Long-horizon Manipulation

**Local Policies Enable Zero-shot Long-horizon Manipulation** 收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) **第 026/139** 篇，分组 **Action / Foundation Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

LongchaoDa AwesomeSim2Real 收录；分组 Action / Foundation Models。 本页是清单索引，细节以原文为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型/基础模型增强迁移 |

## 为什么重要

- LongchaoDa AwesomeSim2Real 收录；分组 Action / Foundation Models。 本页是清单索引，细节以原文为准。
- 想横向对照同一分组的其他工作，可以从 [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md) 逐条展开。
- 顺着列表实体 [AwesomeSim2Real](../entities/awesome-sim2real.md) 与站内 Sim2Real 方法/任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph P1["阶段 1：对象级 RL 专家"]
        A1["仿真中处理资产<br/>UnidexGrasp 3.5K 物体 / PartNet 2.6K 把手"] --> A2["PPO 训练 3.5K+ 单物体<br/>state-based RL 专家<br/>(pick/place/grasp handle/open/close)"]
    end
    subgraph P2["阶段 2：蒸馏通才策略"]
        B1["Multi-task DAgger<br/>(带 replay buffer 的变体)"] --> B2["通才视觉运动局部策略 π_loc<br/>局部深度图 + 分割掩码 + 相对本体感知"]
    end
    subgraph D["部署：零样本长时程操作"]
        C0["用户语言指令 g + 场景图像 O"] --> C1["VLM (GPT-4o)<br/>分解为 (object, skill) 子目标序列"]
        C1 --> C2["Grounded SAM 分割点云<br/>估计目标位姿 X_targ"]
        C2 --> C3["IK 得目标关节 q_f<br/>Neural MP 运动规划 (π_reach)"]
        C3 --> C4["执行局部策略 π_loc<br/>(Task Space Impedance 控制器)"]
        C4 -->|"任务未完成：下一子目标"| C2
    end
    A2 -->|"特权状态专家示范"| B1
    B2 --> C4
```

图依据：论文 §III（Fig. 3 ManipGen Method Overview；§III-A Framework、B 训练局部策略、C 蒸馏、D 零样本部署）。循环箭头依据 §III-D「alternate between motion planning and deploying local policies until the task is complete」。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 026/139 |
| 分组 | Action / Foundation Models |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2410.22332> |

## 核心机制（归纳）

### 策展导读要点

LongchaoDa AwesomeSim2Real 收录；分组 Action / Foundation Models。 本页是清单索引，细节以原文为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2410.22332) 为准。
- 横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 的 **Action / Foundation Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Local Policies Enable Zero-shot Long-horizon Manipulation」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Action / Foundation Models**；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的分组标签当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[AwesomeSim2Real](../entities/awesome-sim2real.md)
- 技术地图：[AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- 方法/任务：[sim2real.md](../concepts/sim2real.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/lc_awesome_sim2real_2410_22332_local-policies-enable-zero-shot-long-hor.md`](../../sources/papers/lc_awesome_sim2real_2410_22332_local-policies-enable-zero-shot-long-hor.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<https://arxiv.org/abs/2410.22332>
- [arXiv PDF](https://arxiv.org/pdf/2410.22332)：已通读方法部分（实验与附录以略读为准）：§III-A～D。

## 推荐继续阅读

- [AwesomeSim2Real 仓库](https://github.com/LongchaoDa/AwesomeSim2Real)
- [原文](https://arxiv.org/abs/2410.22332)
