---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "1807.06757"
venue: "2018"
summary: "RCL Awesome WAM 清单收录（Evaluation metrics）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_1807_06757_on-evaluation-of-embodied-navigation-age.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# On Evaluation of Embodied Navigation Agents

**On Evaluation of Embodied Navigation Agents** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 142/564** 篇，分组 **Evaluation metrics**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Evaluation metrics）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Evaluation metrics）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    G["§2 目标指定 Goal Specification<br/>PointGoal / ObjectGoal / AreaGoal"] --> GEN
    GEN["§3 Generalization and Exploration<br/>先验暴露程度须量化并报告:<br/>No prior exploration / Pre-recorded prior exploration / Time-limited exploration by the agent"] --> TB
    TB["§5 Experimental Testbeds 仿真平台 + Recommendation 4–6<br/>(连续状态空间 / 可解释且可互操作的动作与度量 / 配套真机部署软件)"] --> SC
    SC["§6 Standard Scenarios<br/>SUNCG / AI2-THOR / Matterport3D / Gibson 各选场景子集, 划分 train / validation / test<br/>随机采样 start 与 goal, 按 PointGoal / ObjectGoal 给出动作集"] --> EP
    EP["运行 N 个 test episodes<br/>agent 必须输出 DONE 动作 (Recommendation 1)"] --> SUC
    SUC["§4 判定 episode 成功 S_i<br/>用 geodesic distance (Recommendation 2)<br/>到目标距离 < 阈值 τ (默认 2× agent 身宽)"] --> SPL
    SPL["SPL = 1/N · Σ S_i · l_i / max(p_i, l_i)<br/>(Recommendation 3, 主指标)"] --> AUX
    AUX["辅助指标: 成功率 vs 路径长度 / 终点到目标距离 / τ 扫描 / 归一化逆路径长度分布 / 违规次数 / 执行时间与能耗"]
    ARCH["§7 Agent Architectures<br/>purely reactive → 短期 recurrent memory → 富内部表示<br/>Recommendation 7"] -.-> EP
```

图依据：基准构建与评测协议图，论文 §2–§6（Recommendation 1–6 与 Standard Scenarios）、§4 式 (1) SPL 定义；§7 以虚线表示被评测的 agent 类型。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 142/564 |
| 分组 | Evaluation metrics |
| 出处 | 2018 |
| 论文 | <https://arxiv.org/abs/1807.06757> |
| 子类 / 象限 | 导航评估指标 · 评估指标与协议 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Evaluation metrics）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/1807.06757) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Evaluation metrics** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「On Evaluation of Embodied Navigation Agents」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Evaluation metrics**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_1807_06757_on-evaluation-of-embodied-navigation-age.md`](../../sources/papers/rcl_awesome_wam_1807_06757_on-evaluation-of-embodied-navigation-age.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/1807.06757>
- [arXiv PDF](https://arxiv.org/pdf/1807.06757)：已通读全文（§1 Introduction、§2 Goal Specification and Sensory Input、§3 Generalization and Exploration、§4 Evaluation Measures、§5 Experimental Testbeds、§6 Standard Scenarios、§7 Agent Architectures）

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/1807.06757)
