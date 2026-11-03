---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2507.05198"
venue: "arXiv 2025"
code: https://github.com/GigaAI-research/EmbodieDreamer
summary: "Advancing real2sim2real transfer via embodied world modeling."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2507_05198_embodiedreamer-advancing-real2sim2real-t.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# EmbodieDreamer

**EmbodieDreamer: Advancing Real2Sim2Real Transfer for Policy Training via Embodied World Modeling** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 291/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Advancing real2sim2real transfer via embodied world modeling.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Advancing real2sim2real transfer via embodied world modeling.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph PA["PhysAligner：物理参数辨识（Real2Sim）"]
        R["真实数据：动作 a(t-1) 与状态 S(t-1)"] --> SIM["随机采样 (摩擦 f, 刚度 p, 阻尼 d)<br/>在仿真器中执行动作，得到仿真轨迹数据集"]
        SIM --> SUR["训练代理模型 M(f,p,d)<br/>MSE 拟合仿真器的状态转移，得到可微近似"]
        SUR --> GR["固定网络权重，对 (f,p,d) 做梯度下降<br/>使预测轨迹贴合真实轨迹"]
        GR --> OPT["优化后的 PD 控制器参数 (f*, p*, d*)"]
    end
    OPT --> SIMR["对齐物理的仿真器"]
    subgraph VA["VisAligner：外观迁移（Sim2Real）"]
        SR["仿真中回放机器人（URDF）得到机器人观测"] --> CD["+ GroundSAM 前景物体分割 mask<br/>逐像素相加得到仿真条件帧"]
        BG["参考图像（背景）作为首帧"] --> CC["条件序列 = 参考图 + 后续条件帧"]
        CD --> CC
        CC --> VAE["VAE 编码，与噪声沿通道拼接"]
        VAE --> DN["条件视频扩散去噪<br/>（空间 + 时间注意力）→ VAE 解码"]
        DN --> PV["逼真视频观测"]
    end
    SIMR --> SR
    PV --> POL["机器人策略（如 ACT）"]
    POL -->|"动作"| SIMR
    POL --> RL["RL：批量生成轨迹，按终态末端到目标距离<br/>取 top-M / bottom-M 做偏好学习，迭代微调"]
    PV --> IL["IL：改变前景外观 / 背景，生成多样未见观测作为增广数据"]
```

图依据：论文 §3.1–3.3 / Fig.1–4、§4.3 RL 与 §4.4 IL 设定

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 291/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2507.05198> |
| 代码/项目 | <https://github.com/GigaAI-research/EmbodieDreamer> |

## 核心机制（归纳）

### 策展导读要点

Advancing real2sim2real transfer via embodied world modeling.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2507.05198) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EmbodieDreamer」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2507_05198_embodiedreamer-advancing-real2sim2real-t.md`](../../sources/papers/sun_awesome_wm_2507_05198_embodiedreamer-advancing-real2sim2real-t.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2507.05198>
- [arXiv PDF](https://arxiv.org/pdf/2507.05198)：已通读方法部分（实验与附录以略读为准）：§1 引言（贡献部分）、§3 方法（3.1 框架概览、3.2 PhysAligner、3.3 VisAligner）、§4.3 RL / §4.4 IL 实验设定

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2507.05198)
