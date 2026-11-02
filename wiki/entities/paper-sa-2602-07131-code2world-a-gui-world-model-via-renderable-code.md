---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2602.09856"
venue: "arXiv 2026"
summary: "Models GUI dynamics by generating renderable code rather than pixels."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2602_07131_code2world-a-gui-world-model-via-rendera.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Code2World

**Code2World: A GUI World Model via Renderable Code Generation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 499/571** 篇，分组 **83 GUI / Computer-Use Agents**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Models GUI dynamics by generating renderable code rather than pixels.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Models GUI dynamics by generating renderable code rather than pixels.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph D["数据合成：AndroidCode（>80K 样本）"]
        S1["AndroidControl 截图"] --> S2["GPT-5 约束式初始合成 HTML"]
        S2 --> S3["渲染回图像"]
        S3 --> S4{"SigLIP 分数 与真值对齐"}
        S4 -->|"高于阈值"| S5["保留样本"]
        S4 -->|"低于阈值"| S6["GPT-5 视觉反馈修订<br/>输入真值图+渲染图+代码"]
        S6 --> S3
    end
    D --> T1["阶段1：SFT 冷启动<br/>Qwen3-VL-8B，代码 token 负对数似然"]
    T1 --> T2["阶段2：Render-Aware RL（GRPO）<br/>奖励 R_sem + R_act，均为 VLM-as-a-Judge"]
    T2 --> M["Code2World：Mθ(I_t, a_t, G) → HTML 代码"]
    M --> R["浏览器渲染 R → 预测下一屏 Î_t+1"]
    subgraph A["应用：Propose, Simulate, Select"]
        P1["GUI Agent 提出 K 个候选动作"] --> P2["Code2World 逐个模拟下一屏"]
        P2 --> P3["VLM 验证器 S 打分，选最优动作"]
    end
    R -.-> P2
```

图依据：论文 §3.1 / §3.2 / §4.2 与 Fig.2、Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 499/571 |
| 分组 | 83 GUI / Computer-Use Agents |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.09856> |

## 核心机制（归纳）

### 策展导读要点

Models GUI dynamics by generating renderable code rather than pixels.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.09856) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **83 GUI / Computer-Use Agents** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Code2World」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **83 GUI / Computer-Use Agents**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2602_07131_code2world-a-gui-world-model-via-rendera.md`](../../sources/papers/sun_awesome_wm_2602_07131_code2world-a-gui-world-model-via-rendera.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2602.09856>
- 编号核对（2026-11-02）：上游清单登记的 2602.07131 经 arXiv 官方记录核对为《Behavior Score Prediction in Resting-State Functional MRI by Deep State Space Modeling》，与本页论文不符；已更正为 2602.09856《Code2World: A GUI World Model via Renderable Code Generation》。
- [arXiv PDF](https://arxiv.org/pdf/2602.09856)：已通读方法部分（实验与附录以略读为准）：§3.1 训练数据合成、§3.2 模型优化、§4.2 GUI Agent 应用（Propose, Simulate, Select）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2602.09856)
