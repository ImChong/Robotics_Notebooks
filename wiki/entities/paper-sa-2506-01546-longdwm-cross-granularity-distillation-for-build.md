---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.01546"
venue: "arXiv 2025"
summary: "Cross-granularity distillation for building long-term driving world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_01546_longdwm-cross-granularity-distillation-f.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# LongDWM

**LongDWM: Cross-Granularity Distillation for Building a Long-Term Driving World Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 456/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Cross-granularity distillation for building long-term driving world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Cross-granularity distillation for building long-term driving world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph TRAIN["训练"]
        A["长时视频低 fps（如 1 fps）稀疏采样 K 帧，3D-VAE 逐帧独立编码"] --> B["Coarse DiT（CogVideoX 架构）：MLLM 详细标注 + T5 文本嵌入，图生视频预测；损失 = 扩散损失 + 高通滤波结构保持损失"]
        C["短时视频连续采 K 帧，3D-VAE 编码"] --> D["Fine DiT：同时学习视频预测（因果）与视频插值（双向）"]
        D --> E["冻结 Fine DiT 作教师；粗帧夹在细帧序列中，各段共享时间步与噪声做一步去噪"]
        B --> F["学生 Coarse DiT 初始化自已训好的 CDiT"]
        E --> G["Flow Distillation：细粒度 flow 蒸馏粗粒度 flow，只需数百步"]
        F --> G
    end
    subgraph INFER["长视频生成"]
        H["首帧输入 Coarse DiT，预测大运动关键帧序列"] --> I["相邻关键帧之间由 Fine DiT 并行插值出连续帧"]
        I --> J["长时、时间连贯的驾驶视频"]
    end
    G -.-> H
    D -.-> I
    K["轨迹控制（可选）：轨迹 → 相机位姿 → 无深度图像 warp → 补丁嵌入注入 Fine DiT"] -.-> I
```

图依据：论文 §3.1 Coarse DiT / §3.2 Fine DiT / §3.3 Flow Distillation / §3.4 Warp-guided 控制（Fig. 2–3）；推理阶段先粗后细并行插值见附录实验设置

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 456/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.01546> |
| 项目页 | <https://wang-xiaodong1899.github.io/longdwm/> |

## 核心机制（归纳）

### 策展导读要点

Cross-granularity distillation for building long-term driving world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.01546) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「LongDWM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **82 Autonomous Driving**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2506_01546_longdwm-cross-granularity-distillation-f.md`](../../sources/papers/sun_awesome_wm_2506_01546_longdwm-cross-granularity-distillation-f.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.01546>
- [arXiv PDF](https://arxiv.org/pdf/2506.01546)：已通读方法部分（实验与附录以略读为准）：摘要、§3.1–3.4 方法、附录中长视频预测评测设置；实验数字略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.01546)
