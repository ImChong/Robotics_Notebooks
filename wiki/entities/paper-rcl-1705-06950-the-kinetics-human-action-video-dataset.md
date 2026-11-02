---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "1705.06950"
venue: "2017"
summary: "RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_1705_06950_the-kinetics-human-action-video-dataset.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# The Kinetics Human Action Video Dataset

**The Kinetics Human Action Video Dataset** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 127/564** 篇，分组 **Datasets**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["Stage 1 获取动作列表<br/>已有动作数据集 / Motion capture 文件标题 / Mechanical Turk 众包补充"] --> B["Stage 2 获取候选 clips<br/>Step 1 按视频标题匹配 YouTube 动作名<br/>Step 2 用 Google Image Search 训练的图像分类器逐帧定位, 取 top-k (k=2) 响应处前后各 5 秒成 10 秒 clip"]
    B --> C["Stage 3 人工标注 (Amazon Mechanical Turk)<br/>问 Can you see a human performing the action class-name?<br/>每 clip 至多 5 人标注, 至少 3 个 Yes 才入选<br/>注入 groundtruth clips 估计工人准确率"]
    C --> D["Stage 4 清理与去噪<br/>每个 YouTube 视频只保留一个 clip<br/>Inception-V1 特征 + cosine 相似度阈值 + 连通分量去重<br/>two-stream 分类器检测 noisy classes, 合并/拆分/删除类别<br/>最终人工 clip 过滤"]
    D --> E["导出 video id / clip 时间 / 标签"]
    E --> F["发布 Kinetics: 400 类<br/>train / validation / test 三个划分"]
    F --> G["Sec. 4 基线: ConvNet+LSTM / Two-Stream / 3D ConvNet<br/>在 UCF-101, HMDB-51 与 Kinetics 上比较"]
```

图依据：数据采集流程图，论文 §3 How the Dataset was Built（§3.1–§3.4 四个 Stage）、§2 Statistics（Tab. 2 划分）、§4 Benchmark Performance。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 127/564 |
| 分组 | Datasets |
| 出处 | 2017 |
| 论文 | <https://arxiv.org/abs/1705.06950> |
| 子类 / 象限 | 视频动作理解数据 · 类别与动作标注 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/1705.06950) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Datasets** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「The Kinetics Human Action Video Dataset」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Datasets**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_1705_06950_the-kinetics-human-action-video-dataset.md`](../../sources/papers/rcl_awesome_wam_1705_06950_the-kinetics-human-action-video-dataset.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/1705.06950>
- [arXiv PDF](https://arxiv.org/pdf/1705.06950)：已通读 §2 An Overview of the Kinetics Dataset、§3 How the Dataset was Built（含 §3.5–3.6 数据偏差讨论）、§4 Benchmark Performance 的基线描述；实验数字与附录以略读为准

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/1705.06950)
