---
type: entity
tags:
  - paper
  - 3d
  - segmentation
  - manipulation
  - vla
status: complete
updated: 2026-09-15
arxiv: "2609.12898"
code: https://github.com/xinqiangyu/UniPart
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ./paper-datafarm.md
  - ../overview/vla-tamp-planning-11-papers-technology-map.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/papers/unipart_arxiv_2609_12898.md
  - ../../sources/repos/unipart.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md
summary: "CLIP 文本条件化前馈 3D Transformer + LangPart-1M（8M 文本—部件对），面向开放词汇 3D 部件分割与语言条件抓取。"
---

# UniPart（arXiv:2609.12898）

**UniPart**（[UniPart: Towards Zero-shot Language-Grounded 3D Part Segmentation for Embodied Interaction](https://arxiv.org/abs/2609.12898)）来自 [具身智能小站 11 篇 VLA/TAMP 盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)。精细操作需要定位可操作部件而非整物体；UniPart 用语言指向抽屉把手、杯盖边缘等部件。

## 一句话定义

**CLIP 文本条件化前馈 3D Transformer + LangPart-1M（8M 文本—部件对），面向开放词汇 3D 部件分割与语言条件抓取。。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| TAMP | Task and Motion Planning | 任务与运动规划 |
| OOD | Out-of-Distribution | 分布外泛化评测 |

## 为什么重要

- 精细操作需要定位可操作部件而非整物体；UniPart 用语言指向抽屉把手、杯盖边缘等部件。
- 开源状态：**已开源**（步骤 2.5 核查，2026-09-14）。
- 与 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.12898](https://arxiv.org/abs/2609.12898) |
| **项目页** | https://xinqiangyu.github.io/UniPart/ |
| **代码** | https://github.com/xinqiangyu/UniPart |
| **开源** | **已开源** |
| **文内指标** | 构建 LangPart-1M 数据集（8M 文本—部件对）；面向零样本语言接地 3D 部件分割。 |


## 源码运行时序图

```mermaid
sequenceDiagram
  participant U as 用户/脚本
  participant R as 官方仓库入口
  participant M as 模型/训练或推理
  participant E as 仿真或真机环境
  U->>R: clone + 安装依赖
  U->>M: 加载配置/权重
  U->>E: rollout / 评测
  M-->>E: 动作或轨迹
  E-->>U: 成功率/指标日志
```


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | 构建 LangPart-1M 数据集（8M 文本—部件对）；面向零样本语言接地 3D 部件分割。 |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

- **物体级 3D 检测/分割** — 输出「这是一个杯子」，但操作需要的是「杯盖边缘在哪」；UniPart 把粒度下推到 **部件级**，这正是 [图像分割分类学](../concepts/image-segmentation-taxonomy.md) 里语义/实例/部件三层粒度之分在 3D 上的延伸。
- **闭集部件分割** — 类别表写死，换个没见过的物体就失效；UniPart 用 **CLIP 文本条件化** 做开放词汇接地，代价是精度受文本–几何对齐质量牵制，且依赖 LangPart-1M 这类大规模文本—部件配对数据（8M 对）。
- **优化式/迭代式 3D 接地** — 每次查询都要跑优化，机载帧率难保证；UniPart 是 **前馈 3D Transformer**，一次前向出结果，把成本压在训练侧。
- **[抓取位姿估计](../methods/grasp-pose-estimation.md) 与 [AnyGrasp vs GraspNet](../comparisons/anygrasp-vs-graspnet.md)** — 那条线回答「从哪下手抓得稳」（几何可抓性），UniPart 回答「该抓哪个部件」（语言意图）；两者串联才构成语言条件抓取的完整链路。
- **[ArtManip](./paper-artmanip.md)（同批）** — 互补而非竞争：UniPart 在 **感知侧** 定位可操作部件，ArtManip 在 **控制侧** 处理接触内操作；一头一尾。
- **[机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md)** — 按该指南分层，UniPart 落在 ③「2D→3D 提升与语义建图」层：它直接在 3D 上出部件级语义，绕开了 [2D→3D 语义提升 Gap](../concepts/2d-to-3d-semantic-lifting-gap.md) 的一部分损失，但对点云质量的依赖相应更重。

- **读法：** 以上为知识库内 **路线级** 对照；与原文 baseline 的逐项定量比较以 **原文 PDF** 为准（[参考来源](#参考来源)）。

## 结论

**UniPart 适合作为本期「已开源」边界下的快速索引页，部署前请核对项目页/仓库可运行性。**

1. 核心贡献：精细操作需要定位可操作部件而非整物体；UniPart 用语言指向抽屉把手、杯盖边缘等部件。
2. 开源结论：**已开源** — 以项目页实际链接为准（入库日 2026-09-14）。
3. 横向对照见 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)
- [机器人视觉感知栈选型闭环知识链](../queries/robot-perception-stack-selection-loop.md) — 本页属③层「3D 语义几何」一支，出的是部件级开放词汇接地

## 参考来源

- [unipart_arxiv_2609_12898.md](../../sources/papers/unipart_arxiv_2609_12898.md)
- [wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)
- [arXiv:2609.12898](https://arxiv.org/abs/2609.12898)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.12898)
- [项目页](https://xinqiangyu.github.io/UniPart/)
- [https://github.com/xinqiangyu/UniPart](https://github.com/xinqiangyu/UniPart)
