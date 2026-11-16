---
type: entity
project_id: maxinsights-physical-ai-data-services
tags: [physical-ai, egocentric-data, data-collection, robot-learning, dataset]
status: complete
updated: 2026-11-12
project: https://www.maxinsights.ai/
related:
  - ../overview/ego-category-01-data-collection.md
  - ../concepts/embodied-scaling-laws.md
  - ./dyna-2.md
original:
  - ../../sources/sites/maxinsights-official.md
  - ../../sources/blogs/maxinsights-beyond-1m-hours-physical-ai-data.md
sources:
  - ../../sources/sites/maxinsights-official.md
  - ../../sources/blogs/maxinsights-beyond-1m-hours-physical-ai-data.md
  - ../../sources/blogs/dyna-2-million-hour-scaling-law.md
summary: "Maxinsights 是一家自述位于旧金山湾区的 Physical AI 第一人称多模态数据服务商；官网披露累计经验与月产能指标，但未公开代码、可下载数据集或客户级数据来源。"
---

# Maxinsights：Physical AI 第一人称数据服务

## 一句话定义

**Maxinsights** 是一家自述位于旧金山湾区的 Physical AI 数据服务商，官网介绍其采集和处理第一人称视频及多模态信号，并将其整理为机器人训练与评估可用的数据。公开页面披露的是公司自报规模，未披露可下载语料或客户级来源。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric | 以佩戴者视角记录的第一人称数据 |
| MoCap | Motion Capture | 对手部或身体运动进行捕捉 |
| VLA | Vision-Language-Action | 视觉、语言与动作联合建模 |
| WAM | World-Action Model | 联合预测世界变化与动作的模型 |
| QA | Quality Assurance | 数据采集与加工中的质量检查 |

## 为什么重要？

机器人遥操作数据往往需要机器人、操作员与场景配合。第一人称人类操作视频可以扩展真实场景的覆盖面；若进一步保留手部姿态、空间、物体运动和接触等信息，模型就能获得比原始 RGB 更直接的物理监督。

Maxinsights 的公开定位是工业化采集与加工服务。它体现了具身数据产业从“记录更多小时”走向“让每小时包含更多可学习信息”的方向。该定位来自公司官网和公司研究文章，应与独立验证过的研究结论区分。

## 核心机制：采集规模 × 可学习信息密度

Maxinsights 的官方研究文章把具身数据供给分为两条互补维度：**Experience Scale（经验规模）**指采集小时数与环境、任务覆盖；**Experience Density（经验密度）**指每段记录保留的手/身体、物体与接触等可学习物理信息。作者使用“有效经验 ≈ 小时数 × 每小时信息量”作为**核算框架而非实测拟合定律**。同一小时视频未必有同样的训练价值。

以下是根据官网业务步骤与研究文章整理的**概念流程图**，并非 Maxinsights 公开的内部系统架构或已验证的自动化程度：

```mermaid
flowchart LR
    A[任务设计与覆盖规划] --> B[真实场景采集]
    B --> C[视频及可选多模态信号]
    C --> D[动作理解与手部/身体/空间状态恢复]
    D --> E[同步、自动质量检查与专家复核]
    E --> F[结构化训练数据]
    E -->|疑难样本| G[人工复核与修正]
    G --> E
```

官方文章主张用端到端质检通过率、每小时新增物理信息和物理状态覆盖衡量数据价值；后两项仍是作者提出的待标准化指标。公司官网还把训练、真实环境评估和部署反馈描述为发展方向，不能据此认为其完整部署闭环已获第三方验证。

## 官网公开的业务与规模

官网将其服务概括为大规模多模态采集、手部追踪、上/全身动作捕捉、触觉与力信号、仿真和模型评估。页面列出的结构化信号包括视频理解、深度、物体跟踪、手部姿态、上身姿态、触觉/力、定位、分割和点云。

| 官网指标 | 公司披露 | 读法 |
|----------|----------|------|
| 累计经验 | 2,000,000 小时 | 官网标注为截至 2026 年 9 月累计经验；不是某一客户的交付时长 |
| 月产能 | 最高 450,000 小时 | 产能上限，不等于每月实际采集或通过质检的可训练数据 |
| 环境覆盖 | 50,000+ 真实世界环境 | 公司自报的采集环境指标，非独立测量的状态空间覆盖率 |
| 采集人员与网络 | 3,500+ 现场操作人员、8 个国家 | 官网当前自报的采集网络规模 |
| 结构化输出 | 9 类信号 | 公司列出的视频理解、深度、物体/手部/身体状态、力、定位、分割与点云等 |

这些数字均为企业官网自述，公开页面未说明独立审计方法、端到端合格率或各项指标的详细口径。评估服务能力时，应把原始采集时长、质检后可用时长、标注覆盖与客户实际交付分开核对。

## 数据集速查维度

| 维度 | 公开信息 | 备注 |
|------|----------|------|
| 规模 | 累计 2,000,000 小时、月产能最高 450,000 小时（官网自报） | 非可下载语料规模，未说明质检后可用时长；与公开采集人员规模不是同一口径 |
| 模态 | 第一人称视频，外加深度、物体跟踪、手部/上身姿态、触觉/力、定位、分割、点云 | 来自官网结构化信号列表 |
| 许可证 | 无公开数据集与许可证；以商业数据服务形式交付 | 客户级授权与使用条款未公开 |
| 重定向就绪度 | 输出为人类第一人称视频与手/身体姿态，不是机器人本体数据 | 用于机器人训练需按目标形态做手部/身体重定向，官网未说明是否提供该环节 |

## 与百万小时人类视频预训练的关系

Dyna Robotics 的 Dyna-2 官方报告称，模型使用超过 100 万小时的人类第一人称视频预训练，并报告了随数据规模上升的留出数据表现。报告提到语料来自未具名的数据合作方和 Dyna 内部采集；没有列出 Maxinsights。

Maxinsights 的官方文章讨论了 Dyna-2，并提出“经验规模”和“经验密度”两个观察维度，但没有声称为 Dyna-2 供应数据。因此，**Dyna-2 的百万小时研究可作为该产业方向的背景，不能据此推定 Maxinsights 是其供应商**。

## 工程实践与开放状态

- 官网描述的生产流程包括任务设计、采集、动作理解、姿态/空间状态恢复、自动检查和人工复核；触觉、力等高密度信号依赖额外传感设备，不能假定每段第一人称视频均具备。
- 官网自述与 **Genesis AI / GENE-26.5** 存在第一人称数据合作，但此表述仅证明公司公开声明了该关系，尚不能据此量化实际交付、授权条件或模型贡献。
- 截至 2026-11-12，检查到的官网和研究文章没有链接可公开下载的训练数据、算法权重或完整运行代码，**源码运行时序图不适用**；无法据这些材料复现其商业数据管线。

## 局限与风险

- **量化口径：** 官网自报的累计录制小时、最高月产能和环境数量不等于验收通过数据、已交付数据或有效状态覆盖；没有独立审计披露。
- **交付细节：** 客户级许可证、标注误差、传感器配置、质检通过率和人到机器人跨本体重定向的具体实现未公开。
- **主张证据：** 本次核对未从上述官方页面确认“Maxego”算法、5000 名采集员网络、2024 年成立时间，亦未确认 Google DeepMind、Figure 的客户关系。官网当前自报 3,500+ 现场操作人员。
- **供应关系：** Dyna-2 技术报告确认语料包括未具名合作方和内部采集，但没有单独点名 Maxinsights；不得仅根据其 Scaling Laws 讨论推定它是 Dyna-2 供应商。

## 结论

- Maxinsights 官网证实其业务方向是第一人称多模态 Physical AI 数据采集与加工；其公开产能和覆盖规模属于公司自报指标。
- 需要把“累计经验”“月度产能”“质检后可用时长”和“客户交付量”分开比较。
- Dyna-2 官方报告证明百万小时人类视频已被用于机器人模型预训练，但没有识别其具体数据供应商。
- 官网现公开自报 3,500+ 现场操作人员及 Genesis AI 合作；其交付量与模型收益尚缺独立验证，不能与 Dyna-2 合作方信息混同。
- 当前资料不足以验证 Maxego、成立年份及其他所列客户关系。

## 关联页面

- [Ego 分类 01：数据采集](../overview/ego-category-01-data-collection.md)
- [具身 Scaling Laws](../concepts/embodied-scaling-laws.md)
- [Dyna-2](./dyna-2.md)

## 参考来源

- [Maxinsights 官方网站归档](../../sources/sites/maxinsights-official.md)
- [Maxinsights 官方研究文章摘录](../../sources/blogs/maxinsights-beyond-1m-hours-physical-ai-data.md)
- [Dyna-2 官方技术报告摘录](../../sources/blogs/dyna-2-million-hour-scaling-law.md)

## 推荐继续阅读

- [Maxinsights 官网](https://www.maxinsights.ai/)
- [Maxinsights：Beyond 1M Hours](https://www.maxinsights.ai/beyond-1m-hours.html)
- [Dyna Robotics：Dyna-2 技术报告](https://www.dyna.co/dyna-2)
