---
type: entity
project_id: maxinsights-physical-ai-data-services
tags: [physical-ai, egocentric-data, data-collection, robot-learning, dataset]
status: complete
updated: 2026-10-22
project: https://www.maxinsights.ai/
related:
  - ../overview/ego-category-01-data-collection.md
  - ../concepts/embodied-scaling-laws.md
  - ./dyna-2.md
sources:
  - ../../sources/sites/maxinsights-official.md
  - ../../sources/blogs/maxinsights-beyond-1m-hours-physical-ai-data.md
  - ../../sources/blogs/dyna-2-million-hour-scaling-law.md
summary: "Maxinsights 是一家自述位于旧金山湾区的 Physical AI 第一人称多模态数据服务商；官网披露累计经验与月产能指标，但未公开代码、可下载数据集或客户级数据来源。"
---

# Maxinsights：Physical AI 第一人称数据服务

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

## 官网公开的业务与规模

官网将其服务概括为大规模多模态采集、手部追踪、上/全身动作捕捉、触觉与力信号、仿真和模型评估。页面列出的结构化信号包括视频理解、深度、物体跟踪、手部姿态、上身姿态、触觉/力、定位、分割和点云。

| 官网指标 | 公司披露 | 读法 |
|----------|----------|------|
| 累计经验 | 2,000,000 小时 | 官网标注为截至 2026 年 9 月累计经验；不是某一客户的交付时长 |
| 月产能 | 最高 450,000 小时 | 产能上限，不等于每月实际采集或通过质检的可训练数据 |
| 场景覆盖 | 50,000+ 场景类型 | 公司自报的环境覆盖指标 |
| 结构化输出 | 9 类信号 | 公司列出的视频理解、深度、物体/手部/身体状态、力、定位、分割与点云等 |

这些数字均为企业官网自述，公开页面未说明独立审计方法、端到端合格率或各项指标的详细口径。评估服务能力时，应把原始采集时长、质检后可用时长、标注覆盖与客户实际交付分开核对。

## 数据集速查维度

| 维度 | 公开信息 | 备注 |
|------|----------|------|
| 规模 | 累计 2,000,000 小时、月产能最高 450,000 小时（官网自报） | 非可下载语料规模，未说明质检后可用时长 |
| 模态 | 第一人称视频，外加深度、物体跟踪、手部/上身姿态、触觉/力、定位、分割、点云 | 来自官网结构化信号列表 |
| 许可证 | 无公开数据集与许可证；以商业数据服务形式交付 | 客户级授权与使用条款未公开 |
| 重定向就绪度 | 输出为人类第一人称视频与手/身体姿态，不是机器人本体数据 | 用于机器人训练需按目标形态做手部/身体重定向，官网未说明是否提供该环节 |

## 与百万小时人类视频预训练的关系

Dyna Robotics 的 Dyna-2 官方报告称，模型使用超过 100 万小时的人类第一人称视频预训练，并报告了随数据规模上升的留出数据表现。报告提到语料来自未具名的数据合作方和 Dyna 内部采集；没有列出 Maxinsights。

Maxinsights 的官方文章讨论了 Dyna-2，并提出“经验规模”和“经验密度”两个观察维度，但没有声称为 Dyna-2 供应数据。因此，**Dyna-2 的百万小时研究可作为该产业方向的背景，不能据此推定 Maxinsights 是其供应商**。

## 工程实践与开放状态

- 官网描述的生产闭环包括任务设计、采集、质检和结构化输出；业务可见输出覆盖视频、手部/身体姿态、空间与接触信息。
- 截至 2026-10-22，检查到的官网和研究文章没有链接公开代码仓库或可下载数据集；目前无法据公开资料复现其采集与处理管线。
- 官方公开材料中未核实“Maxego”算法名称及实现、5000 名采集员网络、2024 年成立时间或 Google DeepMind、Figure、Dyna Robotics 等具体客户关系。这些主张不在此页当作事实记录。
- Dyna-2 报告确认其语料使用未具名合作方，但没有将合作方与 Maxinsights 联系起来。若未来出现公司公告、客户公告或公开合同信息，应再补充供应关系证据。

## 结论

- Maxinsights 官网证实其业务方向是第一人称多模态 Physical AI 数据采集与加工；其公开产能和覆盖规模属于公司自报指标。
- 需要把“累计经验”“月度产能”“质检后可用时长”和“客户交付量”分开比较。
- Dyna-2 官方报告证明百万小时人类视频已被用于机器人模型预训练，但没有识别其具体数据供应商。
- 当前公开资料不足以验证 Maxego、采集员人数、成立年份及所列客户关系。

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
- [Maxinsights：Beyond 1M Hours](https://www.maxinsights.ai/blog/Beyond_1M_Hours)
- [Dyna Robotics：Dyna-2 技术报告](https://www.dyna.co/dyna-2)
