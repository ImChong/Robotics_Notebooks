# Beyond 1M Hours: The Data Scaling Laws of Physical AI

> 来源归档（Maxinsights 官方研究文章）

- **原文：** <https://www.maxinsights.ai/beyond-1m-hours.html>（旧址 `/blog/Beyond_1M_Hours` 已重定向）
- **发布：** 2026-09-09
- **发布方：** Maxinsights Research
- **类型：** 企业研究/观点文章，非同行评审论文
- **核查日期：** 2026-11-12
- **核对范围：** 通读《Beyond 1M Hours》全文，包括 Dyna-2 与 GENE-26.5 两个案例、Experience Scale、Experience Density、Scale × Density、10M 小时目标、指标建议与全文结论。
- **通读：** 已完成

## 核心内容

文章将 Physical AI 数据扩展分成两个相互关联的维度：

1. **Experience Scale：** 人类真实操作经验的小时数。
2. **Experience Density：** 每小时保留多少可学习的物理信息，例如手部/身体姿态、物体状态、接触、力与时序。

文章以 Dyna-2 和 Genesis AI 的公开成果作为产业案例，并讨论采集、同步、质检、标注与覆盖率对可用训练数据的影响。作者提出从“每录制小时成本”转向“每可用训练小时成本”，进一步追踪状态/接触覆盖、增量信息收益与端到端通过率；`有效经验≈时长×每小时信息量` 被作者明确界定为核算框架，**不是已拟合验证的幂律**。文章用四道质检关口的示例解释：每关通过率 95% 时总通过率约 81%，均为 99% 时约为 96%。这些是概念性计算，非 Maxinsights 真实产线的审计结果。文章所述价值判断属于企业研究观点，不能视为客户、算法或模型效果的独立验证。

**归属边界：** 文章讨论 Dyna-2 的百万小时人类视频研究，但没有声称 Maxinsights 为 Dyna-2 供应数据。Dyna-2 的训练数据规模和方法应以 [Dyna Robotics 官方技术报告](https://www.dyna.co/dyna-2)为准；该报告提到数据合作方与内部采集，但未点名合作方。

## Wiki 映射

- [Maxinsights：Physical AI 第一人称数据服务](../../wiki/entities/maxinsights-physical-ai-data.md)
- [具身 Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md)
