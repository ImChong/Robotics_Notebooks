# Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models

> 来源归档（Dyna Robotics 官方技术报告）

- **原文：** <https://www.dyna.co/dyna-2>
- **发布：** 2026-08
- **发布方：** Dyna Robotics
- **核查日期：** 2026-10-22

## 一手资料摘录

- Dyna Robotics 报告称 Dyna-2 使用超过 **100 万小时**的人类第一人称视频预训练，并以 1,000、10,000、100,000 和 1,000,000 小时构造嵌套规模的数据实验。
- 报告称语料主要是头戴式日常操作视频，由数据合作方和 Dyna 内部运营共同采集；Dyna 使用清洗、手部姿态提取、验证与过滤流程。
- 通过 3D 手部姿态轨迹，报告构造腕部末端轨迹和连续抓握信号作为 pseudo-action 监督。
- Dyna 报告其模型在留出的人类与机器人数据评测中随预训练规模提升，并将视频世界建模视为跨具身扩展的重要因素。此为作者报告的实验结论。

**归属边界：** 报告仅提到未具名的数据合作方，没有列出 Maxinsights；因此该一手资料支持 Dyna-2 的数据规模和训练方法，不支持具体供应商归属。

## Wiki 映射

- [Dyna-2](../../wiki/entities/dyna-2.md)
- [Maxinsights：Physical AI 第一人称数据服务](../../wiki/entities/maxinsights-physical-ai-data.md)
