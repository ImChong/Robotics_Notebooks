# RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation

> 来源归档

- **类型：** 论文 / 数据集基准
- **作者：** Kun Wu 等，RoboMIND Team
- **机构：** 北京人形机器人创新中心、北京大学、北京智源人工智能研究院（项目页列示）
- **发表：** Robotics: Science and Systems XXI（RSS 2025）；DOI: 10.15607/RSS.2025.XXI.152
- **arXiv：** https://arxiv.org/abs/2412.13877
- **RSS 论文页：** https://www.roboticsproceedings.org/rss21/p152.html
- **项目页：** https://x-humanoid-robomind.github.io/
- **数据集：** https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND
- **入库日期：** 2026-10-26
- **一句话说明：** 以统一遥操作采集协议构建跨机器人本体操作数据集，并提供多视角观测、本体状态、语言描述、失败演示和数字孪生评测资源。

## 论文摘录要点

1. **统一采集标准：** RoboMIND 汇集 4 种机器人本体、479 项任务和 96 类物体的遥操作演示，记录多视角观测、机器人状态与语言任务描述；RSS 页面摘要报告 107K 条演示轨迹。
2. **失败数据与数字孪生：** 论文介绍约 5K 条真实失败演示及其原因标注，并构建 Isaac Sim 数字孪生环境，用于低成本补充数据与评测。
3. **版本口径需要区分：** 当前 Hugging Face 数据卡的版本说明称初始 1.0 有 55K 条轨迹、279 个任务、69 类物体；后续 1.1/1.2 扩展至 107K、479 项任务和 96 类物体。引用时应明确采用论文摘要还是数据卡的具体版本定义。

## 对 wiki 的映射

- [RoboMIND 主实体](../../wiki/entities/painode-094-robomind.md)：版本口径、数据组成、跨本体使用注意事项。
- [机器人操作任务](../../wiki/tasks/manipulation.md)：多任务操作数据与评测语境。
- [VLA 方法](../../wiki/methods/vla.md)：数据作为视觉语言动作策略训练与评测来源。

## 参考链接

- RSS 官方论文页：<https://www.roboticsproceedings.org/rss21/p152.html>
- arXiv：<https://arxiv.org/abs/2412.13877>
- 项目主页：<https://x-humanoid-robomind.github.io/>
- Hugging Face 数据卡：<https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND>
