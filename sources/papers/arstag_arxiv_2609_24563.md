# ARSTAG（arXiv:2609.24563）

> 来源归档（paper）

- **标题：** ARSTAG: An Agentic Real2Sim2Real System for Task-Specific Robot Data Generation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24563>
- **PDF：** <https://arxiv.org/pdf/2609.24563>
- **项目页：** <https://boweili666.github.io/ARSTAG/>
- **代码：** <https://github.com/boweili666/ARSTAG>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 协调场景重建、专家生成与数据扩增，把单张图像和任务指令转成仿真训练示范。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：项目页链接的仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 围绕任务构建仿真可减少逐任务的工程量，是单图到示范的自动化路径。
- 单张图像 + 指令 → 场景重建 → 专家示范生成 → 数据扩增，由智能体协调。
- 指标：公众号称真机验证覆盖抓取、放置和堆叠，仍有初始姿态限制；未给具体数字。

## 对 wiki 的映射

- [paper-arstag](../../wiki/entities/paper-arstag.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
