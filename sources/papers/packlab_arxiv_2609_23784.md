# PackLab（arXiv:2609.23784）

> 来源归档（paper）

- **标题：** PackLab: A Comprehensive Framework for Developing, Training, and Evaluating MLLMs in Robotic Bin Packing
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.23784>
- **PDF：** <https://arxiv.org/pdf/2609.23784>
- **代码：** <https://github.com/Correr-Zhou/PackLab>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 把仿真平台、专用多模态模型和分难度评测连在一起，让模型随箱内状态更新选择物体与落点。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 装箱不仅看单次落点，还要看后续空间；装入体积与紧凑度共同评价布局。
- 仿真平台 + 专用 MLLM + 分难度评测，随箱内状态更新决策。
- 指标：公众号称主要证据来自物理仿真；未给具体数字。

## 对 wiki 的映射

- [paper-packlab](../../wiki/entities/paper-packlab.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
