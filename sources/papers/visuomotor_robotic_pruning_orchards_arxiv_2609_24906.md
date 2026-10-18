# Visuomotor Pruning（arXiv:2609.24906）

> 来源归档（paper）

- **标题：** Visuomotor Robotic Pruning in Planar Orchards Using Hybrid Reinforcement Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24906>
- **PDF：** <https://arxiv.org/pdf/2609.24906>
- **代码：** <https://github.com/OSUrobotics/lpy_treesim>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 用腕部相机光流驱动闭环控制，混合离线示范与在线仿真学习，在平面整形果树上做修枝。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：仓库可达（git ls-remote）；仓库名为 lpy_treesim（树木仿真），是否含完整训练代码未核实。

## 核心摘录（公众号转述，未核验原文表格）

- 不必完整三维重建也能闭环引导修枝；纯仿真训练后可迁移实地。
- 腕部相机光流 → 闭环控制；离线示范 + 在线仿真混合 RL。
- 指标：公众号提示：范围是平面整形果树与预先指定的切点；未给具体数字。

## 对 wiki 的映射

- [paper-visuomotor-robotic-pruning-orchards](../../wiki/entities/paper-visuomotor-robotic-pruning-orchards.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
