# Knee Exoskeleton Sim-Derived Torque（arXiv:2610.41981）

> 来源归档（paper）

- **标题：** Design and Validation of a Robotic Knee Exoskeleton with Musculoskeletal Simulation-Derived Biological Torque Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.41981>
- **PDF：** <https://arxiv.org/pdf/2610.41981>
- **代码：** <https://github.com/MetaMobilityLabCMU/Knee-Exoskeleton-Project>
- **入库日期：** 2026-11-03
- **线索来源：** [机器人不仅要完成任务，还要知道哪里出了错](../blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)（具身智能小站，2026-11-03）
- **一句话说明：** 从肌骨仿真提取膝关节力矩并蒸馏成传感器估计器，减少新硬件专门采集训练数据的需求。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-03）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 用仿真训练外骨骼辅助控制。人体实验针对健康参与者下蹲，不代表临床疗效。
- 从肌骨仿真提取膝关节力矩并蒸馏成传感器估计器，减少新硬件专门采集训练数据的需求。
- 指标：公众号转述：减少硬件数据依赖：由肌骨仿真生成训练信号，再蒸馏为仅使用机载传感器的力矩估计器；七名健康参与者的下蹲结果不能外推至康复人群。

## 对 wiki 的映射

- [paper-knee-exoskeleton-sim-derived-torque](../../wiki/entities/paper-knee-exoskeleton-sim-derived-torque.md)
- [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-03-49-papers-technology-map.md)
