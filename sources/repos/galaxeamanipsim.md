# GalaxeaManipSim

> 来源归档（国内具身开源全景）

- **标题：** GalaxeaManipSim
- **类型：** repo
- **机构：** 星海图
- **链接：** https://github.com/OpenGalaxea/GalaxeaManipSim
- **分类：** 仿真环境
- **入库日期：** 2026-09-06
- **核对范围：** 已克隆 `OpenGalaxea/GalaxeaManipSim@abe7f51`（核对 `README.md`、`galaxea_sim/scripts/collect_demos.py`；未在本地物理运行 Isaac Sim/GPU 仿真环境）。
- **通读：** 已完成
- **一句话说明：** 星海图（Galaxea AI）开源的双臂操作仿真环境，面向 Galaxea R1 / R1 Pro / R1 Lite，提供 30+ 仿真环境与 17 项任务及 mplib 专家轨迹采集/LeRobot 格式导出工具，MIT 许可。
- **沉淀到 wiki：** [`wiki/entities/cn-os-galaxeamanipsim.md`](../../wiki/entities/cn-os-galaxeamanipsim.md)

## 开源状态

- **已开源**：公开仓库（MIT License，<https://github.com/OpenGalaxea/GalaxeaManipSim>）。

## 一手仓库要点摘录（`OpenGalaxea/GalaxeaManipSim@abe7f51`）

- **支持本体与控制模式**：Galaxea R1、R1 Pro、R1 Lite；提供 `bimanual_joint_position` 与 `bimanual_relaxed_ik` 控制模式。
- **数据管线**：`galaxea_sim/scripts/collect_demos.py` 基于 mplib 运动规划自动采集专家演示并保存 HDF5，配套回放脚本与 LeRobot 数据集转换脚本。
- **基线成功率（README 基准表）**：R1 `Dual Bottles Pick Easy` 98%；R1 Pro `Blocks Stack Easy` OpenDP 68% / LeRobot 64%；R1 Lite `Blocks Stack Easy` OpenDP 51% / LeRobot 42%。

## 对 wiki 的映射

- [wiki/entities/cn-os-galaxeamanipsim.md](../../wiki/entities/cn-os-galaxeamanipsim.md)
