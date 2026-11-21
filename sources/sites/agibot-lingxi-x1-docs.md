# 智元灵犀 X1 官方文档中心与开发指南

> 来源归档（官方站点与开发文档）

- **标题：** 智元灵犀 X1 开发指南（Agibot DOCS）
- **类型：** site / docs
- **机构：** 智元机器人（Agibot）
- **官方入口：** <https://www.agibot.com.cn/DOCS>（英文版：<https://agibot.com/DOCS/X1>、<https://agibot.com/DOCS/OS/X1-PDG>）
- **开源仓库：**
  - 硬件设计：<https://github.com/AgibotTech/agibot_x1_hardware>
  - 强化学习训练：<https://github.com/AgibotTech/agibot_x1_train>
  - 真机/仿真推理部署：<https://github.com/AgibotTech/agibot_x1_infer>
- **入库日期：** 2026-11-21
- **核对范围：** 通读中文文档中心 <https://www.agibot.com.cn/DOCS> 与英文文档 <https://agibot.com/DOCS/X1>、<https://agibot.com/DOCS/OS/X1-PDG> 全文（涵盖整机结构、硬件/代码入口、执行器固件与 MIT 控制模式、桌面联调、IMU 与上电标零流程），并核对 `AgibotTech/agibot_x1_train@e6651b9` 与 `AgibotTech/agibot_x1_infer@9e0b818` 的 README 与入口源码；未运行实机硬件或 Isaac Gym 训练。
- **通读：** 已完成
- **一句话说明：** 智元官方发布的灵犀 X1 开源人形全栈指南，包含 29 自由度（R86-2×9、R86-3×6、R52×10、L28×4）+ 双夹爪结构、BOM/SolidWorks/装机 SOP、PowerFlow 执行器标定与 AimRT 推理部署全流程。

## 官方资料摘要

1. **整机硬件规格：** 全身 29 个驱动关节（R86-2×9、R86-3×6、R52×10、L28×4）+ 2 个夹爪，支持扩展 3 自由度头部；IMU 采用 YIS320（串口波特率 921600，接下肢 DCU）。
2. **三类开源仓库：** `agibot_x1_hardware` 提供 BOM、STEP 与 SolidWorks 图纸（亦提供百度/Google 网盘镜像）；`agibot_x1_train` 基于 Isaac Gym Preview 4 + RSL-RL/Humanoid-Gym 扩展 `DHPPO` 训练并支持 MuJoCo `sim2sim.py` 与 ONNX 导出；`agibot_x1_infer` 基于 AimRT 中间件 + ROS 2 Humble + ONNX Runtime 实现 1000 Hz 实时控制与仿真运行。
3. **联调与标定流程：** 执行器通过 `REF-CLI v1.0.3.exe`（Windows USB-C）查询 SN 版本与 MIT 混合控制模式（模式 6）、分配 CAN ID 并标零（L28 上电自动回零，不支持手动标零）；桌面联调使用 `xyber_dcu_test_v0.1.3.zip` 在 Ubuntu 22.04 实时系统验证 EtherCAT/DCU/IMU 通信后，再按 SOP 整机组装与标零。

## 对 wiki 的映射

- [智元灵犀 X1（Agibot 开源人形）](../../wiki/entities/agibot-lingxi-x1.md)
- [agibot_x1_train 源码节点](../../wiki/entities/cn-os-agibot-x1-train.md)
- [agibot_x1_infer 源码节点](../../wiki/entities/cn-os-agibot-x1-infer.md)
- [agibot_x1_hardware 源码节点](../../wiki/entities/cn-os-agibot-x1-hardware.md)
