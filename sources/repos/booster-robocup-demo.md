# BoosterRobotics/robocup_demo

- **标题**: Booster Robotics RoboCup Demo Framework
- **链接**: [https://github.com/BoosterRobotics/robocup_demo](https://github.com/BoosterRobotics/robocup_demo)
- **类型**: repo
- **作者**: Booster Robotics
- **摘要**: Booster Robotics 为其人形机器人（Booster K1/T1）提供的 RoboCup 足球比赛官方演示框架。基于 ROS 2 Humble，涵盖了基于 YOLOv8 的感知、基于状态机的决策以及比赛控制模块。

## 核心要点

1. **硬件适配**: 原生支持 Booster K1（固件 1.5.2+），支持 Booster T1 及 NVIDIA Jetpack 6.2。
2. **感知系统 (`vision`)**: 基于 YOLOv8 的目标检测，支持 TensorRT 加速。能够识别足球、机器人及场地特征。
3. **决策中心 (`brain`)**: 融合视觉与 GameController 状态，控制机器人执行搜索、追逐、对齐和踢球逻辑。
4. **强化学习集成**: 包含 `RLVisionKick`（基于 RL 的视觉踢球）功能，提升动态踢球精度。
5. **仿真支持**: 与 Booster Studio 模拟器深度集成，支持在虚拟环境中验证逻辑。

## 为什么值得保留

- 代表了国产人形机器人（Booster 系列）在自主足球竞技场景下的完整技术方案。
- 展示了如何将大规模视觉模型（YOLOv8）与实时控制系统（Booster SDK）集成的工程实践。
- 提供了从比赛协议（GameController）到物理执行的端到端参考。

## 对 wiki 的映射

- `wiki/entities/booster-robocup-demo.md`: 创建详细的系统说明页。
- `wiki/entities/unitree-g1.md` / `wiki/entities/anymal.md`: 对比不同平台的 demo 实现。
- `wiki/methods/auto-labeling-pipelines.md`: 视觉模型训练数据相关。

---
- **录入日期**: 2026-04-27
