# Isaac ROS Deploy

> 来源归档

- **标题：** Isaac ROS Deploy
- **类型：** repo
- **链接：** https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_deploy
- **机构：** NVIDIA
- **代码：** https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_deploy（**已开源**，Apache-2.0；步骤 2.5 以 GitHub 为准，无独立项目页）
- **Stars：** ~7（2026-09-25）
- **入库日期：** 2026-09-25
- **一句话说明：** ROS 2 侧加载 LEAPP 策略 bundle，经 Triton 跑 ONNX 推理，把 policy term 映射到 ROS topic 或 `ros2_control`，可选 SafetyController 门控。
- **沉淀到 wiki：** [isaac-ros-deploy](../../wiki/entities/isaac-ros-deploy.md)

---

## 核心定位

**Isaac ROS Deploy** 把 **Python 训练栈**（Isaac Lab RL、GR00T 经 [gr00t-leapp-export](https://github.com/nvidia-isaac/gr00t-leapp-export) 等）导出的 **[LEAPP](https://nvidia-isaac.github.io/leapp/) bundle** 接到 **真机或仿真机器人** 的 ROS 2 / `ros2_control` 环上：加载 bundle → **NVIDIA Triton** ONNX 推理 → 观测/动作 term 与 ROS 接口对齐 → 可选 **SafetyController** 混合/限幅。

官方文档：<https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html>

**2026-09-21 更新（README Latest）：** 增加 InferenceController 调试 topic（flattened 输入/选定输出 tensor）；补充 LEAPP runtime、URDF、ROS 通信与安全控制器排障指南。**本 release 不含 Isaac Sim 内部署支持。**

---

## 仓库结构（摘要）

| 路径 | 角色 |
|------|------|
| `isaac_deploy/isaac_deploy_core` | 核心运行时与 LEAPP 加载 |
| `isaac_deploy/isaac_ros_deploy_bringup` | Launch / 组合入口 |
| `isaac_deploy/isaac_ros_deploy_converters` | 消息与 policy term 转换 |
| `isaac_deploy/isaac_ros_deploy_interfaces` | 自定义 msg（如 `JointCommand`） |
| `isaac_deploy/isaac_ros_deploy_ros2_control` | `InferenceController`、`SafetyController` 插件 |
| `isaac_deploy/isaac_ros_deploy_reference_applications` | 参考应用 |
| `isaac_deploy/isaac_ros_inverse_dynamics` | 逆动力学相关辅助 |
| `isaac_ros_deploy/` | 顶层 metapackage / 聚合 |

---

## 对 wiki 的映射

- 实体页：[isaac-ros-deploy](../../wiki/entities/isaac-ros-deploy.md)
- 上游训练：[Isaac Lab](../../wiki/entities/isaac-lab.md)、[Isaac GR00T](../../wiki/entities/isaac-gr00t.md)
- 栈位置：[NVIDIA Physical AI 工具链技术地图](../../wiki/overview/nvidia-physical-ai-toolchain-technology-map.md) 第⑥段部署
- 推理 runtime：[TensorRT](../../wiki/entities/tensorrt.md)（LEAPP/Triton 链路与 TRT 导出并列）
