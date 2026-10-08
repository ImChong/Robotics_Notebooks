# Exploy 官方文档站

> 来源归档

- **标题：** Exploy 0.1.0 Documentation
- **类型：** site / 项目文档
- **机构：** Robotics and AI Institute（RAI Institute）
- **链接：** <https://rai-opensource.github.io/exploy/>
- **导出器教程：** <https://rai-opensource.github.io/exploy/tutorial/exporter/exporter_tutorial.html>
- **控制器教程：** <https://rai-opensource.github.io/exploy/tutorial/controller/controller_tutorial.html>
- **GitHub：** <https://github.com/rai-opensource/exploy>
- **版本线索：** 文档与仓库声明为 0.1.0（2026-10-08 复核）
- **许可证：** MIT（源码仓库 LICENSE）
- **入库日期：** 2026-10-08
- **一句话说明：** Exploy 的安装、Exporter / Controller 教程与 API 参考；覆盖 Isaac Lab、MjLab 的策略导出和 ONNX Runtime C++ 部署。
- **代码：** <https://github.com/rai-opensource/exploy>（已开源）
- **沉淀到 wiki：** 是 → [`wiki/entities/exploy.md`](../../wiki/entities/exploy.md)

---

## 文档结构与复现路径

| 文档 | 内容 |
|------|------|
| Getting Started | 环境安装、Python 导出器、C++/ROS 集成 |
| Exporter Tutorial | 实现环境 adapter，注册张量与 memory，导出 ONNX，再对比 ONNX 与原环境输出 |
| Controller Tutorial | 实现 `RobotStateInterface`、`CommandInterface`、`DataCollectionInterface`，加载 ONNX 并运行闭环控制周期 |
| Framework API | Isaac Lab 与 MjLab adapter |

官方示例以 Pixi 环境组织，提供 `export-isaaclab` / `export-mjlab` 导出任务和对应 C++ loopback controller 示例。示例运行说明假设有 NVIDIA GPU；通用 C++ 库则可通过 CMake 构建。ROS 2 可将 `ros/exploy_vendor` 放入 colcon workspace 后构建。

## 边界提示

文档将部署循环封装在通用控制器中，但使用者仍需将目标机器人的传感器/状态与命令映射到接口；数值等价性需要通过 evaluator 对原 PyTorch 环境与 ONNX 输出实测。