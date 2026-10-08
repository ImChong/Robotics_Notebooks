# rai-opensource/exploy

> 来源归档

- **标题：** Exploy: EXport and dePLOY Reinforcement Learning policies
- **类型：** repo
- **机构：** Robotics and AI Institute（RAI Institute）
- **链接：** <https://github.com/rai-opensource/exploy>
- **主页 / 文档：** <https://rai-opensource.github.io/exploy/>
- **作者：** Dario Bellicoso、Annika Wollschläger（仓库元数据）
- **许可证：** MIT
- **版本：** `0.1.0`（README / pyproject.toml / CMake 均对齐）
- **入库日期：** 2026-10-08
- **一句话说明：** Python 导出器把强化学习环境与策略的观测、前向推理和动作后处理整合为 ONNX 图；C++ ONNX Runtime 控制器与 ROS wrapper 提供部署侧接口。
- **开源状态：** **已开源**（Python、C++、ROS、示例、测试和文档均在仓库中；MIT）
- **沉淀到 wiki：** 是 → [`wiki/entities/exploy.md`](../../wiki/entities/exploy.md)

---

## 仓库结构与能力

| 路径 / 模块 | 职责 |
|-------------|------|
| `python/exploy/` | Python exporter；核心库与 framework adapter |
| `control/` | C++20 controller，依赖 ONNX Runtime、Eigen、nlohmann-json、fmt |
| `ros/` | ROS wrapper `exploy_vendor` |
| `examples/exporter_scripts/` | Isaac Lab、MjLab 导出示例与测试 |
| `examples/controller/` | 加载导出 ONNX 的 C++ controller 示例 |

README 声明内置 Isaac Lab 与 MjLab 适配，并面向可扩展的 PyTorch 环境。Exporter 支持注册 inputs、outputs、groups、memory；文档示例生成的 ONNX 分为策略频率运行的 Default 子图，以及仿真频率的 ProcessActions 子图。循环策略可通过 memory 张量维持跨周期状态。

## 运行入口

```bash
# 在源码仓库使用 Pixi 环境
git clone https://github.com/rai-opensource/exploy.git
cd exploy
pixi install
pixi run build

# 导出示例策略
pixi run export-isaaclab
pixi run export-mjlab

# 用 C++ loopback controller 运行对应示例
pixi run run-cpp-example-isaaclab
pixi run run-cpp-example-mjlab

# ROS 2 wrapper
colcon build --packages-up-to exploy_vendor
```

使用 Python 包时可直接从 Git 安装；Isaac Lab / MjLab extra 分别为 `exploy[isaaclab]`、`exploy[mjlab]`。C++ 库亦可独立 CMake 构建安装。运行仓库示例需要按文档准备对应框架环境；GPU 示例明确假设 NVIDIA GPU。

## 复现与部署边界

- 仓库带 Python 与 C++ 测试、Pixi 锁定环境、导出和 controller 示例；这使它不仅是概念性代码或仅有权重的项目。
- 控制器通过 `RobotStateInterface`、`CommandInterface`、`DataCollectionInterface` 抽象设备接入；自定义机器人仍要实现这些接口及必要的 matcher。
- `LICENSE` 为 MIT；模型能否完整导出仍取决于 PyTorch tracing 路径及 ONNX 支持的算子。项目提供输出对比 evaluator，但目标硬件上的实时性、安全与驱动集成需要使用方验证。
- RAI 博客列举真实平台使用案例，但仓库与文章未提供可横向比较的部署 benchmark。

## 对 wiki 的映射

- [Exploy](../../wiki/entities/exploy.md)
- [Exploy 官方项目文档](../sites/exploy-docs.md)
- [原始介绍文章](../blogs/introducing_exploy_rai_2026-10-07.md)