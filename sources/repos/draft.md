# Draft — 官方源码仓库归档

> 来源归档（项目代码与运行入口）

- **仓库：** <https://github.com/davidhnguyen2000/draft>
- **主语言 / 版本：** Python；`draft-robots` 0.1.0，Python >=3.10
- **许可证：** MIT（代码）；厂商机器人模型资产有各自许可
- **状态：** 公开开源
- **入库日期：** 2026-10-12
- **对应实体：** [Draft：参数化机器人设计探索](../../wiki/entities/paper-draft-parametric-robot-design.md)
- **论文：** [arXiv:2609.38405](https://arxiv.org/abs/2609.38405)

## 功能概览

Draft 将短小的机器人规格编译为仿真就绪 MuJoCo MJCF，不依赖 CAD。设计者指定连杆长度和关节扭矩/速度需求；拟合趋势补足质量、电机尺寸、转子惯量等属性，并可筛查模型是否落在经验可行域。论文演示 humanoid 与 quadruped 参数化模型、四款商用平台 twin，以及三种四足在统一 RL 课程下的能力比较。

## 代码地图

| 路径 | 用途 |
|---|---|
| `src/draft/generation` | 参数表与运动学树编译到 MJCF |
| `src/draft/trends` | 执行器/结构拟合关系、可行性筛查 |
| `src/draft/robots` | humanoid/quadruped 规格 |
| `src/draft/twins` | 平台 twin 复现 |
| `src/draft/tasks`、训练/评测模块 | quadruped locomotion 任务与课程 |
| `src/draft/editor` | 浏览器规格编辑界面 |
| `datasets/` | 执行器目录、公开机器人描述等数据 |
| `scripts/` | 生成、编辑、设置外部数据及 RL 工作流入口 |

## 安装与运行入口

官方 README 建议创建 conda 环境，以 editable 模式安装项目及相应用途 extras。核心入口：

| 目的 | 命令 |
|---|---|
| 打开浏览器规格编辑器 | `python scripts/robot_tuner.py` |
| 无头生成 quadruped MJCF | `python scripts/generate_robot.py --robot quadruped` |
| 获取非随仓分发的机器人模型 | `python scripts/setup_data.py` |

项目依赖 MuJoCo、NumPy、PyYAML、trimesh、SciPy 等；编辑器和 twins/figures 为可选 extras。RL extra 包含 mjlab、MuJoCo Warp、Warp、rsl-rl-lib、PyTorch 等，官方 README 指出训练需要 CUDA GPU。基础生成器不需要训练级 GPU。

## 许可与复现注意

- MIT 适用于 Draft 代码；不要据此推断仓库引用/下载的所有机器人模型也采用 MIT。
- 厂商 twin 模型未直接随项目仓库分发；运行数据设置脚本后应逐个遵循其上游许可证。
- README 描述的生成器/筛查是早期设计探索工具，不是 CAD/FEA、安全校核或自动外层优化器的替代。
- 项目依赖/训练生态较重；只验证生成流程可从基础依赖开始，训练实验才需 RL/CUDA 环境。
