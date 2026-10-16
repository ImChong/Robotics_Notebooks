# RoboSimStudio

> 来源归档

- **标题：** RoboSimStudio
- **类型：** repo
- **来源：** UC Berkeley AUTOLab 与 NVIDIA 团队
- **链接：** <https://github.com/robosimstudio/robosimstudio>
- **项目页：** <https://robosimstudio.github.io/>
- **技术报告：** <https://arxiv.org/abs/2610.18835>
- **入库日期：** 2026-10-15；补充核查：2026-10-16
- **许可证：** Apache-2.0
- **一句话说明：** 面向机器人操作 Real2Sim 的开放交互式 GPU 仿真工具包，可由人或编码 Agent 根据真实任务视频搭建、检查并调整物理场景。
- **沉淀到 wiki：** 是 → [wiki/entities/robosimstudio.md](../../wiki/entities/robosimstudio.md)

---

## 开源状态核查

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — 官方 GitHub 仓库公开，Apache-2.0 |
| **项目页** | <https://robosimstudio.github.io/> |
| **技术报告** | <https://arxiv.org/abs/2610.18835> |
| **核心组件** | Newton、Warp、MuJoCo Warp；项目页演示将 Newton 中的刚体仿真与 SuperDex 海绵软体 FEM 耦合 |
| **示例** | 仓库提供 examples/；README 给出由 Codex 或 Claude Code 根据任务指令和本地视频创建仿真环境的提示词范例 |

## README 摘要

官方 README 将 RoboSimStudio 描述为开放交互式 GPU 仿真工具包，可由人工或编码 Agent 重建真实机器人操作任务。它构建在 Newton、Warp 和 MuJoCo Warp 等组件上，并从 SuperDex、Isaac Lab、PyRoKi、Genesis、LeRobot 等项目汲取实现或模型资源。README 展示 48 个并排运行的操作任务场景；技术报告另介绍 18 个接触丰富操作任务仿真，二者口径不同，不应混为一个基准数量。

项目页与报告展示的代表性流程是：将双臂机器人擦拭任务的手机视频与一行提示交给 Claude 或 Codex；Agent 搭建场景、相机和物理模型，再由人检查、调整。海绵擦拭案例约 56 分钟完成可运行仿真。场景以 Newton 中的 MuJoCo Warp 处理手臂和刚体，并用 SuperDex FEM 模拟海绵形变。

报告还展示以仿真反馈迭代 Graph-as-Policy，并将改进后的策略部署到真机完成擦拭任务。真机运行暴露出支撑杆固定约束与实物不符的问题；调整约束后，仿真能够表现支撑杆移动和倾倒。该结果是具体任务演示，不代表对任意任务的仿真精度或 Sim2Real 成功率保证。

## 快速开始线索

官方 README 推荐查看 examples/，或让 Codex / Claude Code 克隆仓库，再根据任务描述和本地视频创建仿真环境。具体安装依赖、硬件要求与示例命令以仓库当前 README 和 examples 为准。

## 对 wiki 的映射

- RoboSimStudio 工具页：[wiki/entities/robosimstudio.md](../../wiki/entities/robosimstudio.md)
- Newton 引擎：[wiki/entities/newton-physics.md](../../wiki/entities/newton-physics.md)
- SuperDex 软体物理组件：[Project SuperDex](https://github.com/facebookresearch/project_superdex)
