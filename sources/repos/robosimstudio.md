# RoboSimStudio

> 来源归档

- **标题：** RoboSimStudio
- **类型：** repo
- **来源：** RoboSimStudio 项目团队
- **链接：** <https://github.com/robosimstudio/robosimstudio>
- **项目页：** <https://robosimstudio.github.io/>
- **技术报告：** <https://arxiv.org/abs/2610.18835>
- **入库日期：** 2026-10-15
- **许可证：** Apache-2.0
- **一句话说明：** 开放交互式 GPU 仿真工具包，用于通过人工或编码 Agent 将真实机器人操作任务重建到仿真中。
- **沉淀到 wiki：** 是 → [wiki/entities/robosimstudio.md](../../wiki/entities/robosimstudio.md)

---

## 开源状态核查

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — 官方 GitHub 仓库公开，Apache-2.0 |
| **项目页** | <https://robosimstudio.github.io/> |
| **技术报告** | <https://arxiv.org/abs/2610.18835> |
| **核心依赖** | [Newton](https://github.com/newton-physics/newton)、[Warp](https://github.com/NVIDIA/warp)、[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) |
| **示例** | 仓库提供 examples/；README 给出由编码 Agent 根据本地视频创建仿真环境的提示词范例 |

## README 摘要

项目目标是让研究者或编码 Agent 快速构建机器人操作任务的交互式物理仿真。项目页展示从机器人擦拭手机视频生成全动态数字孪生的流程；演示对象覆盖海绵、线缆和布料。官方 README 将项目描述为面向 manipulation tasks 的 Real2Sim 工具，并将 Newton、Warp 与 MuJoCo Warp 列为基础组件。

## 快速开始线索

官方 README 推荐查看仓库 examples/，或让 Codex / Claude Code 克隆仓库，再根据任务描述和本地视频创建仿真环境。具体安装依赖、硬件要求与示例命令以仓库当前 README 和 examples 为准。

## 对 wiki 的映射

- RoboSimStudio 工具页：[wiki/entities/robosimstudio.md](../../wiki/entities/robosimstudio.md)
- Newton 引擎：[wiki/entities/newton-physics.md](../../wiki/entities/newton-physics.md)
