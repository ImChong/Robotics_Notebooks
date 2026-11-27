# VIA: Visual Interface Agent for Robot Control

- **作者：** Hengyuan Hu、Jensen Gao、Priya Sundaresan、Satvik Sharma、Jeannette Bohg、Dorsa Sadigh
- **机构：** 斯坦福大学（Stanford University）
- **版本 / 载体：** arXiv:2607.11119v2 [cs.RO]（2026-07-13 提交 v1，2026-10-07 修订 v2）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2607.11119)、[arXiv 摘要页](https://arxiv.org/abs/2607.11119)
- **项目页 / 代码：** <https://github.com/hengyuan-hu/via>（已开源仿真与 MCP 服务代码；MIT License；当前仓库布局覆盖仿真环境、Browser 3D UI、MCP Server 与 Claude Code / Codex 评测框架，未见 Stretch 4 真机控制端实现）
- **核对范围：** 全文通读 arXiv HTML v2 §1–5、Figure 1–7、Table 1–8 及 Appendix A–E（含仿真与真机 MCP 工具文档、工具调用次数与估算 API 成本、基线逐任务成功率、Stretch 4 平台规格与系统/任务提示词）。
- **通读：** 已完成

## 原文核对笔记

### 1. 问题定义与核心机制（§1–§3）

VIA 将机器人控制重构为通用基础模型（FM）智能体的视觉工具使用任务，无需针对机器人微调或设计动作原语。系统由「可视化交互界面 + 人类可读 MCP 工具 + 底层路点控制器」构成：

1. **虚拟工作区（§3.1）：** 仿真与稠密 RGB-D 桌面场景下，依据 RGB-D 传感器构建 3D 点云工作区并在界面右侧呈现，左侧提供第三人称与腕部相机画面；桌面/真机稀疏深度（Stretch 4 房间尺度）下，直接在 `head` 与 `wrist` 2D 相机图像上叠加红色虚拟目标夹爪（腕部为 `|----.----|` 符号）以及 `nav_map` 俯视图。所有输入均来自原始传感与本体状态，不使用特权状态。
2. **嵌套智能体闭环（§3.2）：** 外循环为「观测当前场景 → 规划并执行路点 → 观测真机/仿真后果 → 重规划」；内循环（mini-loop）允许智能体在调用 `execute_waypoint` 之前，利用 `hover`、`gripper_teleport_via_click`、`gripper_translate`、`gripper_rotate`、`camera_orbit_via_key` 等 MCP 工具在虚拟工作区中反复试错和可视化校核目标夹爪位姿，确认后再交由线性插值 PI 控制器或全身控制器执行。

### 2. 实验设置、主表结果与消融（§4, Appendix B–C）

- **仿真评测（Table 1, Table 6–8）：** 基于 `robosuite` 与 Franka Panda 机械臂（PI 控制器线性插值），覆盖 6 个任务（Turn on stove、Open drawer、Put bowl on plate、Square、Rainbow、T-block），每任务 10 个随机种子、推理强度 `xhigh`、禁用文件/网络/跨回合记忆以防读取仿真源码作弊。
  - VIA 默认提示词平均成功率：Claude Code Sonnet 5（43.3%）、Opus 4.8（65.0%）、Fable 5（85.0%）；Codex Terra 5.6（38.3%）、Sol 5.6（73.3%）、Astra 6（100.0%）。
  - 无虚拟工作区界面的朴素基线（Table 8，仅 5 个直发控制器工具）：CC Opus 4.8（6.7%）、CC Fable 5（23.3%）、Codex Terra 5.6（6.7%）、Codex Sol 5.6（25.0%）、Codex Astra 6（93.3%）。
  - 详细文本提示词消融（Table 2，5 项任务不含 Rainbow）：CC Opus 4.8 由 62% 升至 76%；Codex Sol 5.6 由 68% 升至 74%。
  - 平均成功回合工具调用次数（Table 6）与估算 API 成本（Table 7）：Codex Astra 6 均值为 51 次调用 / $3.2；CC Fable 5 为 72 次 / $16.3。
- **真机评测（§4.3, Table 3, Appendix D）：** Hello Robot Stretch 4（3-DoF 全向底盘 + 1-DoF 升降 + 1-DoF 伸缩臂 + 3-DoF 腕部，头部 LiDAR + 腕部双目经 Fast-FoundationStereo 计算深度），使用 Codex Astra 6 每任务测 5 回合，共 24/25（96%）：Lamp 4/5、Shelf 5/5、Towel 5/5、Wipe 5/5、Basket 5/5。

### 3. 局限与工程边界（§5, Appendix A/E）

- 推理延迟高与 API 成本限制使其仅适用于准静态任务（无法胜任接球等动态任务）。
- 依赖简单线性插值 PI / 全身控制器，缺乏显式避障能力；当前仅覆盖平行夹爪而非灵巧手。
- 真机深度噪声较大，工具文档与系统提示词明确要求以视觉图像判断为准，不可盲信 `hover` 或 `show_approach_axis` 深度读数。

对 wiki 的映射：[VIA: Visual Interface Agent for Robot Control](../../wiki/entities/paper-agentic-2607-11119.md)。
