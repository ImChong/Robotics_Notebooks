# robot-agent-exploration/rax（官方代码仓）

> 来源归档（ingest）

- **标题：** RAX: Scaling Online Exploration for Robot Agents
- **类型：** repo
- **URL：** <https://github.com/robot-agent-exploration/rax>
- **默认分支：** main（项目与分支索引；各实现位于下列分支）
- **配套论文：** [RAX（官方 PDF）](../papers/rax_scaling_online_exploration.md)
- **项目页：** <https://robot-agent-exploration.com/> — [项目页归档](../sites/rax-project.md)
- **许可证：** README 称 open-source；根目录未见 LICENSE，具体许可待核实
- **入库日期：** 2026-10-17

## 一句话摘要

RAX 提供统一的 coding-agent 探索与技能沉淀流程；根分支是索引，仿真与真机实现各自放在自包含分支中，通过机器人 API 接入，不要求修改机器人厂商控制栈。

## 分支与运行入口

| 分支 | 运行环境 | README 说明 |
| --- | --- | --- |
| [robodojo-behavior/container](https://github.com/robot-agent-exploration/rax/tree/robodojo-behavior/container) | RoboDojo、BEHAVIOR-1K 仿真 | Isaac Sim 5.1；agent CLI 在离线 Docker 容器中运行。 |
| [robodojo-behavior/native](https://github.com/robot-agent-exploration/rax/tree/robodojo-behavior/native) | RoboDojo、BEHAVIOR-1K 仿真 | 本机运行 agent CLI，使用 CLI 自身沙箱。 |
| [real/franka](https://github.com/robot-agent-exploration/rax/tree/real/franka) | Franka FR3 真机 | 7-DoF 单臂、Franka Hand、三台 ZED 相机。 |
| [real/rainbow](https://github.com/robot-agent-exploration/rax/tree/real/rainbow) | Rainbow RB-Y1 真机 | 轮式底盘、躯干、双臂、夹爪与头部 ZED 2i。 |

## 仓库核查

| 项 | 状态 |
| --- | --- |
| **公开可见** | 是；主分支 README 与上述实现分支可访问。 |
| **主要组件** | coding agent、MCP / Bash / Python 工具接口、机器人服务与各本体 backend；agent 在工作区保存 TASK.md、AGENTS.md、技能、记忆及编译程序。 |
| **运行闭环** | 探索时逐步调用观测与控制工具；总结成功序列并编译为 controller 程序；仿真在 held-out layouts 测试，真机由操作者复位后运行程序。 |
| **持续技能资产** | companion skills-lib 按机器人本体分支保存技能；README 称技能包含前提、检查、失败模式与运行记录，连续两次无人值守成功后标为 verified。 |
| **许可证** | 根目录未见 LICENSE，许可条款需进一步核实。 |
| **安全边界** | README 描述真机通过单独 robot service 发命令，agent 沙箱仅能通过 MCP 工具访问；物理急停与厂商控制器限值仍保留。 |

## 复现读法

1. 根据目标先选仿真或真机实现分支，每个分支有独立安装和运行指南。
2. 将 agent 可写工作区、机器人服务与硬件 backend 分开看；程序由探索生成，agent 不必在编译程序运行时持续推理。
3. 对照同一分支的 benchmark、场景划分和 session logs 复现实验；不要把不同任务的平均成功率和执行加速合并成单一指标。

## 关联资料

- 论文：[sources/papers/rax_scaling_online_exploration.md](../papers/rax_scaling_online_exploration.md)
- 项目页：[sources/sites/rax-project.md](../sites/rax-project.md)
- 技能库：[robot-agent-exploration/skills-lib](https://github.com/robot-agent-exploration/skills-lib)
- wiki：[wiki/entities/rax-scaling-online-exploration.md](../../wiki/entities/rax-scaling-online-exploration.md)
