# robot-agent-exploration/skills-lib（RAX 技能库）

> 来源归档（ingest）

- **标题：** RAX Skill Library
- **类型：** repo
- **URL：** <https://github.com/robot-agent-exploration/skills-lib>
- **配套框架：** [RAX](https://github.com/robot-agent-exploration/rax) — [代码归档](rax.md)
- **配套论文：** [Scaling Online Exploration for Robot Agents](../papers/rax_scaling_online_exploration.md)
- **许可证：** README 显示根分支仓库未列 LICENSE 文件；使用或再分发前应核实。
- **入库日期：** 2026-10-20

## 一句话摘要

该仓库按机器人本体分支保存 RAX agent 在真机任务中写出的可执行技能、适用前提、检查、失败模式和运行证据，供后续会话读取、组合与改进。

## 公开结构核查

| 分支 | 本体 | README 所列内容 |
| --- | --- | --- |
| [`rby1`](https://github.com/robot-agent-exploration/skills-lib/tree/rby1) | Rainbow RB-Y1 轮式双臂机器人 | README 列出 6 项技能，包括双臂搬运、椅子归位、叠杯、全身伸手与组合整理任务。 |
| [`franka`](https://github.com/robot-agent-exploration/skills-lib/tree/franka) | Franka FR3 单臂 | README 列出 5 项技能，包括找方块、空气炸锅、擦桌、插销和 Bandu 塔。 |

每个技能目录含 `SKILL.md`、执行入口、运行记录与图像证据。库使用 `draft`、`verified`、`deprecated` 状态；README 定义 `verified` 为已提交代码至少两次无人值守成功运行。各本体技能存储在独立分支，`main` 保存协议与说明。

## 与 RAX 的关系

RAX harness 在会话工作区载入对应本体技能库；agent 可检查技能前提、运行或组合技能，并把改进后的内容并回技能库。技能的适用性仍取决于机器人、场景和验证条件，不能只凭同名技能认定可直接迁移。

## 关联资料

- 框架：[RAX 官方代码仓](rax.md)
- 数据：[RAX 真机 session logs](../datasets/rax-robot-session-logs.md)
- wiki：[RAX 实体页](../../wiki/entities/rax-scaling-online-exploration.md)
