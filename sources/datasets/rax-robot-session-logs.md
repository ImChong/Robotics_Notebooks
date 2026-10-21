# RAX robot session logs（Franka、RB-Y1）

> 来源归档（ingest）

- **类型：** dataset / robot-agent session logs
- **数据集：** <https://huggingface.co/datasets/robot-agent-exploration/robot-session-logs>
- **许可证：** CC BY 4.0（数据集卡）
- **配套代码：** [RAX](https://github.com/robot-agent-exploration/rax) — [代码归档](../repos/rax.md)
- **技能库：** [RAX Skill Library](https://github.com/robot-agent-exploration/skills-lib) — [归档](../repos/skills-lib.md)
- **入库日期：** 2026-10-20

## 数据内容

数据集卡列出 5 段 Franka FR3 会话和 7 段 Rainbow RB-Y1 会话。记录包括 coding agent 对话与工具结果、机器人工具调用、机器人状态、相机录制、操作者指令，以及 agent 在会话中生成的代码、笔记与技能；每段会话配有说明和统计。部分早期 RB-Y1 会话没有视频或连续状态录制，具体覆盖以数据卡为准。

该数据更适合研究机器人 agent 如何从在线交互形成可复用技能、审计工具调用与分析错误恢复。它是有限数量的真实机器人会话记录，不应视为大规模通用行为训练集，也不能仅凭其中的成功演示推断策略可靠性。

## 使用边界

- 解包前先查看数据卡、会话清单及 CC BY 4.0 署名要求。
- 日志含操作者指令、agent transcript、相机视频和机器人状态；用于发布、训练或二次共享前应审查可能暴露的个人信息、环境信息及第三方内容。
- 数据集卡说明媒体经过人脸模糊清单处理；这不替代对原始日志和其他敏感信息的审查。

## 关联资料

- 代码：[RAX 官方代码仓](../repos/rax.md)
- 技能：[RAX Skill Library](../repos/skills-lib.md)
- wiki：[RAX 实体页](../../wiki/entities/rax-scaling-online-exploration.md)
