# RAX: Scaling Online Exploration for Robot Agents（官方项目页）

> 来源归档（ingest）

- **标题：** RAX: Scaling Online Exploration for Robot Agents
- **类型：** site / project-page
- **URL：** <https://robot-agent-exploration.com/>
- **论文 PDF：** <https://robot-agent-exploration.com/rax.pdf>
- **代码：** <https://github.com/robot-agent-exploration/rax> — [代码仓归档](../repos/rax.md)
- **技能库：** <https://github.com/robot-agent-exploration/skills-lib>
- **Session logs：** <https://huggingface.co/datasets/robot-agent-exploration/robot-session-logs>
- **入库日期：** 2026-10-17；关联资源补录：2026-10-20
- **配套论文：** [RAX（官方 PDF）](../papers/rax_scaling_online_exploration.md)

## 一句话摘要

项目页介绍 RAX 如何通过扩展在线探索，让冻结的 coding agent 经机器人 API 解决任务、把成功经验编译为可执行程序，并在不同本体间复用带有前提、检查和失败模式的技能。

## 公开信息核查（2026-10-20 更新）

- **源码状态：已开源。** 官方 README 将 `main` 定义为分支索引；可用实现包括 RoboDojo / BEHAVIOR-1K、HumanoidArena（Unitree G1 + Dex3）、DexJoCo（Franka + Allegro），以及 Franka FR3、Rainbow RB-Y1 真机分支。
- **数据状态：部分公开。** 项目页提供真实机器人 session logs 数据集链接；它包含会话记录，不应默认视作通用训练集或覆盖全部实验的 benchmark 数据。
- **关联资源已分别归档：** [skills-lib 技能库](../repos/skills-lib.md) 按本体保存可执行技能；[session logs 数据集](../datasets/rax-robot-session-logs.md) 数据卡列出 5 段 Franka 与 7 段 RB-Y1 会话，并标注 CC BY 4.0。日志包含操作者指令、agent transcript、机器人状态和相机录制，二次使用前需检查隐私与再分发边界。
- **许可状态：已确认。** 官方仓库 `main` 含 `LICENSE` 和 `NOTICE`，GitHub 标注 Apache License 2.0；配套技能库许可需单独核对。
- **核查日期：** 2026-10-20（官方 README、main 分支 LICENSE / NOTICE）。
- **数字口径：** 未见场景成功率、仿真相对基线提升及编译执行加速分别来自不同评测；速度加速不代表机器人闭环频率提升。

## 关联资料

- 论文：[sources/papers/rax_scaling_online_exploration.md](../papers/rax_scaling_online_exploration.md)
- 代码：[sources/repos/rax.md](../repos/rax.md)
- wiki：[wiki/entities/rax-scaling-online-exploration.md](../../wiki/entities/rax-scaling-online-exploration.md)
