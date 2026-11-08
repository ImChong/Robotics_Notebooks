# RAX: Scaling Online Exploration for Robot Agents（官方论文）

> 来源归档（ingest）

- **标题：** RAX: Scaling Online Exploration for Robot Agents
- **类型：** paper / preprint
- **作者：** Runhan Huang、Xiangcheng Zhang、Haoyu Chen、Yitong Li、Yen-Jen Wang、Bohan Lyu、Haotian Ye、Chuye Hong、Yixuan Wang、Chloe Su、Haoyu Xiong、Yangzhen Wu、Enze Xie、Travis Manderson、Stefano Ermon、Pieter Abbeel、Antonio Torralba、Dawn Song、Weirui Ye、Yilun Du
- **机构：** Harvard University、NVIDIA、UC Berkeley、MIT、Stanford University
- **论文：** <https://robot-agent-exploration.com/rax.pdf>
- **arXiv：** <https://arxiv.org/abs/2610.22496>
- **项目页：** <https://robot-agent-exploration.com/> — [项目页归档](../sites/rax-project.md)
- **代码：** <https://github.com/robot-agent-exploration/rax> — [仓库归档](../repos/rax.md)
- **入库日期：** 2026-10-17；关联 arXiv 编号补录：2026-10-20；全文复核：2026-11-08
- **核对范围：** [arXiv HTML v1](https://arxiv.org/html/2610.22496v1) 论文全文（正文 §1–7 与附录 §8–15：方法、实验协议、消融、真机案例与局限）；另对照官方 `real/franka` README、`scripts/configure_session.py`、`scripts/start_agent.sh`、`harness/claude_cli/mcp_bridge.py`、`services/controller/frontend.py` 的公开实现。
- **通读：** 已完成

## 核心内容

RAX（Robot Agent eXploration）研究冻结的通用 coding agent 如何仅通过机器人感知与控制接口在线探索任务、从交互反馈中整理决策经验，并将有效流程编译成无需 agent 在线推理即可运行的模块化程序。真机任务无需动作示范；部分仿真实验使用示范图像或动作片段（见下方评测边界）；经验主要保存在 agent 工作区的笔记、程序与已验证技能中。

论文报告：仿真中在线探索从 4 个扩展到 128 个 episode 时，未见场景平均成功率由 25% 升至 79%；仿真任务平均较顶级基线高 37.4%。系统覆盖单臂、双臂、灵巧手、移动与人形平台，并含真实机器人任务。论文表 9 中，空气炸锅与 Bandu tower 的推理耗时分别减少约 241× 与 103×；该数字不含机器人调用时间。

## 阅读边界

- 论文版本：[arXiv HTML v1](https://arxiv.org/html/2610.22496v1)，2026-10-15；已按本文的「核对范围」通读 v1 全文并核对方法、评测协议、消融与边界；论文实验结果未由本次独立复现。
- 表 9 的 72.3 分钟→18 秒和 17.2 分钟→10 秒是机器人调用之外的推理墙钟时间，机器人时间分别为 2.9→2.5 分钟及 9.3→10.9 分钟，不是任务端到端速度。
- BEHAVIOR-1K 使用无动作或状态的成功示范关键帧；HumanoidArena 主要设置使用每任务 3 条遥操作示范且可复用动作片段。表 10 去除动作后仍保留示范信息，平均 81.9%，主要设置为 89.8%，且探索预算不同。
- RoboDojo 按基准原生评分，允许部分得分。自动程序和 agent-in-the-loop 评估须区分；前者运行时无 agent 修改或重试，但按协议可调用视觉语言模型。
- 附录 §14.2：空气炸锅任务曾碰撞相机，部分运行需要恢复；另有 4 次无需人工接管成功。附录 §15：所有实验合计 API 费用超过 10 万美元，并非单任务成本。

## 全文核对补充（2026-11-08）

- **领先幅度单位（附录 §13）：** 五组仿真基准上，相对各自最强基线的平均差值为 **37.4 个百分点**，逐项为 RoboDojo +53.4、BEHAVIOR-1K +38.3、HumanoidArena +7.9、DexJoCo 五项双臂任务 +45.5、LIBERO-Pro +42.1；这不是每个任务均相对提升 37.4%，也不等于在同一指标上对同一模型的统一比较。
- **评估协议（附录 §10）：** 区分编译程序自动评估与 agent-in-the-loop 评估；BEHAVIOR-1K 中部分任务使用后者。HumanoidArena 主实验使用每任务 3 条遥操作示范，而无动作消融仍有无动作的示范信息。
- **真实任务执行边界（附录 §14.4、§14.6）：** 高精度插销任务由 agent 通过工具交互，未进行无人值守编译程序测试；长时程移动操作案例含操作者介入。真机演示不能一概视为全程独立运行。
- **官方源码入口：** `real/franka` 分支从 `scripts/configure_session.py` 创建会话配置，再通过 `scripts/start_agent.sh` 调用 `harness.claude_cli.launcher`。agent 的 MCP 请求通过 `harness/claude_cli/mcp_bridge.py` 转发到 `services/controller/frontend.py` 持有的单一 robot service；实际机器人控制与安全检查由 backend 承担。详情见[仓库归档](../repos/rax.md)。

## 对 wiki 的映射

- 统一编译到 [RAX 实体页](../../wiki/entities/rax-scaling-online-exploration.md)
- 比较 coding agent 工作区演进路线时，关联 [ASENA](../../wiki/entities/paper-asena-self-evolving-agents.md) 与 [EmbodiedEvo](../../wiki/entities/paper-embodied-evo-coding-agent.md)
