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
- **入库日期：** 2026-10-17；关联 arXiv 编号补录：2026-10-20

## 核心内容

RAX（Robot Agent eXploration）研究冻结的通用 coding agent 如何仅通过机器人感知与控制接口在线探索任务、从交互反馈中整理决策经验，并将有效流程编译成无需 agent 在线推理即可运行的模块化程序。方法不依赖动作示范数据、手工动作原语或价值函数引导；经验主要保存在 agent 工作区的笔记、程序与已验证技能中。

论文报告：仿真中在线探索从 4 个扩展到 128 个 episode 时，未见场景平均成功率由 25% 升至 79%；仿真任务平均较顶级基线高 37.4%。系统覆盖单臂、双臂、灵巧手、移动与人形平台，并含真实机器人任务。编译程序在空气炸锅与 Bandu tower 任务上分别比 agent 执行快 241× 与 103×。

## 阅读边界

以上数值来自论文摘要与项目页，应按各自 benchmark、任务和评估协议理解；速度比比较的是 agent 在线解决与编译后程序执行。真机演示不等同于跨场景统计可靠性。该论文当前通过官方项目站点提供 PDF，项目站点未标注 arXiv 编号。

## 对 wiki 的映射

- 统一编译到 [RAX 实体页](../../wiki/entities/rax-scaling-online-exploration.md)
- 比较 coding agent 工作区演进路线时，关联 [ASENA](../../wiki/entities/paper-asena-self-evolving-agents.md) 与 [EmbodiedEvo](../../wiki/entities/paper-embodied-evo-coding-agent.md)
