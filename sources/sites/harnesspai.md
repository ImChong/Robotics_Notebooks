# HarnessPAI 官方项目页

> 官方项目页来源归档（核验日期：2026-10-08）。

- **URL：** <https://darwin-agent.github.io/HarnessPAI/>
- **维护团队：** Darwin Agent Team
- **关联论文：** [HarnessPAI: An Evolving Harness for Physical AI（arXiv:2609.29166）](../papers/harnesspai_arxiv_2609_29166.md)
- **关联 GitHub 仓库：** [Darwin-Agent/HarnessPAI](../repos/harnesspai.md)
- **项目节点：** [HarnessPAI](../../wiki/entities/paper-harnesspai.md)

## 项目页描述

HarnessPAI 是模型、机器人 embodiment 无关的 Physical AI harness，以可执行代码协调感知、几何推理、动作原语与执行检查。它区分两个时间尺度：单次 rollout 内运行固定任务程序；rollout 之间利用执行反馈诊断失败、修改候选程序并验证。结构化技能记忆保留失败—修复经验，成功轨迹可以进一步作为下游模型训练数据。

固定程序并非“无反馈”的盲执行：项目页说明程序可使用当前观测、反馈控制、预定义检查和恢复逻辑。选定程序后不进行在线高层 LLM deliberation，但感知、动作推理、仿真/硬件和程序开发仍有成本。

## 官方页面列出的结果

- LIBERO-PRO（Swap / Task 子集）：π₀.₅-LIBERO 34.9% → HarnessPAI 96.5%，+61.6 个百分点。
- RoboCasa Target50 Atomic-Seen（18 项任务）：WorldDreamer 65.0% → HarnessPAI 92.2%，+27.2 个百分点。
- robosuite 七项操作任务：报告相对 ASPIRE +15.3 个百分点；基线来自先前工作，而非同后端受控消融。
- LIBERO 三套件：π₀.₅-LIBERO 96.9% → HarnessPAI 98.1%，+1.2 个百分点。
- 页面另列 BEHAVIOR-1K、VacuSim、MicroDuck 结果，指标分别涉及任务完成、清洁覆盖率与行走横向误差，不可合并成一个总成功率。

项目页说明 LIBERO 与 LIBERO-PRO 使用 50 seeds（其中 15 用于程序演化、35 为额外评估）；RoboCasa 使用 20 seeds。迁移实验和基于成功轨迹的 π₀.₅ 后训练结果是额外实验，不应与冻结动作后端主结果混淆。

## 官方发布状态

页面链接到 GitHub 项目仓库；截至核验日，仓库 README 显示项目主页与演示、论文 PDF 可用，但研究代码仍在整理、尚未公开，安装/复现说明也未提供。因此当前公开页面可用于理解论文主张和查看演示，不构成可运行软件发布。

## 原始出处

- 项目页：<https://darwin-agent.github.io/HarnessPAI/>
- 论文：<https://arxiv.org/abs/2609.29166>
- 官方仓库发布状态：<https://github.com/Darwin-Agent/HarnessPAI>
