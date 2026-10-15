# Astranomy: GPT-6 Astra 作为原生机器人策略（NVIDIA SRL 技术报告）

> 来源归档；核查日期：2026-10-15。

- **标题：** Astranomy: An Empirical Study of GPT-6 Astra as a Native Robot Policy
- **类型：** site / technical report
- **项目页：** <https://research.nvidia.com/labs/srl/projects/astranomy/>
- **作者：** Jenai Xuning Yang、Ashna Khetan、Shun Zhang
- **机构：** NVIDIA Seattle Robotics Lab（SRL）
- **发布日期：** 2026-10-14
- **arXiv：** 项目页标注 coming soon
- **代码：** 项目页标注 coming soon；截至核查日未提供可访问仓库
- **数据：** 使用 RoboLab-120 仿真评测；未见独立数据集下载入口
- **一句话说明：** 在 RoboLab-120 上比较 GPT-6 Astra 直接生成动作、通过 Codex 持续会话进行 agentic 控制，以及监督 Cosmos3-Nano-Policy 的三种策略形态，分析动作接口、上下文、规划、延迟与 token 成本。

## 核心摘录（归纳）

### 研究问题与设置

报告研究通用多模态模型能否在不进行机器人策略微调的情况下，按推理时提示配置成为机器人策略。评测在 RoboLab 仿真中以 DROID 机器人形态执行语言条件抓放；控制频率为 15 Hz，仿真在模型推理期间暂停。每个任务运行 5 个固定 seed episode，并另用 30 个任务子集做消融。

### 三种策略形态

| 形态 | Astra 的角色 | 关键特点 |
|------|--------------|----------|
| Direct Policy | 直接把观测映射为动作 | 每次 API 调用生成动作块，不保留调用间上下文 |
| Agentic Policy | 在 Codex 持续会话中控制 | 保留交互历史，可使用图像检查、计算等工具 |
| Orchestrated Policy | 审查 Cosmos3-Nano-Policy 的动作提议 | 接受 15 步候选，或干预并发出较短动作段 |

### 主要结果

- RoboLab-120 上 Agentic Policy 为 **74.8% 成功率 / 83.6 平均完成分**；Direct Policy 为 **66.3% / 76.7**；Orchestrated Policy 为 **66.5% / 79.1**。
- 报告中的 Direct Policy 平均推理调用耗时为 **18.56 秒**；Agentic Policy 为 **13.37 秒**。这远慢于 15 Hz 控制周期，说明仿真暂停等待推理的结果不能直接视作可实时部署。
- Agentic Policy 每 episode 平均约 **30.9M tokens / 36.40 美元**，Direct Policy 约 **160k tokens / 1.96 美元**。更高成功率伴随显著推理成本。
- 相比报告展示的 RoboLab-120 最强既有策略 Lakeside（51.0% SR），Agentic Policy 高 **23.8 个百分点**。该比较只适用于报告使用的评测设置。
- 动作表示、机器人几何信息、观察历史、任务措辞都会影响表现；因此结果衡量的是模型与提示/运行框架共同组成的策略系统。

### 开源与复现状态

截至 2026-10-15，NVIDIA 项目页将代码和 arXiv 标为 **coming soon**，没有提供可运行仓库或可下载新数据集。本报告使用的 RoboLab 基准本身有独立的[项目页](https://research.nvidia.com/labs/srl/projects/robolab/)和[开源仓库](https://github.com/NVlabs/RoboLab)，但这不等于 Astranomy 的评测实现已开放。本文结果应标记为已发布报告、代码待发布。

### 来源说明

用户提供的 BibTeX 将文献标为 Blog post，项目页则标注为 TECH REPORT，且页面标题为 “An empirical study…”。本归档以 NVIDIA 项目页展示的标题、日期与报告类型为准。

## 对 wiki 的映射

- 主实体：[Astranomy：GPT-6 Astra 原生机器人策略研究](../../wiki/entities/astranomy.md)
- 相关评测：[GPT 6 Astra as an Embodied Policy](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md)
- 使用的评测基准：[RoboLab](../../wiki/entities/robolab.md)
