# Astranomy: GPT-6 Astra 作为原生机器人策略（NVIDIA SRL 技术报告）

> 来源归档；初次核查：2026-10-15；原始技术报告及新增源码复核：2026-11-21。

- **标题：** Astranomy: An Empirical Study of GPT-6 Astra as a Native Robot Policy
- **类型：** site / technical report
- **项目页：** <https://research.nvidia.com/labs/srl/projects/astranomy/>
- **作者：** Jenai Xuning Yang、Ashna Khetan、Shun Zhang
- **机构：** NVIDIA Seattle Robotics Lab（SRL）
- **发布日期：** 2026-10-14
- **arXiv：** 项目页标注 coming soon
- **代码：** <https://github.com/NVlabs/RoboLab/tree/main/policies/ai_native>（2026-11-21 官方项目页 Code 已指向该目录；含 Direct-API、Agentic、Orchestrated 三种实现）
- **数据：** 使用 RoboLab-120 仿真评测；未见独立 Astranomy 数据集下载入口
- **核对范围：** 通读 NVIDIA 官方在线技术报告的 Overview、§1 引言、§2 设计/三策略与消融、§3 各项结果、§4 延迟/费用、§5–7 结论/失败案例/限制及重要结果表；逐项核对官方 Code 入口，阅读 RoboLab 的 policies/ai_native/README.md、DESIGN.md、三种策略 README 和 run.py 入口。
- **通读：** 已完成
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

- **历史状态（2026-10-15）：** 初次归档时，项目页将 Code、arXiv 标为 coming soon；该判断有日期边界，不应覆盖后续发布。
- **复核状态（2026-11-21）：** 官方报告页 Code 已指向 RoboLab [policies/ai_native](https://github.com/NVlabs/RoboLab/tree/main/policies/ai_native)；目录现含 Direct-API、Agentic、Orchestrated 三类实现及运行说明。环境与任务仍依赖 Isaac Lab 等安装和模型访问授权；arXiv 在本次复核仍为 coming soon。
- **关键复现差异：** agentic/README.md 说明公开的参考实验曾向 Agentic 策略暴露原生进度与成功判据元数据；现版代码移除这些输入、更新 reset/seed 协议，**论文分数尚未在新协议下重新评测**。orchestrated/README.md 也指出公开实现部分分辨率、seed、超时和 token 配置与报告文字不完全相同；不得把代码开放等同于原始 74.8% 的逐字节复现。
- **源码入口：** [direct_api/run.py](https://github.com/NVlabs/RoboLab/blob/main/policies/ai_native/direct_api/run.py)、[agentic/run.py](https://github.com/NVlabs/RoboLab/blob/main/policies/ai_native/agentic/run.py)、[orchestrated/run.py](https://github.com/NVlabs/RoboLab/blob/main/policies/ai_native/orchestrated/run.py)；公共契约见 [DESIGN.md](https://github.com/NVlabs/RoboLab/blob/main/policies/ai_native/DESIGN.md)。

### 来源说明

用户提供的 BibTeX 将文献标为 Blog post，项目页则标注为 TECH REPORT，且页面标题为 “An empirical study…”。本归档以 NVIDIA 项目页展示的标题、日期与报告类型为准。


### 原文补充：消融与失效机制（复核 §3–7）

- **动作空间（30-task 子集）：** Direct 的绝对 EEF 成功率 72.7%，关节位置 48.7%，B-spline 37.3%；Agentic 在 EEF（86.0%）和关节位置（84.0%）间差距缩小，不能把不同动作接口分数当作纯模型差异。
- **工具与规划：** Agentic 的可选工具关闭/开启对应 86.0%/88.0%；显式计划、无显式计划、抑制计划分别为 86.0%/90.0%/81.0%。报告没有证明可选工具或显式规划能稳定带来优势。
- **动作纠偏：** Orchestrated 的介入覆盖 14.5% 仿真步；介入块按主因分类，抓取问题 64.4%，目标/规划问题 20.2%。这是介入原因分布，不是全部任务的失败比例。
- **运行时成本：** Direct 成功 episode 的 API 调用均值 18.56 秒，估算墙钟时长约 97.0% 在推理；图表展示的 API token/费用平均值基于**成功 episode**与 2026-10-01 计价，不应泛化为任意失败回合成本。
- **失败与局限：** 抓取对位、夹爪受阻、错误目标、落放不到位、恢复失败；120 项抓放基准每任务 5 个 seed，作者未证明小幅消融差别的统计显著性；推理时暂停仿真，不覆盖快速变化场景与真机实时控制。

## 对 wiki 的映射

- 主实体：[Astranomy：GPT-6 Astra 原生机器人策略研究](../../wiki/entities/astranomy.md)
- 相关评测：[GPT 6 Astra as an Embodied Policy](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md)
- 使用的评测基准：[RoboLab](../../wiki/entities/robolab.md)
