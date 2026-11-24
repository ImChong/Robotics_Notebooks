# AgentRoboCritic：面向机器人直接控制的动作级批评器（Understanding the Capability of Frontier Agents on Direct Robot Control） — papers 来源归档

- **类型：** paper
- **标题：** AgentRoboCritic: Understanding the Capability of Frontier Agents on Direct Robot Control
- **机构：** 卡内基梅隆大学、伊利诺伊大学厄巴纳-香槟分校、博世研究院 / 博世人工智能中心
- **原文：** https://arxiv.org/html/2611.26613
- **arXiv：** 2611.26613
- **发表日期：** 2026-11-19
- **项目主页：** https://adu2021.github.io/blog/AgentRoboCritic/
- **代码：** https://github.com/cxcscmu/AgentRoboCritic
- **开放状态：** MIT；公开评测 harness、EFE 数据采集转换、训练数据与微调 checkpoint；不能据此推断各闭源 actor 权重开放。
- **入库日期：** 2026-11-24
- **核对范围：** 论文正文与附录、批评器 codebook；CLI、runner.py 与 scoring.py 评测调用链。
- **通读：** 已完成
- **对应知识页：** [agentrobocritic](../../wiki/entities/paper-agentrobocritic.md)

## 资料要点

- actor 在提交 action chunk 时同时声明局部意图；评判器接收意图、命令和执行证据，输出意图是否成功、失败机制、抓取子类及原因。将动作级标签汇总成各失败类型与恢复统计。评测 Parquet 包含 id、messages、images、human_label；推理不读取人工标签。
- 作者在 12 个操作基准分析动作，约半数成功回合仍出现失败动作。400 条人工标注动作中有 264 个失败；微调模型失败模式准确率由 32.58%→74.24%（196/264），意图成功判断 76.25%（305/400），联合正确率 67.75%（271/400）。训练数据 12,844 条标注动作。
- critic 的标签误差、模型自偏好与局部意图的真实性会影响结论；动作级诊断并不自动修复策略或保证安全。抓取子类和解释不计入仓库所述主评分。

## 原文通读补充（2026-11-24）

### 方法与训练

- 直接控制指令是末端位姿、关节配置等数值目标，局部意图与动作在同一工具调用内提交。评判器读取全局任务、局部意图、动作前后观察和执行反馈，不读取最终回合结果；区分实现、失败、部分或未确认、无法核实及不适用。七类失败为抓取、对齐、干涉、任务效果、未完成运动、机制未决和错误局部选择；Choice 是相对局部意图，而非全局规划错误。
- 训练集 12844 条动作，400 条人工评测动作来自训练外的回合。评分中失败机制分母为 264 个真实失败动作；联合指标要求意图与机制同时匹配，不把解释和抓取子类纳入主分数。

### 对照、消融与边界

- 12 个任务套件中，49.8% 成功回合仍包含失败动作；1989 回合共 12163 个失败动作，10 动作以内恢复 38.9%，回合结束前恢复 57.7%。RoboDojo 占 9923 个失败动作，因此不能将样本加权统计误作套件均值。对齐恢复率 35.7%，低于抓取 63.7%。
- 400 条人工标注中 Astra 意图/机制/联合准确率 79.25%/80.68%/71.25%；去除局部意图后为 25.25%/25%/24.5%。微调 Muse 的对应值为 76.25%/74.24%/67.75%；400 条评测成本约 0.79 美元，对照 Astra 26.06 美元。附录 H：74 个匹配任务的决策延迟中位数 31.9 s 对 π0.5 的 0.11 s，直接控制尚不适合等同高频闭环。

## 公开资源

- [论文 HTML](https://arxiv.org/html/2611.26613)
- [arXiv](https://arxiv.org/abs/2611.26613)
- [官方项目页](https://adu2021.github.io/blog/AgentRoboCritic/)
- [官方源码](https://github.com/cxcscmu/AgentRoboCritic)
- [人工标注基准](https://huggingface.co/datasets/cx-cmu/AgentRoboCritic-Bench)
- [微调评判器](https://huggingface.co/cx-cmu/AgentRoboCritic-Muse-Glimmer-30B)
- [训练数据](https://huggingface.co/datasets/cx-cmu/AgentRoboCritic-Dataset)

## 对 wiki 的映射

- [agentrobocritic 独立实体](../../wiki/entities/paper-agentrobocritic.md)
- [关联归档](../sites/agentrobocritic.md)
- [关联归档](../repos/agentrobocritic.md)
