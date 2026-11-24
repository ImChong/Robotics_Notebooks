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
- **核对范围：** 已打开 HTML 全文及章节目录、核对项目页和主要指标；尚未逐节通读正文和附录。
- **通读：** 未完成
- **对应知识页：** [agentrobocritic](../../wiki/entities/paper-agentrobocritic.md)

## 资料要点

- actor 在提交 action chunk 时同时声明局部意图；评判器接收意图、命令和执行证据，输出意图是否成功、失败机制、抓取子类及原因。将动作级标签汇总成各失败类型与恢复统计。评测 Parquet 包含 id、messages、images、human_label；推理不读取人工标签。
- 作者在 12 个操作基准分析动作，约半数成功回合仍出现失败动作。400 条人工标注动作中有 264 个失败；微调模型失败模式准确率由 32.58%→74.24%（196/264），意图成功判断 76.25%（305/400），联合正确率 67.75%（271/400）。训练数据 12,844 条标注动作。
- critic 的标签误差、模型自偏好与局部意图的真实性会影响结论；动作级诊断并不自动修复策略或保证安全。抓取子类和解释不计入仓库所述主评分。

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
