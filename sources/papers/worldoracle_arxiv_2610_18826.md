# WorldOracle: Benchmarking Action-Conditioned World Models against Real-World Executions

> 来源归档（paper）

- **类型：** paper / benchmark
- **作者：** Haoyi Duan et al.（共 16 位作者）
- **arXiv：** <https://arxiv.org/abs/2610.18826>（v1，2026-10-13）
- **项目页：** <https://worldoracle-benchmark.github.io/>
- **代码：** <https://github.com/AgibotTech/WorldOracle>
- **数据入口：** <https://huggingface.co/WorldOracle>（项目页资源链接）
- **入库日期：** 2026-10-16
- **一句话说明：** 给定同一起始观测与低层动作序列，将动作条件世界模型 rollout 与配对真机执行对照，评测预测是否复现真实物理后果和任务状态转移。

## 核心摘录（MVP）

### 1) 从视频观感转向执行一致性
- **链接：** <https://arxiv.org/abs/2610.18826>
- **摘录要点：** WorldOracle 针对动作条件世界模型（ACWM）建立真实世界基准。模型获得与现实执行相同的初始观测和低层动作序列后，基准检查预测 rollout 是否复现真实物理后果与任务状态变化；评测对象因此是动作条件预测与真实执行的一致性，而不只是视频逼真度。
- **对 wiki 的映射：**
  - [WorldOracle 实体页](../../wiki/entities/paper-worldoracle.md)
  - [Generative World Models](../../wiki/methods/generative-world-models.md)

### 2) 四个互补维度与两种汇总分
- **链接：** <https://arxiv.org/abs/2610.18826>
- **摘录要点：** 评测包含物理状态正确性、交互空间对齐、关键交互区域保真度、多视角动作一致性。官方实现将前三项等权平均为 WorldOracle-SV；支持同步多视角预测时，将四项等权平均为 WorldOracle-MV。各维度保留终态、交互位置、接触区域和视角一致性等不同失效信号。
- **对 wiki 的映射：**
  - [WorldOracle 实体页](../../wiki/entities/paper-worldoracle.md)
  - [EWMBench](../../wiki/entities/ewmbench.md)

### 3) 数据覆盖多本体、长程片段及失败执行
- **链接：** <https://arxiv.org/abs/2610.18826>
- **摘录要点：** 论文摘要称真实世界训练与评测套件覆盖不同交互动力学和本体，评测集由长程多段 episode 构成，并包含成功和失败执行。官方页列出 AgiBot G2、YAM、human-hand，以及 human-operated 与 policy-rollout 两类动作分布。
- **对 wiki 的映射：**
  - [WorldOracle 实体页](../../wiki/entities/paper-worldoracle.md)
  - [EWMBench](../../wiki/entities/ewmbench.md)

## 开源状态

- **代码：已公开。** 项目页链接 AgibotTech 官方仓库；README 提供 evaluate.py 入口和运行说明，仓库为 Apache-2.0。
- **数据：** 项目页链接 WorldOracle 的 Hugging Face 页面。实现文档区分模型可见的 inference package 与评测用 ground-truth package；未来真值不放在模型输入包内。
- **排行榜：** 项目页提供 live leaderboard 与提交指南；榜单应按评测器版本读取，不与论文数值直接混用。
