# Harness² 论文来源归档

## 元数据

- **题目：** Harness²: Recursive Agent Harnessing for an Open World
- **作者：** Ruiyao Xu, Yanfei Chen, Zhongying CuiZhu, Bhavana Dalvi Mishra, Yifei Ming, Han Yu, Rujun Han, Chen-Yu Lee, Tomas Pfister
- **单位：** Google Cloud AI Research；Northwestern University（作者单位，据论文）
- **预印本：** <https://arxiv.org/abs/2610.14224>（cs.AI；2026-10-09 提交）
- **HTML：** <https://arxiv.org/html/2610.14224v1>
- **PDF：** <https://arxiv.org/pdf/2610.14224>
- **论文许可：** CC BY 4.0（arXiv 页面所示）
- **项目页：** <https://harness2.github.io/>
- **代码：** <https://github.com/google-research/harness2>
- **入库日期：** 2026-10-22

## 论文要解决的问题

开放世界中的 agent 会遇到 harness 设计时未见过的新任务，通常没有可直接用来指导在线改动的标准答案或可靠验证器。只针对当前任务改 harness，也难以把“怎样诊断与修复”的经验迁移到后续任务。

## 方法要点（据论文与官方实现）

- **冻结模型，编辑运行时 harness：** agent 写作业时由模型和 harness 共同构成；研究中模型权重保持固定，改动对象是周围的运行文件。
- **八类可编辑组件：** system prompt、guardrails、memory、tool guidance、subagents、skills、scripts、plugins。
- **任务级递归（Propose–Probe–Compose）：** 先运行基线 harness 取得参考轨迹；分别提出面向跨领域通用缺陷的 invariant 候选与面向当前领域的 domain-specific 候选；在同一任务、匹配配置下运行候选，再依据轨迹和交付物之间的行为差异合并有依据的改动，生成并运行后继 harness。该过程不靠任务标准答案判断候选是否正确。
- **两种任务级扩展：** parallel 并行探测多个候选并组合；sequential 将前一步 harness 与证据带入下一步。
- **跨任务递归：** 每个任务结束后反思完整改进记录，把被采用的 harness 和有证据支持的经验写入 improvement library。后续任务仍从公共初始 harness 开始，再按需检索和适配经验，而非机械继承上一个任务的专用 harness。

## 报告结果（论文主表，均值为三次运行）

- WorkBuddy-Bench：GLM-5 + OpenCode 从 base harness 的 64.2 提升到 Harness² 的 75.7（+11.5）；Gemini 3.5 Flash + OpenCode 为 72.4 → 77.5（+5.1）。
- 跨配置观察：GLM-5 + Harness² 的 75.7 高于 Gemini 3.5 Flash + base harness 的 72.4；这是两个配置的分数对照，不是同模型消融。
- 论文摘要报告单步最高提升：WorkBuddy-Bench +11.5 分、JobBench +4.5 分、LAB Partial pass rate +11.8 个百分点。LAB 的 Partial 是按评分标准条目的通过率。
- 主表另外报告标准差；更高的递归预算可带来额外收益，但不同任务、模型和评估指标的幅度不同。

## 开源状态与复现边界

官方仓库提供 Apache-2.0 许可的 Harness² Python 包、CLI、确定性本地 demo、benchmark 安装与运行脚本。官方 README 的最小 demo 不需模型凭证；模型驱动 benchmark 需配置执行器、改进器与评测器，且会产生 API 调用成本。上游及 benchmark 数据依赖保留其各自许可。

复现时应限制执行器的网络访问：官方实验指南指出公开 benchmark 的 rubric / 评测信息可能被 agent 访问，可能造成评测泄漏；应先核对实验所用网络策略。该实现面向 WorkBuddy、JobBench 与法律 benchmark，并非机器人控制框架，不能从这些结果推出真机机器人任务收益。

## Robotics_Notebooks 映射

- 实体页：[Harness²](../../wiki/entities/paper-harness2.md)
- 概念页：[递归自改进](../../wiki/concepts/recursive-self-improvement.md)

## 一手来源

- arXiv 论文及 HTML 正文
- [官方项目页](https://harness2.github.io/)
- [官方代码仓库](https://github.com/google-research/harness2)
