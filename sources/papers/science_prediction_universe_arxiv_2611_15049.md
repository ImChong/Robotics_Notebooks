# Science Prediction Universe（SPU）：隐藏湿实验结果预测基准

> 一手论文归档；以下为原文核读后提炼，不是论文全文转载。

- **论文标题：** Science Prediction Universe / 科学预测宇宙
- **类型：** paper / benchmark / autonomous scientific agent / AI for Science
- **主论文：** [arXiv:2611.15049v1](https://arxiv.org/abs/2611.15049) · [HTML 全文](https://arxiv.org/html/2611.15049v1) · [PDF](https://arxiv.org/pdf/2611.15049)
- **首次提交：** 2026-11-11
- **作者（项目组）：** Zengji Tu、Xiang Liu、Jiarui Sun、Ruoyu Wu、Zhihui Qi、Chengda Wu、Yutao Zhou、Bo Yang、Ziyan Zhang、Jialong Wu、Liang Chen、Kuan Li；另有顾问委员会及 wet-lab 合作者
- **牵头机构：** UniPat AI；顾问与实验合作者来自 Stanford、Cornell、Harvard、PKU、HKUST 等（完整名单见原文附录 A）
- **项目：** https://spu.unipat.ai/ （官方论文与仓库指向该链接；本次网页直读失败）
- **代码/任务贡献仓库：** https://github.com/UniPat-AI/SPU
- **公开数据与 grader：** https://huggingface.co/datasets/UnipatAI/SPU
- **入库日期：** 2026-11-14
- **核对范围：** arXiv HTML v1 的摘要、§1–6（任务定义、筛选、评分、公开及私有实验、统计对照、轨迹分析、相关工作、结论）、附录 A/B（作者与限制）、附录 D/E/F 的实验格式、补充统计与典型运行；对照官方 GitHub README 和 Hugging Face 数据卡的 Release / Usage / Scoring / Known issues。
- **通读：** 已完成

## 研究问题与定位

现有 agent benchmark 多问已有答案，或按软件测试验证交付物；SPU 问的是：**在不知道真实湿实验测量值时，智能体能否根据实验定义预判不同候选条件的优劣排序？** 不是合成路径规划系统，也不是机器人基准。

输入是实验共享协议、材料/候选条件、终点定义与结构/序列等公共文件；输出是每个 comparison group 的从低到高严格排序（`rankings.json`），不要求猜测测量绝对值。真实值在隔离的 grader 中，不提供给 agent。

## 任务建设与数据口径

- **116 项任务：** 100 项公开任务来自 2025-01 至 2026-08 的 85 篇 Nature Portfolio 论文（11 本期刊），另有 16 项未公开 wet-lab 任务；覆盖化学、材料、生物、医学四领域。全体任务各学科数量 42/31/31/12；**公开部分**为 39/29/20/12，不能混为同一统计口径。
- **517 个比较组、4,207 个实验：** 每任务 4–362 个实验（中位数 16）；公开部分 3,947 个实验、489 组。实验终点例子包括产率、离子电导率、材料强度、荧光与 RNA-seq 计数。
- **公开任务四步：** (1) 专家精选近期文献且要求完整量化；(2) 剔除单参数严格单调或数值差异过小的简单/噪声组；(3) 按共享模板匿名整理并独立复审；(4) 校准难度，从 29 子领域至少各取 1 项，再择优选出 71 项。
- **16 项私有任务：** 来自四组未发表湿实验，覆盖 VHH 纳米抗体变体、反应条件、电解液、Magrolimab 抗体；共 260 实验、28 组；私有真实值未公开且不用于挑题模型评分。

## 评测协议与数学指标

- 一项任务按各比较组真实名次 `r` 与预测名次 `r_hat` 计算 **Spearman 相关系数** `rho_g`；任务分数 `S_t = Σ_g w_g rho_g`，权重在建题时固定且和为 1（100 项公开任务中 97 项等权）。
- 完美名次相关接近 +1，随机合法排序的期望为 0；33 项公开任务存在真实值并列，理论上限可能略低于 1。缺文件、全文件格式错误得 -1；单组错误只对该组施加 -1。
- 对每个任务随机采样 4,000 个合法排序，用分布的 95% 分位数设 chance threshold；不应仅凭单任务高相关就认定显著优于随机。
- 原始运行：每配置 8 CPU 核、16 GiB、NVIDIA H20 GPU、最多 8 h；可用网络与计算工具，但数据结果与评分器隔离，并设来源泄漏审计政策。

## 实验结果及解释边界

- **13 种配置 / 12 个模型 / 2 种 harness**（Codex 与 Claude Code），公开部分官方总运行 1,300 次；论文对 1,155 条有分数公开轨迹及 227,103 次工具调用做分析。
- 公开部分**最佳 task-adjusted Spearman 均值 0.478**（Opus 5.5，但仅 44/100 项有可评分输出），Astra Medium 为 0.431（100/100 项）；不同模型拒绝率不同，**必须同时报告覆盖率**，不可只看均值排名。
- 按论文配对检验，配置大致分高低两层；层内的细微排序尚不能在单次运行条件下确定。
- 同一模型 Astra Max 相比 Medium 的中位运行时长约 4.8 倍，但未证明预测准确率随之提高。agent 之间名次相关平均约 0.52，与真实测量约 0.33；多数失败在于错误选择控制目标量的科学机制，非工具不会调用。
- 私有任务中 12/13 种配置均值经作者检验显著优于随机，但 16 项私有任务样本少、学科分布有限；与公开部分应分开解释。

## 开源边界（核对于 2026-11-14）

- [UniPat-AI/SPU](https://github.com/UniPat-AI/SPU) **已公开**任务投稿、审阅、质量校验与文档代码，许可 Apache-2.0；**不是**论文中所有 agent/harness 的完整发行版。
- [UnipatAI/SPU](https://huggingface.co/datasets/UnipatAI/SPU) **已公开** 100 项 Harbor 任务、`instruction.md`/`task.toml`、公共输入、`tests/` 评分器与实验测量、`solution/` oracle、`SOURCES.json`；数据许可 CC BY 4.0。
- 官网官方链接存在但本次无法直接读取网页；不把其未核实功能写为已验证。
- 公开数据含答案，**不能当 held-out 测试集**；公开版本**不含**论文原始审计网络工具与来源 ID 筛查器。独立重跑分数不可不加限制地与论文论文表格对比。
- 官方给出的自检路径是 `hf download UnipatAI/SPU --repo-type dataset --local-dir SPU`，然后在任务目录用 `harbor run ... -a oracle -e docker`（需 Harbor 0.25.0、Python 3.12+、Docker，oracle 自检不需要 GPU）。这验证发布的评分包，**不等于**复现 13 个 agent 评测配置。

## 局限与可迁移认识

1. 公开任务本质是对已发表结果的回顾性预测；原文用匿名化、检索审计与 16 项未公开实验降低泄漏风险，但不能由此断言完全无污染。
2. 平均分混合模型、推理强度、harness、拒绝率与不同任务，不能当单一模型固有科学能力分数。
3. 真实测量可能受实验噪声影响，按秩判分避免估计绝对数值，却不保证每对接近值都可靠。
4. 对机器人研究的价值是**评测设计类比**：把「会写代码/会用工具」与「能否预判物理后果」分开验收；论文并未评测任何机器人模型、控制策略或真机任务。

## 对 wiki 的映射

- [SPU 独立论文详情](../../wiki/entities/paper-science-prediction-universe.md)
- [AI Agent 评测](../../wiki/concepts/ai-agent-evaluation.md) — outcome-first、隔离评分、反泄漏协议
- [AI Auto-Research](../../wiki/concepts/ai-auto-research.md) — 科研执行与预测判断的边界
- [官网来源](../sites/science-prediction-universe.md) · [仓库来源](../repos/unipat-ai-spu.md)

## 官方参考

- [论文 HTML v1](https://arxiv.org/html/2611.15049v1)（§2–5 与附录 B/E）
- [论文 PDF](https://arxiv.org/pdf/2611.15049)
- [GitHub README](https://github.com/UniPat-AI/SPU)
- [Hugging Face 数据卡](https://huggingface.co/datasets/UnipatAI/SPU)
