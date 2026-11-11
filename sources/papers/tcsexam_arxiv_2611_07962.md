# TCSExam 论文 v1 原文归档与技术核查

## 元数据

- **英文题目：** TCSExam: Evaluating Frontier LLMs on the Full Range of TCS Research
- **中文题目：** TCSExam：评测前沿大语言模型在理论计算机科学研究全谱系中的能力
- **作者：** Yifan Luo、Zhengying Liu、Yang Yuan、Zhilin Yang、Andrew Chi-Chih Yao
- **机构：** 清华大学、Moonshot AI、上海期智研究院（参见论文首页署名）
- **论文：** https://arxiv.org/abs/2611.07962v1
- **全文（v1）：** https://arxiv.org/html/2611.07962v1
- **项目：** https://moonshotai.github.io/TCSExam/
- **代码：** https://github.com/MoonshotAI/TCSExam
- **版本：** arXiv v1（2026-11-05）；本页仅整理 v1 的论文结果，不混入项目页后续更新
- **入库日期：** 2026-11-11
- **对 wiki 的映射：** [TCSExam 论文实体](../../wiki/entities/paper-tcsexam.md)

## 原文阅读记录

- **核对范围：** 阅读 arXiv HTML v1 全文正文 §1–6、主要表格与附录 A–H（任务构造、四个轨道、形式化和程序验收规则、相对评分、Lite 选题、失败与案例分析、开放猜想解释）；核对 Full/Lite 主表、对照设置及局限。大型实例和参考文献条目只抽查，未独立复算论文各项实验分数。交叉检查作者官方 GitHub README、docs/scoring.md、docs/getting-started.md 与 CLI 源码的协议/评分边界。
- **通读：** 已完成
- **复现：** 未运行模型、Docker/Harbor 任务或验证器；原文事实、程序静态核对和独立复现须区分。

## 要解决的问题

SWE 或短题数学 benchmark 无法同时覆盖理论计算机科学研究中的机器可检验证明、算法设计、构造性成果和未解决猜想。TCSExam 的目标是设计包含可执行或形式化验收的研究任务，衡量模型与 agent harness 组合的能力，不是普通数学问答测试。

## 任务结构与验证边界（论文 §2–§3）

| 赛道 | Full | Lite | 主要提交/验收方式 |
|---|---:|---:|---|
| Theorem proving | 20 | 5 | 提交 Lean 4 正式证明；核验完整无占位证明 |
| Solver engineering | 30 | 5 | 实现算法或求解器，隐藏实例上执行与打分；分 Online/Offline 开发协议 |
| Constructions | 20 | 5 | 交付离散构造、证据或改进的数值界，独立检验 |
| Open conjectures | 30 | 5 | 正式完整证明或反驳；进度、局部引理本身不算解题成功 |

- **任务总量：** Full 100，Lite 从 Full 选 20；Solver 每项具有单独的 Online、Offline 包，不是额外的 30 个研究题。
- **Solver 协议：** Full 主实验的 Online 在开发中提供受控 judging 反馈；Lite 主实验用 Offline 的公开自测，隐藏评分在提交后执行。切换协议须分开报告，不能混算。
- **安全/防泄漏：** 题目环境和验证器隔离；隐藏评测材料及评分基线应留在 verifier 一侧，不是把秘密标签交给模型。
- **论文与公开仓库的关系：** 仓库提供任务、工具、评分参考和文档；论文模型成绩不等于用户在任意 API/harness/预算下可复现的分数。

## 分数定义（论文 §3；官方 scoring.py）

TCSExam Index = 30 × theorem success + 30 × fixed-K3-relative Solver score + 30 × original Construction score + 10 × complete Open-conjecture success。

先在任务内平均纳入的尝试，再等权平均该赛道任务，最后按 30/30/30/10 加权。Solver 对固定 Kimi K3 参考归一化；Construction 可能优于历史锚点、得分大于 1，故总 Index **允许超过 100**。未验证的尝试在官方代码中标为不完整/null，不能偷换为“已验证零分”。在线/离线 Solver 不混合打分。

## 论文 v1 报告的主要结果（不同套件不可横向直接比较）

- **Full（100 题）：** GPT-6 Astra 总分 **71.9**，为论文 Full 表中的最高分；其 theorem 70.0%、solver 64.7%、construction 104.9%、open conjecture 0%。Construction 超过 100% 是相对构造质量指标，不是答对率。
- **Lite（20 题）：** Opus 5.5 总分 **72.3**，在 Lite 表中居首；GPT-6 Astra **67.0**。不同套件/模型试次、Solver 开发协议存在差异。
- **Lite 与 Full：** 8 个共同参评系统的跨套件次序 Spearman 相关约 0.93；Lite 题目挑选过程涉及既有模型表现，相关性不等于新模型上严格无偏估计。
- **开放猜想：** 部分任务中出现已知反例的形式化或改写（如 Ryser、MIRUP），作者区分复用先前公开发现与新颖解题；不得把这些当作模型独立解决此前完全未知的研究问题。
- **新颖候选实例：** 报告了离散构造/数值界改进与特定定理子情形，例如 124 顶点、630 边且无三角形/四边形的构造、[[25,7,6]] 量子码与 AJT 特殊情形；这些是特定约束下的验证成果，不能概括为解决所有通用猜想。

## 方法学局限与正确读法

1. Benchmark 衡量模型加 agent harness、工具、算力预算与评测协议的组合，排名并非底座模型纯推理能力排序。
2. 形式验证提供“给定形式化命题/输入上的正确性”保证，不自动覆盖研究贡献的原创性、选题品味和自然语言题面等价性。
3. 参考驱动相对评分受 baseline 选择影响；绝不可直接拿总分当 100 道题正确率。
4. 新发现及开放猜想进展需要检查是否独立于已经发表的结果、是否只验证局部变体。
5. 这是 **TCS 研究基准，不是机器人操作、VLA 或 Sim2Real 基准**；对机器人知识库的价值在 agent 研究任务构建、可验证评分和复现流程，而非具身能力实证。
6. 大量评测依赖耗时模型执行、宿主机容器镜像及特定 verifier；本次只有静态核对，没有独立复跑。

## 来源与对应归档

- [arXiv HTML 全文 v1](https://arxiv.org/html/2611.07962v1)
- [作者项目页、可交互结果](../sites/tcsexam.md)
- [评测源码与运行入口](../repos/tcsexam.md)
- [TCSExam 官方 README](https://github.com/MoonshotAI/TCSExam/blob/master/README.md)
