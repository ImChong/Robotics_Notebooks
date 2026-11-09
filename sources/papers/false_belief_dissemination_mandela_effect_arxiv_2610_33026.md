# False Belief Dissemination, Persistence and Mitigation in Multi-Agent Social Networks Through the Lens of the Mandela Effect

> 原始资料归档（ingest · arXiv:2610.33026v1）

- **中文标题：** 从曼德拉效应视角研究多智能体社交网络中的虚假信念传播、持续与缓解
- **英文标题：** False Belief Dissemination, Persistence and Mitigation in Multi-Agent Social Networks Through the Lens of the Mandela Effect
- **作者：** Dianzhi Yu, Hongru Wang, Ziheng Xiong, Yanyu Chen, Zixing Song, Jianting Chen, Zhenfei Yin, Jeff Z. Pan, Amos Storkey, Irwin King
- **类型：** paper / multi-agent social networks / misinformation / persistent memory / agent safety
- **arXiv：** <https://arxiv.org/abs/2610.33026>
- **论文全文：** <https://arxiv.org/html/2610.33026>（v1）
- **PDF：** <https://arxiv.org/pdf/2610.33026>
- **版本：** v1，2026-10-11；39 页，6 图，13 表
- **入库日期：** 2026-11-09
- **核对范围：** arXiv HTML v1 的 Abstract、§1–§6、Ethics/Reproducibility/AI Use、References、附录 A–E（数据构造、实验配置、消融、提示词模板、案例）；核对文中数学定义、表 1–13 及图注；未将图像像素细节当作独立证据。
- **通读：** 已完成
- **源码/数据开放状态：** **计划发布，尚不能确认开放**。论文 §Reproducibility Statement 使用 “We plan to release the claim corpus, the background pages, and the code.”；arXiv 页面未给出官方代码/数据下载链接（截至 2026-11-09）。本归档不推断任何未核实的 GitHub 实现。
- **一句话说明：** 在可搜索的共享模拟网页 + Agent 私有记忆 + 社交消息的闭环中，虚假断言可传播至大量诚实角色 Agent，且造假者退出后依旧维持，定向纠错比泛化提醒有效。

## 原文结构与摘录要点

### 1. 研究问题：传播、残留和缓解（§1–§3）

- **任务：** 对一个虚构实体是否属于某集合/类别的错误断言 $q=(e\in C)$，每个 Agent 在各轮输出 accept / reject / uncertain；文中把“报告接受”作为可观察的虚假信念代理指标，不等于证明模型拥有人的主观记忆。
- **环境：** N 个 Agent，其中 K 个 fabricator 主动宣传目标断言；其他 honest Agent 仅指角色指令未造假，并不保证判断正确。沟通图有 fully-connected、binary-tree、two-community 三种；共享 simulated web 保存可被所有 Agent 搜到的文章，每个 Agent 维护私有记忆。
- **播种（第 0 轮）：** 造假者发消息并写支持文章。随后每轮先搜索，再根据邻居消息、搜索结果、私有记忆作判断并发言，更新记忆；造假者持续发文，接受错误断言的 honest Agent 也可能将其写回公共网页（victim write-back）。
- **持久性：** 在第 5 轮撤出造假者（不再发言、不再发文），保留既有文章与 honest Agent 记忆，测量后续判断是否改变。
- **指标：** R(t) 是第 t 轮接受断言的 honest Agent 比例；U 是全部 honest Agent 同意错误断言的 claim 比例；速度度量基于最后一次转入接受的轮次（§3.3）。

**对 wiki 的映射：** [论文实体节点](../../wiki/entities/paper-false-belief-mandela-effect-multi-agent.md)「核心原理」「流程总览」「指标与实验」。

### 2. 数据与实验设计（§4.1、附录 A/B）

- **合成断言：** 540 条，8 个领域、49 种实体子型、4 级可信度（L1 143、L2 133、L3 129、L4 135）；关联 1,618 个去重干净背景页面，排除提及虚构实体的页面。
- **筛选：** 从 561 个初始候选开始，用 Qwen3-4B 的无附加证据判断排除已直接接受的断言，再经修订与人工抽检得到 540 条；并非完全由人工逐条验证的真实世界事实问答集。
- **主实验：** 6 个 Agent（1 fabricator、5 honest），第 0 轮播种 + 10 轮交互；每轮按 BM25 返回 top-3 网页结果；10 个被评模型（Qwen3-8B/14B/32B/235B、Llama-3.1-8B/3.3-70B、Claude Sonnet 5/Opus 5、GPT-5.5、Gemini 2.5 Flash）。
- **注意：** 多数消融、对策实验默认 Qwen3-8B + 完全连接拓扑，不代表所有 10 个模型的统一对策分数。

**对 wiki 的映射：** [论文实体节点](../../wiki/entities/paper-false-belief-mandela-effect-multi-agent.md)「实验与评测」「局限与风险」。

### 3. 主结果与影响因素（§4.2、§5.1）

- **表 1：** 没有社交消息、检索和记忆的 direct-query 接受率仅 0–9.1%；交互后的最终 R 随模型/拓扑为 **33.7–96.0%**。Qwen3-32B 在 fully-connected 为 R=96.0%；这不是所有模型平均值。
- **拓扑效应：** 三种拓扑的最终 R 相近，传播速度有差异；Qwen 家族 claim-level 最后接受轮次 two-community 为 3.73、fully-connected 为 5.07。
- **表 2：** 4 个 Qwen 模型跨三拓扑均值，L1/L2/L3/L4 的最终 R 分别为 97.8/96.0/83.9/82.4%。
- **模型规模：** Qwen 家族从 8B 到 32B 风险不一定降低；跨模型对比不能简单称“大模型更安全”。
- **附录 C.1：** 固定 Qwen3-8B 造假者，使用其他模型为 honest Agent 时，仍能诱导错误接受。

**对 wiki 的映射：** [论文实体节点](../../wiki/entities/paper-false-belief-mandela-effect-multi-agent.md)「实验与评测」「与其他工作对比」。

### 4. 退出造假者、缓解与消融（§5.2/5.3、附录 C）

- **表 3：** 第 5 轮撤出造假者后，Qwen3-8B 与 Llama-3.1-8B 各可信度层级 R 相对第 5 轮变化仅在 ±0.8 个百分点内；实验保留共享网页和私有记忆。
- **表 4（Qwen3-8B、完全连接）：** 无防御 R=84.4%；prompt reminder 79.7%；official-source labeling 83.4%；新增针对性反驳文章 57.1%；平台统一广播纠错 27.8%。不同干预施加时间不同，不应把这些数字误解为同一个时间点的严格随机对照临床结论。
- **表 9：** All/Web/Conversation-only/Memory-only 的 R 分别 84.4/85.0/27.5/6.6%；共享网页是该实验中的关键影响通道，但不能归纳为“记忆对持续无作用”。
- **表 11：** honest Agent 永不发文时，R 仍为 83.1%（默认 accept 才发文为 84.4%）；说明 victim write-back 不是达到高接受率的必要条件。
- **附录 C.6：** 对 260 个适用断言询问后续推荐；不同模型及有无重新搜索下，虚构实体被推荐的比例为 13.1–75.8%，表明错误接受可能转为任务输出。
- **附录 C.7：** Claude Sonnet 5 在被指派 fabricator 发帖任务时 13/540 次拒绝（2.4%）、Opus 5 为 3/540（0.6%），其余 8 个被测模型此设置无观察到拒绝；拒绝样本被排除于表 1 分母。

**对 wiki 的映射：** [论文实体节点](../../wiki/entities/paper-false-belief-mandela-effect-multi-agent.md)「评测」「结论」「工程实践」。

### 5. 案例、可复现性与边界（附录 D/E、Ethics）

- **附录 D：** 包含角色指派、讨论轮次、发帖与四类缓解措施、后续推荐任务的提示词结构；不能替代未发布的可执行框架。
- **附录 E：** 案例说明 Agent 把多篇由同一污染来源链生成的文章错误当作独立证据；也给出对不可能数学断言的拒绝实例。
- **伦理边界：** 实验使用隔离的本地 simulated web，文章没有发布到真实互联网；评测针对合成虚构实体、文字立场与推荐，不是现实平台传播率测量。
- **开源边界：** 作者仅承诺“计划发布” corpus、背景页与代码；截至本次核查不能提供已公开官方入口，也不能画“基于源码审阅”的运行时序图。

## 与现有知识的连接

- [Agent 记忆与 dreaming](../../wiki/concepts/agent-memory-and-dreaming.md)：持久记忆的价值与错误信息保留风险（两者并非同一研究）。
- [AI Agent 评测](../../wiki/concepts/ai-agent-evaluation.md)：该论文提供了以环境交互回合而非单轮问答评估可信性的案例。
