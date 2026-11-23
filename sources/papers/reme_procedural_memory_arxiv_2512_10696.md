# ReMe：记住我、精炼我——经验驱动智能体演化的动态程序性记忆框架 / Remember Me, Refine Me

> 来源归档（paper / 一手原文通读；不收录 PDF 二进制）

## 元数据

- **英文题目：** Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution
- **中文题目：** 记住我、精炼我：经验驱动智能体演化的动态程序性记忆框架
- **作者：** Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, Hai Zhao
- **机构：** 上海交通大学 AGI Institute；阿里巴巴通义实验室（Tongyi Lab, Alibaba Group）
- **论文：** <https://arxiv.org/abs/2512.10696>
- **HTML 全文：** <https://arxiv.org/html/2512.10696v2>
- **PDF：** <https://arxiv.org/pdf/2512.10696v2>
- **正式收录：** Findings of ACL 2026，<https://aclanthology.org/2026.findings-acl.829/>
- **版本：** v1 2025-12-11；核对 v2 2026-04-15（20 页，10 图，15 表）
- **官方代码：** <https://github.com/agentscope-ai/ReMe>；当前 main 已演进为文件原生的 ReMe 产品架构，与论文实验实现不能不加区分
- **核对范围：** arXiv HTML v2 导论、相关工作、§3 方法、§4 实验与消融、§5 结论、Limitations、附录 A–E（包括 B.4 复现配置、D 的扩展实验、E 案例）；图示与提示词表请以 PDF 原页为准
- **通读：** 已完成

## 问题与贡献

传统经验记忆常把历史轨迹作为追加式档案：细节冗余、难跨任务调用，旧经验即便无益也持续被检索。ReMe 不更新底座模型权重，而是把可复用的程序性经验视为可筛选、检索、适配与淘汰的动态集合。

1. **多视角提炼（multi-faceted distillation）：** 对任务采样成功和失败轨迹，提炼成功关键步骤、首次失误点、成败对比洞见；通过 LLM-as-judge 验证和向量余弦相似度去重。
2. **场景自适应复用（context-adaptive reuse）：** 把经验组织为 〈使用情境、经验正文、关键词、置信度、工具〉，以使用情境 embedding 索引；新任务 top-K 检索，可选相关性重排与情境改写。
3. **效用驱动精炼（utility-based refinement）：** 成功任务选优写入；失败先反思重试，重试成功才收录；累计被检索频次 f 与有效成功贡献 u，低效且已有足够调用次数的经验被淘汰。

## 原文方法细节（§3 + 附录 B）

- 初始构建按任务采样 N=8 轨迹（temperature=0.9），选奖励最高/最低轨迹分析；单次成功轨迹的关键点比直接存完整历史更易复用。
- 经验字段 E=〈ω,e,κ,c,τ〉：ω 表示适用场景、e 为内容、κ 为关键词、c∈[0,1] 是置信度、τ 为用到的工具；用 ω 的向量作索引。
- 论文主要配置：text-embedding-v4（1024 维），检索 K=5；附录 B.4.2 另提及 Qwen3-Embedding 与 Elasticsearch，具体复现须固定所选 embedding 配置，不能混为相同实验。
- 记忆淘汰：当检索次数 f≥α=5 且 u/f≤β=0.5 时删除；失败后最多进行 3 次反思尝试，任务总迭代上限 30。
- 论文的 ReMe(fixed) 使用固定经验池，ReMe(dynamic) 随任务结果更新经验池；并非每一个工具调用后都重新检索，检索发生在任务开始时。

## 实验（§4、附录 D）

- 基准：BFCL-V3 多轮工具调用（50 个经验构建任务、150 个测试任务）；AppWorld（90 个训练、168 个 test-normal）。
- 指标：Avg@4 为四次独立试验的平均任务成功率；Pass@4 为四次至少一次成功的比例；主表每个结果三次独立运行报告均值与标准差。
- Table 1 跨 BFCL-V3 与 AppWorld 平均：Qwen3-8B No Memory Avg@4=27.65%、Pass@4=46.20%；ReMe(dynamic) 34.94%、55.03%；Qwen3-14B No Memory 35.62%、54.65%。所以小模型动态记忆的 Pass@4 略高于大模型无记忆，但 Avg@4 略低。
- **数据校对注意：** 由 Table 1 实数直接相减，Qwen3-8B 的提升是 Avg@4 **+7.29 个百分点**、Pass@4 **+8.83 个百分点**；原文 §4.2 和摘要附近的 7.29 / 8.83 指标表述有互换迹象。以主表具体数值为准，不转述为相对增幅。
- Table 3（BFCL-V3，Qwen3-8B）：全量写入 40.83/62.00，选择性写入 44.33/64.66，加失败反思 45.00/64.66，再加淘汰 45.17/68.00（Avg@4/Pass@4，%）。
- Table 6（AppWorld）：单任务平均推理延迟无记忆 21.42 秒、ReMe 23.96 秒，增加 2.54 秒；额外调用并非零成本。
- 附录 Table 8：完整 BFCL-V3 Multi-Turn 750 任务 Qwen3-8B Pass@4 37.78→43.02（+5.24 个百分点），并非只在筛选的 Base 测试上报告改善。

## 局限（原文 Limitations 与工程边界）

- **静态召回时机：** 每个任务开始时召回一次，任务中途状态突变不会自动重新注入经验。
- **验证误差：** LLM-as-judge 不保证所提炼经验在新场景真实有效；u/f 也不是严格因果贡献估计。
- **迁移边界：** BFCL/AppWorld 是软件工具使用基准，不能视作机器人控制或真机的成功率证明。
- **代码版本漂移：** 当前 ReMe main 有 Markdown 文件记忆与 Auto Dream 等产品功能；不能用当前主分支的运行步骤直接宣称复现论文 Table 1。

## 对 Robotics_Notebooks 的映射

- 论文详情：wiki/entities/paper-reme-procedural-memory.md
- 对照：wiki/entities/paper-procedural-graphs-self-evolving-llm-agents.md
- 纵深路线：roadmap/depth-rsi.md（Stage 1，外部经验持久化，不涉及权重级 RSI）
- 官方仓库：sources/repos/agentscope-ai-reme.md

## 一手来源

- [arXiv v2 HTML](https://arxiv.org/html/2512.10696v2)
- [Findings of ACL 2026 出版页](https://aclanthology.org/2026.findings-acl.829/)
- [论文配套 ReMe 仓库](https://github.com/agentscope-ai/ReMe)
