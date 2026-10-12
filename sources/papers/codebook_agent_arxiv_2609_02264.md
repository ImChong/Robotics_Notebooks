# Codebook Agent 论文来源归档

## 元数据

- **题目：** Codebook Agent: Amortized Topology Design for LLM Multi-Agent Systems
- **作者：** Jinxi Yu, Yubei Li, Eric Hanchen Jiang, Zhi Zhang, Dong Liu, Wenxiao Zhao, Levina Li, Kai-Wei Chang, Ying Nian Wu
- **单位：** University of California, Los Angeles（arXiv HTML 署名页；其余作者的具体机构行未写明）
- **预印本：** https://arxiv.org/abs/2609.02264
- **HTML：** https://arxiv.org/html/2609.02264v1
- **版本：** v1，2026-09-02，cs.AI（交叉 cs.LG、cs.MA）
- **代码：** https://github.com/jinxiy1104/CodebookAgent （2026-10-12 核查，已克隆浏览）
- **周报线索：** [WaytoAGI 转载 DAIR.AI 周报](../blogs/waytoagi_feishu_top_ai_papers_2026-09-07-09-13.md)（仅作线索，技术事实以 arXiv 为准）

## 论文要解决的问题

现有 LLM 多智能体拓扑设计器（GPTSwarm、G-Designer、ARG-Designer、TopoDIM、GTD 等）把“每个查询选什么通信图”当作条件图生成问题：用变分、自回归或扩散解码器采样邻接矩阵，再用图网络代理打分，推理时需迭代搜索。论文提出三个观察：(1) 通过奖励过滤的拓扑在码本容量 8 到 64 时都坍缩到约六种图；(2) 边数与实测 token 消耗呈负相关（r 约 -0.4），稀疏图不一定便宜；(3) 当各智能体画像相同（已发布基准的默认设置）时，基于节点画像的消息传递评分器无法区分候选图。

## 方法与训练信息

- **拓扑码本：** VQ 自编码器把展平的邻接矩阵编码为 32 维潜变量并量化到 16 个码；只用成功拓扑（utility > 0.5）训练；EMA 更新码本，commitment 系数 0.25，50 个 batch 后重置死码；200 epoch，Adam，lr 3e-4，batch 16。
- **奖励：** R = u - λ·τ̃，λ = 0.1，τ̃ 为同一任务内按均值归一化的 token 数。
- **码预测器：** MLP（d→256→256→K，dropout 0.1），用温度 γ = 2 的奖励软目标训练，输入为查询嵌入（冻结 all-MiniLM-L6-v2，384 维）。
- **执行代理：** MLP，输入展平邻接矩阵加查询嵌入，预测效用与归一化 token 成本；训练期带辅助结构 token 头（32 码，权重 0.1）；200 epoch，Adam，lr 1e-3，batch 256。
- **推理：** 取预测器 top-5 码，解码、去重，选 û - λĉ 最大者，一次批量前向。
- **训练数据：** 每个基准 50 个训练任务 × 6 种固定拓扑 = 300 条执行记录。
- **骨干：** 主实验 gpt-4o-mini（temperature 0.7，top-p 1.0，max tokens 1024）；迁移实验 Qwen-3-8B。

## 论文报告的结果

- 基准：GSM8K、MATH、MultiArith、SVAMP（推理），MBPP、HumanEval（代码，Pass@1），迁移到 MMLU。
- gpt-4o-mini 六基准平均：Codebook Agent 84.62，最强先验 GTD 83.02，Vanilla 77.63。逐项为 GSM8K 94.8、MATH 56.5、MultiArith 99.4、SVAMP 95.4、MBPP 83.5、HumanEval 78.1，均为表中最高。
- Qwen-3-8B（GSM8K / MATH / MMLU）：Codebook Agent 92.7 / 63.5 / 65.8，平均 74.0；GTD 平均 72.7；CoT 平均 67.5。
- 延迟：每个拓扑 2.4 ms，迭代式生成器为 301–396 ms。
- token：相对完整的现有流水线减少 21.9%–33.2%，例如 GSM8K 1239→927，MATH 2304→1611。

## 复现与解释边界

- 论文无独立局限性小节；从文中可读到：设计轴分析每基准只用 200 个任务、每配置单次运行；训练集仅 300 条记录；主结果多为同质智能体团队。
- 数据集文件不随仓库提供，需自行按 README 下载并抽取固定的 test200 子集，绝对数值依赖抽到的题目。
- 结果依赖付费 LLM API（OpenAI 兼容接口）。
- 附录未单独核对；上述数字来自 arXiv HTML 正文表格抽取。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-codebook-agent-topology-design.md
- 代码归档：sources/repos/codebook-agent.md
- 概念入口：wiki/concepts/ai-agent-evaluation.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2609.02264>
- arXiv HTML：<https://arxiv.org/html/2609.02264v1>
- 代码：<https://github.com/jinxiy1104/CodebookAgent>
