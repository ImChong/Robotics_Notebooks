# MAPE：通过程序进化发现有效的智能体记忆设计

> Discovering Effective Agent Memory Designs through Program Evolution（arXiv:2611.17068v1）

- **类型：** paper / agent memory / program evolution / retrieval / evidence composition
- **英文标题：** Discovering Effective Agent Memory Designs through Program Evolution
- **中文标题：** 通过程序进化发现有效的智能体记忆设计
- **论文：** https://arxiv.org/abs/2611.17068
- **HTML：** https://arxiv.org/html/2611.17068v1
- **PDF：** https://arxiv.org/pdf/2611.17068
- **版本：** arXiv:2611.17068v1，2026-11-12（24 页，预印本）
- **代码：** https://github.com/ash-ding/MAPE
- **机构：** 明尼苏达大学（University of Minnesota）、史蒂文斯理工学院（Stevens Institute of Technology）
- **作者：** Xinheng Ding、Seyyed Saeid Cheshmi、Ziyang Zhang、Tairan Chen、Wenya Xie、Yu Hou、Rui Zhang、Zirui Liu
- **入库日期：** 2026-11-23
- **通读：** 已完成
- **核对范围：** 通读 arXiv v1 HTML 全文：摘要、§1–6、附录 A（接口）/B（搜索与选择）/C（切分、基线和指标）/D（跨基准、消融、RQS 迁移）/E（QA 与 Judge 提示）；核对表 1–3、附录 4–10 的数据口径，并对照官方 MAPE README、发布程序、运行脚本和架构。未自行执行模型实验。

## 论文核心摘录

### 问题

记忆系统通常由人手选定写入摘要、索引、查询时检索与上下文组织的规则。MAPE 不只调一个 prompt，而让候选 Python 记忆程序在统一接口下整体进化，并以准确率和服务成本反向驱动设计。

### 方法（§2、附录 A/B）

1. **记忆生命周期**：Build 将对话历史 H 转成持久记忆 M；Retrieve 按问题 q 选证据 E；Compose 把 q、E 编排为回答上下文 c；Answer 使用固定的 QA 模型 G 生成答案。候选程序可改前三阶段，回答器和判分协议在同一次比较中固定。
2. **程序进化**：从 no-memory / full-context 等种子起步；proposer 使用归档候选与诊断写出程序及假设；sanity check 失败可有限修补；在固定 search split 评测准确率与生命周期服务成本；独立诊断读取轨迹并记录失败原因；程序及诊断都归档，进入下一轮。MAPE-A 只优化准确率，MAPE-P 在准确率与成本上维护 Pareto frontier。
3. **选择**：只用 search split 进化，随后对入围程序在 held-out evaluation split 评测并筛除不可扩展实现；论文报告的是最终筛选结果，不能误写为完全零后验选择。
4. **发现的设计**：同时索引抽取的原子笔记和原始对话轮次，防止摘要丢细节；Query-Routed Scaffolding（RQS）先按问题意图组织证据，以日期运算、计数求和、当前值更新和偏好判断等专用脚手架准备上下文，最终回答仍由固定 QA 模型输出。

### 实验（§3–4、附录 C/D）

- LoCoMo：10 条长对话；2 条对话/233 题用于搜索，8 条/1,307 题用于评估，排除对抗题；GPT-4.1 mini 下 MAPE-A / MAPE-P 准确率 **87.94% / 85.18%**，成本 **$7.21 / $2.03 每千问**。
- LoCoMo 改 Qwen3-30B-A3B-Instruct-2507（未重新进化）：MAPE-P **81.33% / $1.14**，SimpleMem **69.32% / $3.24**；即 +12.01 个百分点、约 35% 的服务成本，不能与 GPT-4.1 mini 的成本或排名混用。
- LongMemEval-S：100 题搜索、400 题评估；Qwen3 下 MAPE-A **77.00% / $52.84**、MAPE-P **72.25% / $5.76**，RAG **66.00% / $2.62**（准确率与 $/千问）。
- 消融：移除原始轮次使 MAPE-A_LME 准确率在 LoCoMo/LongMemEval 分别下降 **12.17 / 6.50** 个百分点；移除 RQS 分别下降 **4.44 / 6.25** 个百分点；将 RQS 接入五个既有系统，LongMemEval 整体准确率都提高，增幅 **2.75–9.25** 个百分点。
- 代价转移：MAPE-A_LME 的构建阶段约占服务成本的 92%；成本联合优化版本减少构建期 LLM 调用，按问题进行选择性处理。论文服务成本含四阶段 LLM/embedding token 费用，**不含**本地计算、存储、搜索 agent 与 judge 成本，也不等同端到端时延。

### 重要边界

1. 这是 **语言对话 Agent 的持久记忆研究**；LoCoMo / LongMemEval 是文本 QA，不是机器人控制或真机记忆实验。
2. 20% search + 80% evaluation 的隔离用于进化，但最后**评估集参与候选最终选择**；因此不是一次性完全独立 test 评估。
3. RQS 只改善证据利用；检索源缺失时不能凭编排补出真实事实；证据完整度及正确性仍需独立测量。
4. 官方代码已开放，但 README 明确 **无附带模型运行结果/trace**；未运行实验，不宣称复现准确率。

## 对 wiki 的映射

- [MAPE 独立论文详情](../../wiki/entities/paper-mape-agent-memory-program-evolution.md)
- [Agent 记忆与 dreaming](../../wiki/concepts/agent-memory-and-dreaming.md)
- [AI Agent 评测](../../wiki/concepts/ai-agent-evaluation.md)
- [官方代码阅读归档](../repos/mape-agent-memory-program-evolution.md)

## 参考原文

- [论文 HTML v1](https://arxiv.org/html/2611.17068v1)
- [论文 PDF](https://arxiv.org/pdf/2611.17068)
- [官方实现](https://github.com/ash-ding/MAPE)
