# Procedural Graphs 论文来源归档

## 元数据

- **题目：** Procedural Graphs: Self-Evolving Execution Structures for LLM Agents
- **作者：** Yuxing Lu, Yicheng Chen, Shanchan Wu, Sercan Ö. Arık
- **单位：** Google（四位作者）；Yuxing Lu 另列 Georgia Institute of Technology、Peking University（以 arXiv HTML 页脚署名为准）
- **预印本：** https://arxiv.org/abs/2609.09153
- **HTML：** https://arxiv.org/html/2609.09153v1
- **版本：** v1，2026-09-08，cs.AI（交叉 cs.CL、cs.MA）；36 页、6 图、11 表
- **代码与数据：** arXiv 摘要页与正文均未给出官方代码或项目页链接；截至 2026-10-12 未发现发布。

## 论文要解决的问题

长时程、需调用工具的 LLM 智能体常因“下一步做什么、什么顺序、什么条件下做”这类程序性知识只隐含在不断增长的历史里，而丢失目标、乱序调用工具、重复失败动作。论文把这类知识显式化为可查询的图，并让图在使用中自我修订。

## 方法与训练信息

1. **Procedural Graph（PG）：** 有向属性图 G=(V,R,E,Φ)，每条边是 (procedure, relation, procedure) 三元组；节点可为工具函数、技能、推理步或任务状态。关系词表四种：LEADS_TO、TRIGGERS、PROVIDES_INPUT_FOR、CONVERGES_TO。每条边带 condition、guidance、pitfalls 三个文本属性。
2. **活动节点定位：** 用上一步工具调用与节点精确匹配；成功则取 h=2 跳出边邻域子图，失败则给完整图。
3. **引导模型：** 输入子图、查询和最近 w=3 步轨迹，产出情境化 guidance g_t；求解器（ReAct）据此采样动作，引导只偏置不强制。
4. **精炼器（Refiner）：** 离线 LLM 对比失败与成功轨迹，提出对拓扑的 Add / Delete 编辑（属性修改 = 删除再加回），并做结构有效性检查与环修复。
5. **接受规则：** 候选图的验证分数不低于当前图才提交，否则回退；被拒候选及其轨迹保存为负面证据交回精炼器。
6. **初始化：** 可从专家先验或最小骨架（scratch）开始。
7. **评测设置：** 基准 HotpotQA、MultiChallenge、GDPval、ALFWorld、τ-bench、BFCL v3、EnterpriseArena；LLM 为 Claude Sonnet 4.6、Gemini 3.1 Pro、Gemini 3.5 Flash、Grok 4.1 Fast（temperature 0，引导模型与精炼器沿用求解器的 LLM）；基线 Vanilla ReAct、MemoryBank、RAP、ExpeL、AutoGuide、AWM、KnowAgent。附录：HotpotQA 1000/1000/1000 与 MultiChallenge 100/100/56 的 train/val/test；EnterpriseArena 演化 10 轮；agent 最长生成 2048 token，精炼器 8192。h、w 之外 L_max 的取值附录未给出。

## 论文报告的结果

- **主表（Table 1，PG 得分）：** 例如 Gemini 3.1 Pro：HotpotQA 87.30、MultiChallenge 95.78、GDPval 78.78、ALFWorld 100.00、τ-bench 80.00、BFCL v3 66.00；Grok 4.1 Fast 在 ALFWorld 仅 39.55。
- 24 个“模型×基准”设置中 PG 第一或并列第一 21 个；对各设置最强基线 19 胜 2 平 3 负（符号检验 p=4.3×10⁻⁴）。最大优势：BFCL v3 + Gemini 3.5 Flash（67.00 vs 58.00）、GDPval + Gemini 3.1 Pro（78.78 vs 71.37）、τ-bench + Gemini 3.1 Pro（80.00 vs 73.04）；HotpotQA 相对最强基线差距在 −0.90 至 +1.30。
- **EnterpriseArena（长时程财务模拟）生存率：** Claude 44.0%→58.0%，Gemini 3.1 Pro 6.0%→34.0%，Grok 4.1 Fast 26.0%→40.0%；Gemini 3.5 Flash 全配置全程存活率为 0.0%。10 轮自演化，验证生存率 0%→90%（第 8 轮），返回图测试 85% 对基线 0%（Fisher p=2.6×10⁻⁸）；单轮最佳 95%，论文报告的是返回图以避免测试集挑选。
- **构建方式消融（Table 2）：** 仅专家先验使 MultiChallenge 成功率由 87.50% 降到 58.93%，专家先验+在线演化回到 92.86%；从骨架起+在线演化在 HotpotQA 为 66.30 EM / 78.79 F1，无引导为 58.8 EM / 71.21 F1。
- **使用方式消融（Table 3，Gemini 3.5 Flash）：** 局部化生成式引导 MultiChallenge 89.31、GDPval 63.99、ALFWorld 81.53，优于无图；整图注入/整图生成式引导在 ALFWorld 分别 70.34 / 54.48，低于无图 72.58。局部化相对整图引导省 token：ALFWorld 70.9%、GDPval 18.1%、MultiChallenge 14.8%；但相对无图仍多耗 token（GDPval +33.4%，ALFWorld +55.4%）。

## 复现与解释边界

- 官方未发布代码，结果无法独立复现；基线实现细节与提示词以论文附录为准（本归档未逐表核对）。
- 引导模型与精炼器共用求解器 LLM、贪心解码；跨求解器与工具接口迁移未评估。
- EnterpriseArena 每个划分仅 20 个 episode，单次接受/拒绝可能由一两个 episode 决定，论文称之为搜索轨迹而非显著性检验。
- 论文自述引导增加 token 消耗，聚合统计不能指认具体哪条边编辑带来差异。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-procedural-graphs-self-evolving-llm-agents.md
- 概念入口：wiki/concepts/ai-agent-evaluation.md、wiki/concepts/recursive-self-improvement.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2609.09153>
- arXiv HTML：<https://arxiv.org/html/2609.09153v1>
