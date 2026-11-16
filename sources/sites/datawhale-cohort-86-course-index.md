# 第 86 期组队学习：课程一手来源核验与入库导航

- **类型：** 课程 / 项目来源索引
- **输入：** 用户提供的「组队学习 No.86」21 行课程表截图；非官方课程原文
- **收录日期：** 2026-11-13
- **核对范围：** 对照各项目维护者 GitHub README 与沐曦官方课程页面；已存在节点通过本库检索确认；课程全章节与执行代码尚未全部阅读
- **通读：** 未完成
- **状态：** 来源索引；不是代表课程全文均已精读的百科节点

## 21 项与去重后的来源

| 序号 | 专题 | 图片中的课程 / 书籍 | 官方来源 | 核验结果 |
|---:|---|---|---|---|
| 01 | LLM | Agentic RL Lab | [一手来源](https://github.com/datawhalechina/agentic-rl-lab) | 已有 wiki/entities/agentic-rl-lab.md 与来源归档；当前 draft |
| 02 | LLM | Happy-LLM | [一手来源](https://github.com/datawhalechina/happy-llm) | 官方课程仓库 |
| 03 | LLM | llm-algo-leetcode · 推理优化 | [一手来源](https://github.com/datawhalechina/llm-algo-leetcode/tree/main/topic_discussion/inference_optimization) | 同一课程的专题入口 |
| 04 | LLM | llm-algo-leetcode · 后训练优化 | [一手来源](https://github.com/datawhalechina/llm-algo-leetcode/tree/main/topic_discussion/post_training_optimization) | 同一课程的专题入口 |
| 05 | LLM | llm-algo-leetcode · 模型架构与训练 | [一手来源](https://github.com/datawhalechina/llm-algo-leetcode/tree/main/topic_discussion/llm_architecture_evolution) | 同一课程的专题入口；训练细目以 README 为准 |
| 06 | LLM | llm-algo-leetcode · 算子优化 | [一手来源](https://github.com/datawhalechina/llm-algo-leetcode/tree/main/topic_discussion/operator_optimization) | 同一课程的专题入口 |
| 07 | Agent | Deep Agents 实战 | [一手来源](https://github.com/datawhalechina/deepagents-in-action) | 官方课程仓库 |
| 08 | Agent | Hello-Agents 入门篇 | [一手来源](https://github.com/datawhalechina/hello-agents) | 与进阶篇同一课程仓库 |
| 09 | Agent | Hello-Agents 进阶篇 | [一手来源](https://github.com/datawhalechina/hello-agents) | 与入门篇同一课程仓库 |
| 10 | Agent | All-in-RAG | [一手来源](https://github.com/datawhalechina/all-in-rag) | 官方课程仓库 |
| 11 | Agent | Easy Data × AI | [一手来源](https://github.com/datawhalechina/easy-data-x-ai) | 官方课程仓库 |
| 12 | AI 编程 | Easy-Vibe | [一手来源](https://github.com/datawhalechina/easy-vibe) | 已有 wiki/entities/easy-vibe.md 与来源归档 |
| 13 | FDE | 企业本体论：人人都是 FDE · 黄佳 | [一手来源](https://github.com/huangjia2019/ontology-fde) | 作者配套可运行项目；不是书籍全文 |
| 14 | FDE | 前线部署工程师 · 范冰 | [一手来源](https://github.com/xdash/FDE-the-Guidance-Book-of-Forward-Deployed-Engineer) | 作者公开全文；纸质版标题以原书为准 |
| 15 | AI 硬件 | Nailongbot | 待核实 | 未确认与图中描述相符的官方仓库；不以其他奶龙同名项目代替 |
| 16 | AI Infra | hello-gpu | [一手来源](https://github.com/datawhalechina/hello-gpu) | 官方课程仓库，聚焦 AMD GPU / ROCm / Profiling |
| 17 | AI Infra | hello-rocm | [一手来源](https://github.com/datawhalechina/hello-rocm) | 官方教程仓库 |
| 18 | AI Infra | GGUF 量化内幕 | [一手来源](https://github.com/datawhalechina/gguf-quant-internals) | 官方教程仓库 |
| 19 | AI Infra | cpp-llm-engine | [一手来源](https://github.com/datawhalechina/cpp-llm-engine) | 官方课程仓库 |
| 20 | AI4S | AI for Science：科学模型与科研智能体实战 | 待核实 | 图中课程身份尚未确认；磐石实训营与此可能同主题但未经对应确认 |
| 21 | AI4S | 沐曦 AI4S 实战营 | [一手来源](https://developer.metax-tech.com/activities/40) | 沐曦官方活动课程页面 |

## 归一化原则

- **21 个课表条目不等于 21 个实体。** llm-algo-leetcode 的 4 条方向共用同一仓库；Hello-Agents 入门与进阶共用同一仓库，因此合计 17 个候选实体。
- **避免重复实体。** Agentic RL Lab 与 Easy-Vibe 已有 wiki 与 sources，不新建同名实体；只能在读完原文后原位升级。
- **图中名称不等于官方标题。** 例如黄佳《企业本体论》作者配套仓库已公开，但不能等同公开书籍全文；范冰开源书 README 的书名为《前线部署工程师：人工智能时代的客户价值交付秘籍》。
- **未确认的 Nailongbot 与「AI for Science 科学模型与科研智能体实战」** 保持待核实，不借用不相干项目链接充数。
- **来源层不是完整知识节点。** 若未来升格 wiki，必须通读原文、检查源码入口，绘制有依据的 Mermaid 主流程图和时序图，并通过 `make ci-preflight`、`report_health.py`，真实达到 3/3；本索引不宣称任何新 wiki 节点满分。

## 与现有知识库

- [Agentic RL Lab](../../wiki/entities/agentic-rl-lab.md)：原位维护，当前尚有未通读的实验章节。
- [Easy-Vibe](../../wiki/entities/easy-vibe.md)：原位维护，勿重复录入。
- [Agentic Coding 软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)：联系 AI 编程教程的工程约束。
- [检索增强生成（RAG）](../../wiki/concepts/retrieval-augmented-generation.md)：联系 All-in-RAG 与 Easy Data × AI。

## 下一步验收清单

- [ ] 逐个通读教程核心章节并记录精确版本。
- [ ] 核查可运行代码入口和实际模块交互（非示意图）。
- [ ] 先查重、后创建或更新独立 wiki 实体，新增页应被现有 wiki 页链入。
- [ ] 对新增 / 主要更新的实体执行完整 CI 与健康度脚本，并在 PR 正文逐页列出分数。
- [ ] 补齐两个尚未确认的一手课程来源。
