# Robota-Embed: Unified VLA Embeddings for Robot In-Context Learning and Evolution（arXiv:2610.19495）

> 来源归档（ingest）

- **标题：** Robota-Embed: Unified VLA Embeddings for Robot In-Context Learning and Evolution
- **缩写 / 框架：** **Robota-Embed**（统一的 VLA 嵌入模型，联合编码任务、执行阶段与后续动作）
- **类型：** paper / vla / embedding / in-context-learning / retrieval / memory-evolution
- **arXiv：** <https://arxiv.org/abs/2610.19495>（Submitted 2026-10-14，v1；PDF：<https://arxiv.org/pdf/2610.19495>）
- **项目页：** <https://robota-ai.github.io/robota-embed/>（页面标注 Report available；Code、Model &amp; benchmark 均为 **Coming soon**）
- **代码：** 未列出仓库（项目页写 Code Coming soon；第三方代码检索站点无具体仓库 URL）
- **作者：** Yunze Liu、Jing Wang、Shenrui Wu、Wenkun He、Guanqi Zhan、Xianzheng Ma、Junyu Chen、Wenhang Ge、Haozhan Tang、Yuchao Gu、Zhuoyang Zhang、Zekai Wang、Dantong Niu、Zhen Dong、Yuke Zhu、Trevor Darrell、Linxi "Jim" Fan、Song Han、Han Cai
- **机构：** 未核实（arXiv 摘要页与项目页未列出机构）
- **入库日期：** 2026-10-15
- **来源文章：** 微信「具身站 31 篇论文」摘要，见 [sources/raw/wechat_embodied_station_robot_31_papers_2026-10-15.md](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)（文章一句话：联合编码视觉、语言和动作，检索当前阶段可用的后续经验；需分清仿真成功率与真机任务进展）
- **一句话说明：** 训练一个同时编码观测、指令与动作的 VLA 嵌入，用于检索在当前执行阶段可用的经验，并驱动上下文学习（Embed2ICL）与局部记忆演化（Embed2Evolve）。

## 开源状态（步骤 2.5，2026-10-15）

- **项目页核查：** 页面显示 Code「Coming soon」、Model &amp; benchmark「Coming soon」，无 GitHub / Hugging Face 权重链接。
- **结论：** **待发布**（宣称将开源，截至入库日未列任何仓库或权重 URL）。

## 摘录 1：问题与主张（摘要，WebFetch 摘要转述，未逐字核对原文）

- **主张：** 首个同时在 query 与录制的未来中编码任务、执行阶段与后续内容的 VLA 嵌入模型。
- **训练语料：** 约 170 万条清洗后的 anchor。
- **基准：** Robota-Embed-Bench，4,000 条 query，8,000 条 gallery。

## 摘录 2：检索结果（摘要数字）

- 最终 checkpoint：clip 级 R@1 47.33%，episode 级 R@1 81.77%；优于一个不含动作信息的通用嵌入基线。
- 去掉动作数据对检索的损害大于去掉视觉数据。
- 项目页：去掉 query 中的动作后，四域平均 clip 级 R@1 由 47.33% 降至 20.08%。

## 摘录 3：下游应用（摘要数字）

- **Embed2ICL：** 在 DROID-Sim 与 LIBERO 上提升策略成功率（具体数字未核实）。
- **Embed2Evolve：** 在五个 ManiSkill 任务上达到 68% 成功率，策略 token 用量较对照 harness 约减少 79.5%。
- 文章与摘要区分了仿真成功率与真机任务进展，本库页面沿用这一区分。

## 对 wiki 的映射

- 升格页面：[paper-robota-embed](../../wiki/entities/paper-robota-embed.md)
