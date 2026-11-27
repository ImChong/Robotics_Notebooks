# Self-Evolving Embodied Agents via Skill-Harness Evolution

- **作者：** Peidong Wang, Zhiming Ma, Ying Chang, Xufang Luo, Yiqun Zhang, Zihan Wang, Xiaocui Yang, Shi Feng, Yuqing Yang, Dongsheng Li
- **机构：** Northeastern University（东北大学，中国）、Microsoft Research（微软研究院）
- **版本 / 载体：** arXiv:2608.11350v2 [cs.CL]（v1 提交于 2026-08-11，v2 修订于 2026-09-10）
- **原文：** https://arxiv.org/abs/2608.11350（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2608.11350）
- **项目页 / 代码：** 未确认开源代码（arXiv:2608.11350v2 原文、摘要及本次核查材料未给出官方项目主页或 GitHub 代码仓库链接；仅 Appendix A 展示了实验用技能文本、上下文构建函数接口与优化器提示词模板）
- **核对范围：** 通读 arXiv:2608.11350v2 全文（Abstract、Sections 1–5、References、Appendix A.1–A.4、Tables 1–3、Figures 1–4 及 A1–A13 题注与说明）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[SHAPER 实体页](../../wiki/entities/paper-agentic-2608-11350.md)

1. **问题背景与系统分解（Sections 1 & 3.1）**
   - 具身智能体逐渐演化为围绕基础模型构建的复合系统，性能不仅取决于模型参数，还受外围技能、上下文组织、动作接口与执行框架（harness）影响。
   - 监督微调（SFT）与强化学习（RL）需要访问模型权重及额外数据/奖励；现有免训练代码生成方法（如 Code as Policies、Voyager、ASPIRE）依赖可编程机器人 API，在固定或受限动作接口下难以适用。
   - 论文提出 **SHAPER**，将免训练具身适应形式化为非参数系统部件优化：保持 VLM 规划器 $M_\theta$ 与底层执行器 $\mathcal{A}_\phi$ 参数冻结，仅优化可复用文本技能 $s \in \mathcal{S}$ 与可执行 Python 上下文代码框架 $h \in \mathcal{H}$（如 `build_context(history)`）。
   - 单步执行为 $y_t \sim M_\theta(\cdot \mid g, o_t, s, h(\tau_{<t}))$，$a_t = \mathcal{A}_\phi(y_t, o_t)$；目标是最大化目标环境分布上的期望任务回报 $J(s,h) = \mathbb{E}_{g \sim \mathcal{D}}[R(\tau^{(s,h)}(g))]$。

2. **轨迹文本诊断与两阶段束搜索演化（Sections 3.2–3.3, Appendix A.3）**
   - **分层文本诊断**：避免将大量多图轨迹直接喂给优化器导致上下文过载：
     - 轮次级裁判（Round-level Judger）：VLABench 中对每轮执行前后观测 $(o_t, y_t, a_t, o_{t+1})$ 生成局部诊断 $q_t^c(g)$；ESI-Bench 中使用轨迹级诊断裁判（Diagnostic Judger）核查证据获取、探索效率、答案支撑与环境有效性，并将主因归类到 skill / harness / environment / neither。
     - 回合/批次摘要器（Episode / Batch Summarizer）：结合非视觉元数据 $m^c(g)$、终端上下文 $h(\tau_{<T})$、局部诊断与最终奖励 $R(\tau^c(g))$ 压缩生成片段摘要，与汇总执行统计拼接形成文本梯度 $\Gamma(c)$。
   - **两阶段演化与沙盒校验**：同一冻结 VLM 在不同提示词下同时扮演规划器 $M_\theta$ 与优化器 $\mathcal{O}$。优化按固定两阶段推进：先固定初始框架 $h_0$ 演化技能 $s \to s^\star$，再固定 $s^\star$ 演化上下文代码框架 $h \to h^\star$。生成的代码框架须先通过沙盒校验（必须定义契约函数、禁止 `import`/文件 I/O/动态执行、控制负载规模与超时），不合格则将报错回传给优化器；合法候选在固定验证集 $\mathcal{D}_{\mathrm{val}}$ 上评估并通过宽度为 $K$ 的束搜索（实验中 4 轮、束宽 $K=3$、分支因子 2、训练小批量大小 4）保留 Top-$K$。

3. **基准设置与关键实验结论（Section 4 & Appendices A.1–A.2, A.4）**
   - **统一底座**：两个基准均使用冻结的 **Qwen3.6-27B** 作为上层规划器与演化优化器。
   - **VLABench（VLA 子目标接口）**：底层执行器为基于 OpenPI 部署的官方 VLABench $\pi_0$ 检查点（5 动作块输出，单回合上限 10 轮规划 / 400 步环境交互）。用 15 条训练回合与 24 条验证回合演化，在 4 个自定义留出划分 C1–C4（每组 200 回合，共 800 回合）评测：
     - 总体成功率：Direct VLA 23.25%、同数据 SFT 24.00%、VLA+MG-Select 21.38%、VLA+VOTE 22.00%、Seed Agent+MG-Select 24.38%、Seed Agent+VOTE 23.50%、Seed Agent 28.25%、Skill Evolution 33.50%、Harness Evolution 30.50%、**SHAPER 34.50%**。
     - 分布偏移表现：SHAPER 相对 Seed Agent 在 C1（+2.5）、C2（未见目标类别，+6.0）、C3（未见任务形式/常识推理，+10.0）与 C4（双偏移，+6.5）均有提升；仅对 15 条轨迹做 SFT 在 C3 上反而从 45.0% 降至 38.0%，测试时缩放（MG-Select / VOTE）整体均低于未缩放基线。
   - **ESI-Bench（主动感知固定 API 接口）**：10 类闭环空间智能任务，最大 30 步交互；使用 10 条训练题与 10 条验证题演化，在按官方比例采样的 231 题子集评测：
     - Micro Avg：Seed Agent 32.5% → Evolved Skill 41.1% → **SHAPER 49.8%**（+17.3 pp）；Macro Avg：Seed Agent 31.2% → Evolved Skill 38.6% → **SHAPER 42.9%**（外部已发表参考 GPT-5 Passive Single-view 全集 40.3%）。
     - 上下文框架演化在镜像反射（Specular Reflection，20.0% → 60.0%）、感知接地（Perceptual Grounding，59.2% → 69.4%）与空间关系（Spatial Relations，37.3% → 54.9%）增益显著，核心来自稀疏证据保留、确定性 RGB 局部裁剪（基于对比度/边缘密度/清晰度，不使用位姿/深度等特权状态）以及循环检测。
   - **一次性演化开销（Section 4.4, Appendix A.4）**：按阿里云百炼 Qwen3.6-27B 标价（输入 CNY 3/M tokens、输出 CNY 18/M tokens，约 USD 0.42/M 与 USD 2.50/M）折算，单次完整演化 API 等效费用为 **VLABench 约 $2.25**、**ESI-Bench 约 $2.83**（不含最终评测及仿真/GPU 基础设施开销），演化产物可在全部留出测试中免搜索复用。

4. **主要局限（Sections 4.3, 5, Appendix A）**
   - VLABench 的 C1–C4 为作者自定义留出划分，ESI-Bench 为 231 题采样集且 GPT-5 PS 来自外部全集论文报告，非严格配对实验；部分子类（如 Action Sequencing $n=5$）样本数极少，演化后个别子类准确率出现下降（33.3% → 27.8% 及 40.0% → 20.0%）。
   - 论文明确指出跨本体迁移与真实机器人物理验证仍留待后续工作，且尚未公开可复现仓库与完整源码文件。
