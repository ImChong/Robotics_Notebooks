# JitRL：Just-In-Time Reinforcement Learning（arXiv:2601.18510）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2601.18510>（HTML 版 <https://arxiv.org/html/2601.18510> 用于通读；PDF <https://arxiv.org/pdf/2601.18510>）
- **标题（arXiv 元数据）：** Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates
- **作者：** Yibo Li、Zijie Lin（共同一作）、Ailin Deng、Xuan Zhang、Yufei He、Shuo Ji（通讯）、Tri Cao、Bryan Hooi（新加坡国立大学）
- **会议：** ICML 2026 Spotlight（依据代码仓库 README 自述；论文 HTML 页面写 ICML 格式）
- **代码：** <https://github.com/liushiliushi/JitRL>（已开源，公开仓库；见 [sources/repos/jitrl.md](../repos/jitrl.md)）
- **项目页：** 未检索到独立项目页（arXiv 与代码仓库均未列项目页）
- **入库日期：** 2026-11-07
- **最后更新：** 2026-11-07
- **一句话说明：** 无梯度的测试时策略优化：用非参数经验记忆检索优势估计，直接修改 LLM 输出 logits，在 WebArena / Jericho 上超过免训练基线，并在 WebArena-Lite 上报告优于 WebRL 的结果（骨干不同，见 wiki 页）。
- **用户提供的线索：** 知乎/B 站解读专栏（<https://www.bilibili.com/read/cv53311456>，<https://b23.tv/LgGEmvu>）——**本次未能获取**（直接请求返回约 1.4KB 的非正文页面），未纳入核对；解读内容不作为本页依据。
- **同批次对照论文（用户给出，本次未单独 ingest）：** WebArena <https://arxiv.org/abs/2307.13854>、WebRL <https://arxiv.org/abs/2411.02337>、Reflexion <https://arxiv.org/abs/2303.11366>。

## 核对范围

- **核对范围：** 完整 HTML 正文（§1 引言、§2 相关工作、§3 预备知识、§4 方法含定理 4.1–4.3、§5 实验含 Table 1–9、§6 结论、Limitations、Impact Statement）；附录 A（算法）、B–D（定理证明，只看结论与假设）、E（基线）、F（实现：状态抽象与检索）、G（提示词，仅抽样看开头）、H（超参数表 10–11）、I–J（同骨架对比 Table 12–13、Jericho 同骨架 Table 14）、K（Jericho 案例）、L（状态表示）、M–N（步数与可扩展性）、O（GRPO 超参扫描 Table 20）、P（成本）。
- **代码核对：** 仓库 `Jericho/src/jitrl_agent.py`、`Jericho/src/evaluation.py`、`Jericho/src/cross_episode_memory.py`、`Jericho/main.py`；`WebArena/memory_agents/jitrl_agent.py`、`WebArena/run.py`（浅克隆 HEAD `143d221`，2026-07-21）。
- **局限：** HTML 版公式（式 1–12）在文本提取中丢失，相关公式为根据定理 4.1 与附录 B 文字推断（见 wiki 页，已标注）；未逐行核对 Appendix G 提示词全文，也未核对 Table 12 的逐项数值。
- **通读：** 已完成

## 核心论文摘录（人工转述）

### 1) JitRL（Li et al., 2026）
- **链接：** <https://arxiv.org/abs/2601.18510>
- **核心贡献：**
  - 用 `<state, action, reward>` 三元组构建非参数经验库；测试时按当前状态检索近邻，估计动作优势 Â(s,a)，再以加性 logit 更新 `z' = z + Â/β` 修改冻结 LLM 的候选动作分布。
  - 声称该更新是 KL 约束策略优化目标的精确闭式解（定理 4.1，附录 B）；并给出估计量一致性的渐近结论（定理 4.2–4.3，附录 C–D，假设较强）。
  - 步级奖励由 LLM evaluator 在回合结束后给出，经折扣汇总为回报（γ：Jericho 0.5，WebArena 0.1，附录 H）。
- **主要结果（论文自报）：**
  - WebArena 全量（Table 1，Gemini-2.5-flash 骨干，免训练方法之间比较）：JitRL Avg 46.98 / Final 51.35，高于 Reflexion 41.08 / 42.12、Memory 41.36 / 43.00、AWM 39.37 / 40.32、EvoTest 39.24 / 42.49、Static 35.63 / 36.30。
  - WebArena-Lite（Table 2，held-out 子集）：JitRL Final 60.00（Gemini-2.5-flash 骨干）；WebRL 46.06（Llama-3.1-70B 骨干，WebRL 原始 checkpoint）；SFT 23.00。
  - Jericho（Table 3，Final 分数）：Zork1 JitRL 69 vs EvoTest 54、AWM 44、Reflexion 35；Library JitRL 30 vs EvoTest 26；Zork3 JitRL 5 vs EvoTest 4。
  - 成本（Table 9）：JitRL $290；WebRL $9,900（论文按 H200 训练成本估计）；其余免训练方法 $200–250（按 OpenRouter API 计费）。
- **对 wiki 的映射：**
  - [paper-jitrl-just-in-time-rl](../../wiki/entities/paper-jitrl-just-in-time-rl.md)
  - [agent-memory-and-dreaming](../../wiki/concepts/agent-memory-and-dreaming.md)（测试时记忆与经验复用）
  - [ai-agent-evaluation](../../wiki/concepts/ai-agent-evaluation.md)（WebArena / 多轮评测口径）

## 论文中的对照与对比材料（作为线索保留）

- **WebArena**（Zhou et al., 2024）：真实网站任务评测环境；JitRL 使用其 812 任务中的 WebArena-Lite 子集做 held-out 测试（论文 §5、附录 H）。
- **WebRL**（Qi et al., 2025）：在线自进化课程 RL，Llama-3.1-70B 骨干；论文以其公开 checkpoint 作为「权重更新」最强基线。
- **Reflexion**（Shinn et al., 2023）：回合末生成文字反思写入上下文的免训练基线。

## 对 wiki 的映射

- [paper-jitrl-just-in-time-rl](../../wiki/entities/paper-jitrl-just-in-time-rl.md)
