# RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments（arXiv:2609.15364）

> 来源归档（paper）

- **标题：** RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.15364>（v1 2026-09-14；v2 2026-09-18，cs.AI / cs.CL / cs.CV）
- **HTML：** <https://arxiv.org/html/2609.15364>
- **PDF：** <https://arxiv.org/pdf/2609.15364>
- **项目页：** <https://aetherlabsai.github.io/RSIAgent/>
- **代码：** <https://github.com/AetherLabsAI/RSIAgent>（Apache-2.0）
- **作者：** Sibo Zhu†, Shicheng Fan†, Xinyue Wang†, Wenyi Wu, Kun Zhou\*, Biwei Huang（Aether AI、UCSD、UIC）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2609.15364`（v2）正文，含摘要、§1 引言、§2 形式化定义、§3 方法（三角色隔离机制与 BRS/DRS 两阶段探索）、§4 实验（OSWorld 2.0 82 题、ALE Near-term 67 题 Table 1、RSI 轮次曲线、4 题消融 Fig. 4、GameCraft-Bench 40 题 Table 2、三类失败模式分析 §4.6）、§5 相关工作、§6 结论与 §7 局限；未在 Docker/KVM 虚拟机上实跑 OSWorld 评测
- **通读：** 已完成

## 核心摘录（通读 arXiv:2609.15364v2 全文）

- **形式化与免训练设计（§2–§3）：** 智能体 $\pi_\theta$ 固定权重，通过自主探索构建持久记忆 $M$，测试时冻结 $M$ 复用。采用 code-as-policy（生成可执行 Python / Bash 程序操作环境）。
- **三角色隔离（§3.2）：**
  - **Curriculum Agent**（默认 Kimi-K3）：依据目标 query 与已有记忆决定探索任务并判断是否停探。
  - **Actor Agent**（默认 GLM-5.3）：执行代码动作，在 Verifier 返回反馈后提炼经验并更新共享记忆。
  - **Verifier Agent**（默认 Kimi-K3）：上下文隔离（看不到 Actor 私有推理与记忆），直接检查环境可观测状态并给出 PASS/FAIL 与证据。
- **Broad-then-Deep 两阶段（§3.3）：** BRS 并行广度探索（默认名义预算 8 个项目、最多 4 并行，每波结束后统一按序合并记忆）；DRS 串行深度加难（针对失败、未知约束与边界条件做压力练习，直到 Curriculum 判定无需继续）。
- **实验与失败模式（§4, §7）：**
  - OSWorld 2.0（82 题，41 题实跑 RSI + 41 题沿用基线）：w/o RSI 71.97% → RSIAgent 78.98% partial（Binary 37.80% → 42.68%），高于 GPT-6 Astra 72.60%；ALE（67 题，19 题实跑 RSI + 48 题沿用基线）：83.75% → 84.82% partial（Binary 49.25% → 50.75%，低于 GPT-6 Astra 52.24%）。
  - GameCraft-Bench（40 题，Table 2）：在 4 个游戏生成器（Codex+GPT-5.5、Kimi-K2.6、GLM-5.3-Flash、Qwen3.8-27B）上均持续超越 Play2Code 与 w/o RSI 基线。
  - 三类失败模式（§4.6）：探索未打中目标真实弱点（Insufficiently Targeted Exploration）、Verifier 误放过未满足全部条件的执行（Incomplete Verification）、Actor 将欠验证结论固化进记忆（Unreliable Memory Consolidation）。
  - 局限（§7）：全部实验均在软件/计算机使用与游戏环境进行，无物理机器人实验；测试期探索引入显著 token 与时间成本；依赖外部模型 API 与有限预算策略。

## 对 wiki 的映射

- [aether-rsiagent](../../wiki/entities/aether-rsiagent.md)
- 博客与代码归档：[aether_rsiagent.md](../blogs/aether_rsiagent.md)
