# Playful Agentic Robot Learning

- **作者：** Junyi Zhang, Jiaxin Ge, Hanjun Yoo, Letian Fu, Zihan Yang, Yaowei Liu, Raj Saravanan, Shaofeng Yin, Justin Yu, Dantong Niu, Zirui Wang, Roei Herzig, Ken Goldberg, Yutong Bai, David M. Chan, Ion Stoica, Angjoo Kanazawa, Jiahui Lei, Haiwen Feng, Trevor Darrell
- **机构：** UC Berkeley; Impossible Research
- **版本 / 载体：** arXiv:2606.19419v1 [cs.RO]（2026-06-17 提交）；CoRL 2026（Spotlight，见项目页与官方仓库 README）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2606.19419) / [arXiv Abs](https://arxiv.org/abs/2606.19419) / [项目页 PDF](https://playful-rats.github.io/assets/papers/rats-paper.pdf)
- **项目页 / 代码：** 已开源 — 官方项目页 [playful-rats.github.io](https://playful-rats.github.io/) 与官方 GitHub 仓库 [Playful-RATs/RATs](https://github.com/Playful-RATs/RATs)（Apache-2.0 协议，含 LIBERO-PRO、MolmoSpaces、RoboSuite 与真机 Franka 跨环境评测入口及 `capx-baseline/` 子模块）
- **核对范围：** arXiv:2606.19419v1 全文（Abstract、§1–§5、Algorithm 1、Table 1–4、Fig. 1–4、Appendix A–F 含 Table 5–15、Fig. 5–12 与全部智能体 Prompt 模板）及官方项目页 `https://playful-rats.github.io/` 抓取文本
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心问题与设定（§1, §3.1）：** 现有基于 Code-as-Policy 的机器人智能体多为“任务驱动”（收到外部指令后才编写并修正代码，技能只是被动副产品）。本文提出 **Playful Agentic Robot Learning**，在下游任务到达之前引入自主玩耍（Self-Directed Play）阶段，通过 **RATs（Robotics Agent Teams）** 将无外部指令的探索转化为持久化的可调用 Python 代码技能库 $\mathcal{L} = \mathcal{L}_0 \cup \mathcal{L}_{\text{learned}}$ 与失败记忆库 $\mathcal{M}$。
- **三大协作团队架构（§3.2–§3.4, Algorithm 1, Appendix A & F）：**
  1. **Task Proposer Team（任务提议团队）：** 先由 Task Proposer 根据场景上下文、紧凑技能摘要与近期尝试历史生成候选任务池 $\mathcal{T}_t$；再通过 Goldilocks 评分函数 $s(\tau) = \mathcal{N}(\tau)\mathcal{F}(\tau) + w_B B_{\mathrm{retry}}(\tau) - w_P P_{\mathrm{fail}}(\tau)$ 选择“既新颖又处于能力边界（$\bar{r}\approx 0.5$）”的目标（Object-Skill Novelty 与基于 Wilson 成功率下界的 Competence Frontier）；随后由 Environment Creator 生成 LIBERO BDDL 或 MolmoSpaces 桥接任务并由 Environment Verifier 在执行前做确定性/视觉接地校验。
  2. **Execution Team（执行团队，Write-Execute-Verify-Diagnose 闭环）：** 包含 Planner、Planner Verifier、Policy Writer、Quality Checker（静态代码检查）、Goal Verifier、Per-Step Verifier、Failure Diagnoser 以及 SubAgent（在相同复位状态下孤立练习反复卡住的抓取或铰接子动作，产出已视觉验证的局部辅助脚本）。
  3. **Memory-Management Team（记忆管理团队）：** 回合成功后由 Feedback Generator 抽取参数化、自包含函数作为 `experimental` 技能入库；回合失败时记录原始 episode 并提炼 `WHEN ... | WRONG ... | DO ...` 教训。技能采用三层生命周期（`experimental` $\to$ 使用 $\ge 3$ 次且成功率 $\ge 0.5$ 晋升 `verified`；使用 $\ge 10$ 次且成功率 $\le 0.2$ 降为 `deprecated`）。每 $K=5$ 轮由 Memory Curator 合并/去重/重写教训并整理技能，同时由 Skill Proposer 针对高频失败主动拟定候选辅助函数。
- **主实验与消融数值（§4, Table 1–4, Appendix B, C, E）：**
  - **LIBERO-PRO（Table 1，60 任务 × 10 初始状态 = 600 次试验，50 轮 Play）：** OpenVLA 0.0%、$\pi_0$ 0.0%、$\pi_{0.5}$ 12.8%、CaP-Agent0（No Play）23.2%、**RATs 43.8%**（+20.6 pp，其中 Object Pos/Task 分别达到 61.0%/63.0%）。
  - **MolmoSpaces（Table 2，4 类 × 10 任务 × 10 试验 = 400 次试验）：** CaP-Agent0 21.0% vs. **RATs 38.0%**（+17.0 pp；Close 达 73.0%）。391/400 试验调用了学得技能，27 个学得技能中有 14 个被调用共计 5,169 次（Table 9）。
  - **跨环境与 Sim2Real 迁移（Table 3 & Table 8）：** 将 LIBERO-PRO 玩耍学得的冻结技能库直接插入免训练的 `CaP-Agent0`（不微调模型），在未见仿真器 **RoboSuite**（7 任务 × 50 试验 = 350 次）上平均成功率由 40.3% 升至 **49.1%**（+8.9 pp；跨本体双臂抬起 `Two-arm lifting` 从 10.0% 升至 34.0%，+24.0 pp；但 `Two-arm handover` 略降 -4.0 pp，`Nut assembly` 仍为 0.0%）；在真实桌面任务（2 任务 × 40 试验 = 80 次）上由 30.0% 升至 **38.8%**（+8.8 pp）。将 MolmoSpaces 技能库迁移到真实世界 `Swap Cubes` 与 `Close Drawer`（共 60 次试验）上，平均成功率由 3.3% 升至 **25.0%**（+21.7 pp）。
  - **消融与等算力对比（Table 4, Table 12, Table 14）：** LIBERO-PRO 上 Random Play 仅将 CaP-Agent0 从 23.2% 提至 24.7%，而 Curious Play 升至 32.3%，再叠加 RATs Exec. 达 44.3%（5 次试验设置）；在等 Token 算力预算对比中（Table 14，将 play 阶段约 30M tokens 均摊至 60 任务约合测试时多给 5 轮重试），15 轮重试的 CaP-Agent0 仅达 26.0%，低于 10 轮 CaP-Agent0 + RATs 技能库的 32.3%。Token 分析（Table 13，`gpt-5.5`）显示玩耍阶段 10 轮共约 5.12M tokens，开销主要集中在 Failure Diagnoser（40.5%）、Policy Writer（28.8%）与 Failure Memory Distillation（19.4%）。
- **核心局限（§5 Limitations）：** 评估仍以仿真为主，真实物理部署规模有限；玩耍阶段受限于可用仿真环境的物体、动力学与可供性多样性；检索到不匹配任务的技能时可能造成负迁移（如 Two-arm handover -4.0 pp、MolmoSpaces Pick 在插入技能后于 CaP-Agent0 下从 23.0% 降至 14.0%）；推理 Token 开销高、强依赖 VLM 视觉验证，且操作上限受底层抓取/IK 原语能力约束（如 RoboSuite Nut assembly 仍为 0.0%）。
- **对 wiki 的映射：** [Playful Agentic Robot Learning（RATs）](../../wiki/entities/paper-agentic-2606-19419.md)
