# AeroWeaver: An Embodied-Agent Harness for Weaving Aerial Skills into Distributed, Adaptive Swarm Execution

- **作者：** Jiabin Lou, Yirong Yang, Haopeng Wang, Xuxin Lv, Xinyu Liu, Diyuan Hou, Xuehong Liu, Rongye Shi, Wenjun Wu
- **机构：** 北京航空航天大学（Beihang University）
- **版本 / 载体：** arXiv:2609.18520v1 [cs.AI]，2026 年 9 月 16 日提交（8 pages, 7 figures）；HTML 全文归档
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2609.18520) · [arXiv HTML 全文](https://arxiv.org/html/2609.18520)
- **项目页 / 代码：** 已开源 — 官方仓库 [Admire-ljb/AeroWeaver](https://github.com/Admire-ljb/AeroWeaver)（MIT 许可证，含 Mock / AirSim / PX4-Gazebo 适配器与 Web 控制台，另见 [项目主页](https://admire-ljb.github.io/AeroWeaver/) 与 [仓库核查归档](../repos/paper-agentic-2609-18520.md)）
- **核对范围：** 通读 arXiv HTML 全文全部章节（Abstract、§I Introduction、§II Related Work、§III-A/B/C Method、§IV-A/B/C/D Experiments 含 Table I 主实验表与 Table II 消融表、Fig. 1–7 图注、§V Conclusion 与 References [1]–[32]）及官方仓库 README 与目录样本
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 将 LLM 智能体应用于无人机蜂群时，面临三大系统性缺口：（1）语义决策如何稳定落地为绑定到具体载具的可执行能力（semantic-to-physical grounding）；（2）全局任务推理如何与分布式的机载观测、通信链路和执行器协同；（3）如何利用任务执行期积累的经验实现无重训在线适配。
- **三层机制（§III）：**
  1. **Embodied-Agent Harness with Aerial Skills（§III-A）：** 每个技能表示为 $\sigma_i = \langle n_i, d_i, I_i, O_i, P_i, E_i, M_i \rangle$，将 `skill.md` 文档与可执行对象配对；Mission Parser 从指令 $q$ 和蜂群状态 $s_t$ 提取结构化任务记录 $\chi^t$ 并检索任务级目录 $\mathcal{A}^t$；Commander 分配角色 $\rho_k$ 与局部目标 $g_k$，按角色检索 Top-K 激活局部技能子集 $\mathcal{L}_k$。
  2. **Distributed Swarm Orchestration（§III-B）：** 每个智能体 $k$ 绑定至单架无人机 $u_k$，基于隔离上下文 $c_k^t$、局部观测 $o_k^t$ 与通信半径 $r_{\mathrm{comm}}$ 内邻居的消息 $\{m_{\ell \rightarrow k}^{t-1}\}$ 并发决策，Commander 不具备直接飞行控制权，本地角色报告经 Commander 与任务控制台汇总给用户。
  3. **Experience-Guided Online Reinforcement（§III-C）：** 可调用的“技能-参数”对分配单 token 标签，以首 token 对数概率为基线分数 $z_i$（每次最多请求 20 个备选 token 分数）；按精确任务与角色过滤历史记录后，基于分词状态描述的 Jaccard 重合度检索最多 32 条经验，计算角色条件下的技能优势估计 $\widehat{A}(s, \sigma_i) = \bar{G}_{i,\rho}(s) - \bar{G}_{\rho}(s)$，并通过 $z_i' = z_i + \beta \widehat{A}(s, \sigma_i)$ 在线修正排序，执行后以 $\gamma=0.95$ 累积折扣回报并回写角色索引记忆库。
- **实验与关键边界（§IV–V）：** 在 AirSim 仿真环境（$140 \times 140\text{ m}$ 固定高度、$0.5\text{ s}$ 步长、最多 24 轮、22 个任务技能、DeepSeek V4-flash 主骨干）上的 9 个 MPE 启发任务（每组 $n=10$ 回合）中，AeroWeaver 均取得高于 Centralized 与 HMAS-2 的平均回合回报，Token 消耗较 HMAS-2 降低 52.0–70.0%；消融表明“协调报告”与“技能激活”是核心贡献，奖励修正（$\beta=0.8$）在所有 9 个任务上提供额外增益。当前评估限于预定义 22 技能目录与仿真运行条件，尚未开展实体真机飞行验证。
- 对 wiki 的映射：[AeroWeaver 论文实体页](../../wiki/entities/paper-agentic-2609-18520.md)
