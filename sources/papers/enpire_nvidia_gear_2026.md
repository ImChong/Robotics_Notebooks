# ENPIRE：Agentic Robot Policy Self-Improvement in the Real World

> 来源归档（ingest）

- **标题：** ENPIRE: Agentic Robot Policy Self-Improvement in the Real World
- **类型：** paper（公开材料以项目页为主；arXiv 待发布）
- **机构：** NVIDIA GEAR Lab；CMU LeCAR Lab；UC Berkeley（项目页致谢与公开报道）
- **原始链接：**
  - 项目页：<https://research.nvidia.com/labs/gear/enpire/>
  - 站点归档：[`sources/sites/nvidia-research-enpire.md`](../sites/nvidia-research-enpire.md)
- **入库日期：** 2026-06-18
- **一句话说明：** 把 **真机策略自改进** 抽象成 coding agent 可管理的 **闭环优化**：自动 reset/verify 的环境接口 + 多 PI 范式 + 并行 rollout + 跨假设演化，在多种 **高灵巧桌面操作** 上达到 **约 99% pass@8**，并系统评测 frontier coding agent 与 **机器人机队 scaling**。

## 核心论文摘录（MVP）

### 1) 问题：真机灵巧操作仍高度依赖人工监督与手工算法工程

- **链接：** <https://research.nvidia.com/labs/gear/enpire/>（Abstract）
- **摘录要点：** 通用物理智能的瓶颈之一，是 **人类监督 + 算法手工调参** 仍贯穿真机策略开发；数字环境里 coding agent 能自动搜代码，但机器人缺 **可重复的真机反馈环**：reset → execute → verify → refine。
- **对 wiki 的映射：**
  - [ENPIRE](../../wiki/methods/enpire.md) — 把「物理 AutoResearch」作为独立抽象写清楚。

### 2) 系统：EN / PI / R / E 四模块闭环

- **链接：** <https://research.nvidia.com/labs/gear/enpire/>（ENPIRE System）
- **摘录要点：**
  - **Environment：** `env.py` 暴露 reset、observation、reward、step；人类用户与 coding agent 经 **Tool APIs** 调用；底层接感知/规划/控制。
  - **Policy Improvement：** agent 可切换 **启发式、BC、offline/online RL、code-as-policy** 等路线；可读文献、改 infra、做 param sweep。
  - **Rollout：** 预算化真机 trial，产出日志与视频供分析。
  - **Evolution：** 多 agent 分支比较，保留提升团队平均成功率的 idea（页面 **hypothesis git-tree** 可视化）。
- **对 wiki 的映射：**
  - [ENPIRE](../../wiki/methods/enpire.md) — Mermaid 主干流程与模块边界。

### 3) 环境工程：自动评测 + 自动 reset 是真机 autoresearch 的前置条件

- **链接：** <https://research.nvidia.com/labs/gear/enpire/>（From Robot Hardware to an Agent-Operable Environment）
- **摘录要点：**
  - **Auto Evaluation：** 例：扎带插入用 **检测 + 分割**（代码片段示例含 SAM3、boundlsdf）在多相机上独立判定后融合二值 reward。
  - **Auto Reset：** Push-T、插针、扎带、GPU 插入等任务均展示 **随机初始态 + reset 行为 + 成功验证**；无人工复位则无法高密度 trial。
- **对 wiki 的映射：**
  - [ENPIRE](../../wiki/methods/enpire.md) — 与 [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md) 的「评测接口」读点。

### 4) 主结果：多任务 pass@8 ≈ 99% 与 agent 驱动 recipe 搜索

- **链接：** <https://research.nvidia.com/labs/gear/enpire/>（Learned Manipulation Policy / Agents Improve Policies）
- **摘录要点：** Push-T、Pin Insertion、GPU Insertion、Tie/Cut Ziptie 等任务上策略达 **99% pass@8**；页面展示从 0% 启发式 Push-T 到在线 RL 混合 demo、BC 正则、batch size 等 **逐步 +pp** 的 idea 时间线。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md) — 灵巧操作任务与 agent 闭环范例。

### 5) AutoEnvBench 与机队 scaling：MRU / MTU

- **链接：** <https://research.nvidia.com/labs/gear/enpire/>（Evaluate Coding Agent / Scaling Autoresearch on Robot Fleets）
- **摘录要点：**
  - **AutoEnvBench** 跟踪 **Codex GPT-5.5、Claude Code Opus 4.7、Kimi Code K2.6** 在 Push-T（启发式）与 Pin Insertion（梯度学习）上的 **墙钟研究曲线**。
  - **1 / 4 / 8 agent** 团队在两类任务上的 scaling；提出 **MRU** 与 **MTU** 衡量机器人与 token 利用率；更大 fleet 可更快成功但 **token-to-success 上升**。
- **对 wiki 的映射：**
  - [ENPIRE](../../wiki/methods/enpire.md) — 与 [具身规模法则](../../wiki/concepts/embodied-scaling-laws.md) 的「物理侧 scaling」对照。

### 6) 仿真评测：RoboCasa 任务补充

- **链接：** <https://research.nvidia.com/labs/gear/enpire/>（Evaluation in Simulation）
- **摘录要点：** 在 RoboCasa 厨房操作任务上评测，以 **更密 ablation** 分离 agent 研究行为与真机硬件吞吐限制。
- **对 wiki 的映射：**
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md) — 真机闭环 vs 仿真密集评测的分工。

## 当前提炼状态

- [x] 摘要、四模块系统、自动 reset/verify、主任务结果、AutoEnvBench、fleet scaling、RoboCasa 仿真与局限已摘录到可维护粒度
- [ ] arXiv 号发布后补充 HTML/PDF 锚点与 BibTeX
- [ ] 若官方发布代码仓，应单独 `sources/repos/` 索引并回链

## 对 wiki 的映射（汇总）

- [ENPIRE](../../wiki/methods/enpire.md)
- [Manipulation](../../wiki/tasks/manipulation.md)
- [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
