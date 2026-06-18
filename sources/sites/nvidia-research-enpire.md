# NVIDIA Research — ENPIRE（GEAR Lab）

> 来源归档（ingest）

- **标题：** ENPIRE: Agentic Robot Policy Self-Improvement in the Real World
- **类型：** site（官方项目页）
- **发布方：** NVIDIA GEAR Lab；合作方含 CMU LeCAR Lab、UC Berkeley（项目页致谢与公开报道）
- **原始链接：** <https://research.nvidia.com/labs/gear/enpire/>
- **入库日期：** 2026-06-18
- **一句话说明：** 面向 **coding agent** 的物理世界 **AutoResearch** 束具：用 **Environment / Policy Improvement / Rollout / Evolution** 四模块把「重置场景 → 真机 rollout → 自动验证 → 读日志改代码」闭合成可编排优化过程，在 Push-T、插针、GPU 插拔、扎/剪扎带等灵巧任务上报告 **pass@8 约 99%** 成功率，并给出 **AutoEnvBench** 与 **MRU/MTU** 等多机队 scaling 指标。

## 摘录要点（与论文分工）

- **核心主张：** 数字环境里 coding agent 已能自动搜算法，但机器人侧缺的是 **可重复的真机策略改进反馈环**——reset、执行、验证、再 refine。
- **四模块缩写 ENPIRE：**
  - **EN（Environment）** — 自动 reset + 自动 verification（`env.py` 式接口：`reset` / `get_reward` / `get_observation` / `step`）。
  - **PI（Policy Improvement）** — 启动策略精炼；支持启发式、tool calling、BC、offline/online RL 等 **PI regime**。
  - **R（Rollout）** — 单/多真机并行评估，记录 state/action/video/result。
  - **E（Evolution）** — agent 读日志、查文献、改训练基建与算法代码，跨分支比较并采纳有效 recipe（公开报道强调经 **Git** 协调）。
- **任务演示：** Push-T、Pin Insertion、GPU Insertion、Tie Ziptie、Cut Ziptie；页面含 **idea git-tree**（每 agent 一分支、每节点一假设）与团队平均成功率曲线对齐展示。
- **自动评测样例（扎带插入）：** 检测框 + 分割（页面代码示例引用 **SAM3**）→ 多相机独立判定 → 融合为二值 reward。
- **自动 reset：** 各任务展示多组随机初始态与 reset 成功验证帧。
- **AutoEnvBench：** 比较 **Codex（GPT-5.5）**、**Claude Code（Opus 4.7）**、**Kimi Code（K2.6）** 在 Push-T（启发式）与 Pin Insertion（梯度学习）上的 **墙钟研究进展**。
- **机队 scaling：** 1 / 4 / 8 agent 团队在 Push-T 与 Pin Insertion 上的 scaling 曲线；提出 **MRU（Mean Robot Utilization）** 与 **MTU（Mean Token Utilization）** 衡量多 agent 物理 autoresearch 效率。
- **仿真补充：** RoboCasa 上若干厨房操作任务（Coffee Setup Mug、Open Cabinet 等）用于分离 agent 研究行为与真机吞吐瓶颈。
- **局限（页面原文）：** agent 读日志/写代码/等待 LLM 时 **机器人与算力欠利用**；机队扩大后 **MRU 下降、GPU 利用率上升**，但 **token 消耗随 fleet 增大**。

## 论文 / 代码状态

- 截至入库日，arXiv 检索 **尚无** `ENPIRE` 条目；公开材料以 **NVIDIA GEAR 项目页** 为主（媒体报道称 2026-06-16 上传研究论文，待 arXiv 号发布后应回链 [`sources/papers/enpire_nvidia_gear_2026.md`](../papers/enpire_nvidia_gear_2026.md)）。
- 项目页未给出公开 GitHub 仓库链接。

## 对 wiki 的映射

- [ENPIRE](../../wiki/methods/enpire.md) — 四模块闭环、PI regime、AutoEnvBench 与 fleet scaling 读点
- [Manipulation](../../wiki/tasks/manipulation.md) — 灵巧真机任务与 agent 驱动策略迭代交叉引用
- [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md) — RoboCasa 侧补充评测与真机闭环的对照
