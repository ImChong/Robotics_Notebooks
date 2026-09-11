# SoL-Pi 项目页（NVLabs）

> 来源归档

- **标题：** SoL-Pi: Scaling Auto-Research Loops for Efficient Agent Harnesses
- **类型：** site / research blog
- **链接：** <https://nvlabs.github.io/SoL-Pi/>
- **代码：** <https://github.com/NVlabs/SoL-Pi>
- **机构：** NVIDIA（NVLabs）
- **入库日期：** 2026-09-11
- **一句话说明：** NVIDIA 用规模化 auto-research 环在 Pi coding-agent harness 上搜索 token 效率机制；152 条候选中 4 条机制（Action Fusion、Online Context Compact、ObservationPack、Evidence-Preserving Reducer）组成 SoL-Pi，EdgeBench 上约保留 Pi 94% 分数、token/成本降约 45–64%。
- **沉淀到 wiki：** 是 → [`wiki/entities/sol-pi.md`](../../wiki/entities/sol-pi.md)

## 开源状态（步骤 2.5）

- **项目页 Footer / 正文：** 链到 [NVlabs/SoL-Pi](https://github.com/NVlabs/SoL-Pi) 四个 extension 子目录（action-fusion、online-context-compact、observation-pack、evidence-preserving-reducer）。
- **结论：** **已开源**（MIT；Pi 扩展插件，非 Pi 官方发行版）。README 给出 `pi install git:github.com/NVlabs/SoL-Pi` 与配置文档 `docs/configuration.md`。
- **未列：** SoL-Pi 专用 arXiv 预印本（页面引用 Karpathy autoresearch、EdgeBench 等外部链接，非本工作论文条目）。

## 项目页要点（归纳）

### 问题与动机

- 长时程 coding agent 轨迹中，token 是否都在推进任务，还是冗余随长度增长？
- **递归自改进（RSI）** 本身也耗 token；在放大 RSI 前，能否先让 **harness 更省**？
- 研究基底：**Pi**（轻量可扩展 coding-agent harness）；评测：**EdgeBench**（51 任务、长时程 held-out）与 **Terminal-Bench 4**（63 CPU 任务）。

### 方法：规模化 auto-research

- **152** 条机制候选 → **4** 条 survive；六大家族：Context / Progress / Tools / Delegation / Prompt & policy / Improvement & evaluation。
- 每条 lineage：**Trajectory Rollouts → Map–Reduce Analysis → Proposal → Implementation（Ralph Loop）→ Reviewer → In-Trajectory Validation → Held-Out Validation**。
- **Oracle Analysis** 在 rollout 前筛机会；训练/held-out **完全隔离**。
- **535** 个可执行训练环境（495 GitHub issue–PR 轨迹 + 40 verifier 合成）；EdgeBench 数据 **不** 进入搜索。
- **Capability floor：** 能力指标在容忍带内 + 至少一项效率指标改进；禁止「少做活来省钱」。
- 编排演进：compiled workflow → code orchestration → **disposable skill loop**（单次实验实例化模板、结束即弃）。

### 四条存活机制

| 机制 | 作用 |
|------|------|
| **Action Fusion** | 编辑/写入与后续验证命令 **单次 tool call** 本地融合，省中间 model turn |
| **Online Context Compact** | 子任务完成边界触发 compaction，而非仅全局 context 压力 |
| **ObservationPack** | 大 tool 输出归档为 handle + 分页精确 recall，避免全文 replay |
| **Evidence-Preserving Reducer** | 廉价 agent 读长 log，frontier 只收 **可逐行核验** 的 receipt |

### 主要结果（页面数字）

- EdgeBench：**~94%** Pi 平均分（两 backend）；相对 Pi token **−45–49%**、成本 **~−33%**；相对原生 Codex/Claude Code harness token **−35–64%**、API 等价成本 **−50–54%**（`xhigh` reasoning）。
- Terminal-Bench 4（63 CPU）：SoL-Pi **15/63** solved，总成本低于 Codex/Pi，per-solved **$14.07**。
- **Agent swarm：** Anthropic original performance take-home 上，Sol + 20 SoL-Pi workers vs stock Pi swarm：**−17.5%** cycles、**−26.8%** cost（单次非随机对照，页面自述因果不可估）。

### 讨论要点

- Token 效率指向 **跨任务可复用** 的交互浪费，不易 benchmark 过拟合。
- **Breadth-first** 搜索（多 lineage 并行）比 depth-only 更易跳出局部 basin；约 **1/40** 起点 idea  survive validation。
- 人类供 **先验与筛 idea**；loop 内无人工；survivor 由人类 **理解并 refactor** 成可维护实现。

## 对 wiki 的映射

- 升格实体页：[SoL-Pi](../../wiki/entities/sol-pi.md)
- 交叉补强：
  - [karpathy/autoresearch](../../wiki/entities/karpathy-autoresearch.md) — 页面显式引用其 experiment loop
  - [HarnessBank](../../wiki/entities/paper-harnessbank.md) — 同为 harness 自进化/效率，门控与语义银行对照
  - [AI Auto-Research](../../wiki/concepts/ai-auto-research.md) — S3 实验环 + 人机共治
  - [递归自改进](../../wiki/concepts/recursive-self-improvement.md) — 「先 efficiency 再 RSI」叙事

## 参考来源（原始）

- 项目页：<https://nvlabs.github.io/SoL-Pi/>（2026-09-11 抓取）
- 代码仓 README：<https://github.com/NVlabs/SoL-Pi>
