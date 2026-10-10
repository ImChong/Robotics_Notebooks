# RSIAgent: Autonomous Exploration for Recursive Self-Improvement in New Environments（Aether AI）

> 来源归档（blog / Aether AI 官方 + 论文 / 项目页 / 代码元数据）

- **标题：** RSIAgent lets open-source models explore new environments—and surpass GPT-6 Astra on OSWorld 2.0 and Agents' Last Exam（博客标题）；论文题为 *RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments*
- **类型：** blog（附 arXiv 论文、项目页、GitHub 仓库）
- **作者 / 组织：** Sibo Zhu†、Shicheng Fan†、Xinyue Wang†、Wenyi Wu、Kun Zhou*（通讯 / 项目负责人）、Biwei Huang；Aether AI，联合 UCSD、UIC（项目页标注前四位作者为 Aether AI 实习期间工作）
- **原始链接：** <https://aetherlabs.ai/articles/rsiagent-autonomous-exploration-for-recursive-self-improvement.html>
- **论文：** [arXiv:2609.15364](https://arxiv.org/abs/2609.15364)（cs.AI；交叉 cs.CL、cs.CV；50 页）
- **项目页：** <https://aetherlabsai.github.io/RSIAgent/>
- **代码：** <https://github.com/AetherLabsAI/RSIAgent>（Apache-2.0）
- **创始人推文：** <https://x.com/huang_biwei/status/2099664633095401659>（Biwei Huang；按推文 ID 推算约 2026-09-15 01:00 UTC；仅取得页面 meta 摘要，正文被截断）
- **发表日期：** 博客页署名 **2026-09-15**；arXiv v1 **2026-09-14 10:46 UTC** 提交，v2 **2026-09-18**；README 称 2026-09-15 进入 Hugging Face Daily Papers 第 6 位（2026-09-16 核对，自报）
- **入库日期：** 2026-10-10
- **抓取方式：** `curl` 抓取官方博客、项目页静态 HTML、arXiv abs 页、`raw.githubusercontent.com` 上的 `README.md` 与 `docs/PAPER.md`、项目页 `data/benchmark-results.csv` 与 `data/ablation.csv`；GitHub API 在本会话不可用，未取得 star / release 数据；论文 PDF 全文未读
- **一句话说明：** Aether AI 提出 **不更新参数** 的数字智能体自改进框架：curriculum / actor / verifier 三角色围绕可演化记忆循环，先 **并行广度探索（BRS）** 再 **串行深度探索（DRS）**，把「动作–条件–结果」因果经验固化为冻结记忆后复用；以 Kimi-K3 + GLM-5.3 为底座，在 **OSWorld 2.0（0808 offline，82 题）partial 78.98%**、**Agents' Last Exam Near-term（67 题）partial 84.82%**，高于 GPT-6 Astra 的 72.60% / 82.26%（均为自报，非对齐预算对比）。评测全部是 **软件 / 计算机使用** 环境，**无机器人实验**。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-10-10） |
|----|-------------------------|
| 论文 | **已公开**：arXiv:2609.15364（v1 2026-09-14，v2 2026-09-18）；项目页另提供作者 PDF 镜像 |
| 代码 | **已开源**：`AetherLabsAI/RSIAgent`，Apache-2.0；含 `run_osworld.py`、`run_ale.py` 两个批处理入口与 `core/`、`explore/`、`benchmarks/` 等模块；需自备 `OPENROUTER_API_KEY`、Linux + Docker + `/dev/kvm` |
| 权重 | **不适用**：方法免训练，调用现成模型 API（Kimi-K3、GLM-5.3 等），无自有权重 |
| 数据 | 项目页提供 `benchmark-results.csv`、`ablation.csv`、`citation.bib`；探索得到的记忆库是否公开 **未见说明** |
| 关联项目 | RSIGame（<https://github.com/WenyiWU0111/RSIGame>，arXiv:2609.39045），把 RSI 扩到自主游戏开发 |
| 可信度边界 | 公司博客 + arXiv 预印本，非同行评审；所有分数为 **自报**，对比基线多取自各家官方报告 / 榜单，项目页明说 **非匹配预算比较、非实时榜单** |

## 核心摘录（归纳，非全文）

### 动机

- 新软件 / 新工具有隐藏约束、异常状态与失败模式；强推理模型也未必懂环境的因果结构（哪些动作导致哪些结果、什么条件触发失败）。
- 微调 / RL / 人类反馈需要采集与标注数据，企业软件、私有环境中数据难公开、频繁重训不现实。
- 问题：**不更新参数**，智能体能否像人一样探索新环境、发现因果规律、固化经验并持续改进？

### 方法

- **三角色 + 共享记忆：** Curriculum Agent 决定下一步探索什么并判断是否还需练习；Actor Agent 用可执行 Python / Bash 程序加视觉观测操作软件（code as policy），验证后由同一 Actor 蒸馏经验并与已有记忆调和；Verifier Agent 独立检查任务要求与环境结果，**看不到** Actor 的私有推理与记忆。
- **Stage 1 · BRS（Broad Recursive Self-exploration）：** 类比预训练。Curriculum 每轮生成多方向探索任务，多组 actor–verifier 并行执行、验证，经验写入共享记忆，再生成新方向任务。默认名义预算 **8 个项目、最多 4 个并行**，在整波完成后检查预算并按顺序合并记忆。
- **Stage 2 · DRS（Deep Recursive Self-exploration）：** 类比后训练。从较难任务出发，根据刚暴露的失败、未知与弱点设计更难任务，串行推进；直到 Curriculum 判定无需进一步练习。博客称其类似自动化「压力测试」。
- **Phase 3 · 测试期记忆复用：** 记忆冻结，Curriculum 与记忆更新关闭，环境重置后 Actor + Verifier 用同一 harness 执行；官方评分保持在学习环外。
- **默认配置（项目页）：** Actor 用 **GLM-5.3**，Verifier 与 Curriculum 用 **Kimi-K3**（独立上下文）；**Curriculum 在两个探索阶段都能看到目标 query**。
- 记忆内容：过程、脚本、失败教训；论文与博客强调「动作–条件–结果」的因果关系。

### 主结果（自报，partial / binary 均为 %）

| 模型 / 方法 | OSWorld 2.0 Partial | OSWorld 2.0 Binary | ALE Near-term Partial | ALE Near-term Binary |
|-------------|--------------------:|-------------------:|----------------------:|---------------------:|
| Kimi-K3 | 58.30 | — | 71.60 | 40.30 |
| Claude Opus 5 | 70.19 | 34.72 | 79.54 | 46.27 |
| GPT-5.6 Sol | 64.13 | 28.10 | 78.82 | 47.76 |
| GPT-6 Astra | 72.60 | — | 82.26 | **52.24** |
| RSIAgent（w/o RSI） | 71.97 | 37.80 | 83.75 | 49.25 |
| **RSIAgent** | **78.98** | **42.68** | **84.82** | 50.75 |

- 完整对比表还含 Kimi-K2.6、MiMo-V2.5、DeepSeek V4 Pro、Qwen3.8-Max、Claude Opus 4.8、Claude Fable 5、Gemini-3.8-Flash、Muse Spark 1.3（见项目页 CSV）。
- RSI 增益：OSWorld partial **+7.01**、binary **+4.88**；ALE partial **+1.07**、binary **+1.50**（全对 33/67 → 34/67）。
- ALE binary 上 RSIAgent（50.75）**低于** GPT-6 Astra（52.24）；GPT-6 Astra 的 OSWorld binary 未报告。
- **聚合口径：** OSWorld 82 题（T082 安装失败计 0），RSI 行只有 **41 题** 为实际 RSI 评测，其余 41 题沿用基线分；ALE 67 题中 **19 题** 为 RSI 分、**48 题** 沿用基线，含本地修正评分、ECG 公共标签迁移、无 BRS 的 Tax Form 变体。RSI 列含挑选的重试与不同预算的 checkpoint，**不是匹配重复运行的平均**。对比数值取自论文 2026-09-11 的来源快照。

### 消融与轮次（自报）

| 任务 | w/o RSI | w/o DRS | w/o BRS | Full RSI（两次评测均值） |
|------|--------:|--------:|--------:|------------------------:|
| T080 WPS 表格修复 | 0.0500 | 0.5546 | 0.6030 | 0.6869 |
| T085 REAPER 音频编辑 | 0.6800 | 0.9107 | 0.6006 | 0.9415 |
| T089 浏览器演示文稿修复 | 0.6750 | 0.7700 | 0.6400 | 0.8125 |
| T106 3D Slicer 肝脏分割 | 0.3601 | 0.3855 | 0.4166 | 0.5406 |

- 这 4 题是 **按已记录改进挑选的探索性子集**（docs/PAPER.md）；deep-only 条件从空记忆开始且最多 2 个练习项目，与发布版默认设置不同。
- RSI 轮次增加时性能持续提升；简单任务数轮收敛，部分初始 0–40% 的任务最终达 100%（代表性任务，自报）。
- **案例 T085（REAPER）：** partial 68.00 → 94.17；BRS 7 个项目覆盖选源、拼接、静音测量、变调变速；DRS 发现重采样导致样本不一致，修正记忆中的 `RENDER_RESAMPLE` 设置，测试期直接复用。

### 游戏开发扩展（GameCraft-Bench，自报 Overall）

| 生成器 | Baseline | + Play2Code | + RSIAgent（w/o RSI） | + RSIAgent |
|--------|---------:|------------:|----------------------:|-----------:|
| Codex + GPT-5.5 (high) | 52.77 | 51.05 | 57.84 | 61.28 |
| Kimi-K2.6 | 31.28 | 36.02 | 42.61 | 46.37 |
| GLM-5.3-Flash | 30.55 | 38.25 | 44.73 | 48.72 |

### 失败分析（README 归纳）

- 练习可能没打到真正弱点（insufficiently targeted exploration）。
- 验证器可能接受不完整的工作（incomplete verification）。
- 记忆可能保存错误规则（unreliable memory consolidation）。

### 团队脉络（博客第 03 节）

- Causal-Copilot（自主因果分析智能体，2025）→ C-World（计算机使用环境生成器，2026）→ Auto-scaling Continuous Memory（2025）/ Hybrid Self-evolving Structured Memory（2026）→ StructAgent（统一因果结构的长程数字智能体，2026）→ RSIAgent。
- 主张 **Scaling Experience**：参数固定，靠更多 RSI 轮次扩大探索并把因果规律沉淀进可演化记忆，作为模型规模之外的另一条扩展路径。

## 对 wiki 的映射

- [aether-rsiagent](../../wiki/entities/aether-rsiagent.md) — 本篇升格实体页
- [aether-ai](../../wiki/entities/aether-ai.md) — 公司入口页
- [aether-cris-0](../../wiki/entities/aether-cris-0.md) — CRIS-0 机器人系统的「Causality-guided Robot Agent」；媒体称其智能体层前序研究为 RSIAgent（RSIAgent 材料本身不含机器人内容）
- [recursive-self-improvement](../../wiki/concepts/recursive-self-improvement.md) — RSI 概念页

## 可信度与使用边界

- 「超越 GPT-6 Astra」只在 partial 指标上成立；ALE binary 未超越；对比非同预算、非同 harness。
- 从 Kimi-K3 单模型（58.30）到 RSIAgent w/o RSI（71.97）的 +13.67 来自多智能体 harness 与代码动作，RSI 本身贡献 +7.01；ALE 上 RSI 贡献仅 +1.07。
- Curriculum 在探索期可见目标 query，记忆针对目标任务环境构建，属于 **测试期针对性练习**，与零样本泛化不可直接比较。
- 无机器人或物理环境实验；迁移到具身系统的主张需另见 CRIS-0 材料。

## Citation

```bibtex
@misc{zhu2026rsiagentautonomousexplorationrecursive,
      title={RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments},
      author={Sibo Zhu and Shicheng Fan and Xinyue Wang and Wenyi Wu and Kun Zhou and Biwei Huang},
      year={2026},
      eprint={2609.15364},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.15364},
}
```
