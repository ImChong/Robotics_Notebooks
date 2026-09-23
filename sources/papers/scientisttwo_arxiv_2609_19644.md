# ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI（arXiv:2609.19644）

> 来源归档（ingest）

- **标题：** ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI
- **类型：** paper / ai-auto-research / multi-agent / autonomous-discovery / benchmark
- **arXiv abs：** <https://arxiv.org/abs/2609.19644>
- **PDF：** <https://arxiv.org/pdf/2609.19644>
- **Hugging Face Papers：** <https://huggingface.co/papers/2609.19644>
- **项目页：** <https://scientist-two.github.io/> — 归档见 [`sources/sites/scientist-two-github-io.md`](../sites/scientist-two-github-io.md)
- **机构：** Google Cloud AI Research（Jaehyun Nam、Jinsung Yoon、Yanzhou Pan、Rui Meng、Parthasarathy Ranganathan、Tomas Pfister）；Yubo Wang（University of Waterloo）
- **入库日期：** 2026-09-23
- **一句话说明：** **问题驱动** 的全自主 multi-agent 研究框架：以顶会已接受论文为挑战，自动复现 SOTA、提出并验证新想法、跑 ablation、写稿，并经 **模拟 peer-review + rebuttal 实验** 闭环；在 **107** 个人类 SOTA 任务上 **80.4%** 相对提升，生成稿在 ScholarPeer / Stanford Agentic Reviewer 下评分高于 ICLR/NeurIPS 接受稿均值。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://scientist-two.github.io/> | 86 篇生成论文预览、integrity audit、domain 分布 |
| 前作 | ScientistOne / AI Scientist 系列 | 自主研究 agent 谱系 |
| 本库概念 | [ai-auto-research](../../wiki/concepts/ai-auto-research.md) | 学术研究自动化总览 |

## 摘要级要点

- **愿景：** 人类给出 **科学问题**，AI 独立导航知识前沿、诊断瓶颈、执行 **端到端发现周期**（idea→code→exp→paper→review）。
- **相对 prior agent 缺口：** 多数系统优化 **单数据集单指标**；缺 **多 benchmark 推理**、系统 **ablation**、动态 **rebuttal 补实验**。
- **ScientistTwo 机制：** Idea generator → subset/full benchmark evaluator → idea evolution → ablation → drafting → peer-review → rebuttal experiments → meta-review；各 stage 有 **critic/refine** 环（见论文 Table 1）。
- **Integrity（CoE Audit）：** score verification、spec compliance、reference verification、method–code alignment；完整系统 **49/49** 论文通过四维审计（0/1814 幻觉引用）。
- **结果（论文报告）：** 107 挑战中 86 篇 beat human SOTA（+25.2% 平均相对增益）；ScholarPeer 7.5/10（91.9% accept）；Stanford Agentic Reviewer 5.7/10（72.1% accept）。
- **领域：** LLM、robotics、neuroscience、speech、robustness、RL、game theory、privacy、optimization、time series 等。

## 核心摘录（面向 wiki 编译）

### 1) 问题形式化

给定科学问题 $\mathcal{G}$，输出论文 $\mathcal{P}^+$ 与可复现代码库 $\mathcal{C}^+$，超越人类 SOTA baseline。

### 2) 与机器人读者关系

- robotics 为 **评测域之一**，非专为人形/控制设计；选型时勿与 embodied 控制论文混读。
- 对本库价值：**AI Auto-Research** 栈的 S1–S7 闭环实例，可与 [Hermes Agent](../../wiki/entities/hermes-agent.md)、[karpathy/autoresearch](../../wiki/entities/karpathy-autoresearch.md) 对照。

### 3) 开源状态（项目页，2026-09-23）

| 组件 | 状态 |
|------|------|
| 项目页 / 生成论文预览 | 已公开 |
| 统一开源训练框架 | **未列单一 GitHub** — 以生成 codebase 与网页预览为主 |
| 复现 | 论文强调 per-paper 可执行脚本；非传统单仓库 release |

## 对 wiki 的映射

- 新建：[paper-scientisttwo](../../wiki/entities/paper-scientisttwo.md)
- 更新：[ai-auto-research](../../wiki/concepts/ai-auto-research.md) 交叉引用

## 当前提炼状态

- [x] arXiv + 项目页核查
- [x] 开源边界：生成 codebase + 网页，无 monorepo
- [ ] 若官方发布 harness 仓库再补 `sources/repos/`
