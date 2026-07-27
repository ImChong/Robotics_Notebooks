# Kimi K3 Technical Report（官方技术报告）

> 原始资料归档（ingest）

- **标题：** Kimi K3: Open Frontier Intelligence — Technical Report of Kimi K3
- **类型：** paper / technical report（官方 PDF，非 arXiv）
- **作者：** Kimi Team（月之暗面 / Moonshot AI）
- **PDF：** <https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf>
- **镜像入口：** GitHub 仓根目录；HF README 亦链至该 PDF
- **页数 / 体积：** 47 页 · ~2.5 MB（CreationDate 2026-07-27 UTC）
- **关联权重：** <https://huggingface.co/moonshotai/Kimi-K3>
- **关联博客：** <https://www.kimi.com/blog/kimi-k3>
- **入库日期：** 2026-07-27
- **一句话说明：** Kimi K3 官方技术报告：2.8T MoE（激活 **104B**）、KDA + AttnRes + Stable LatentMoE、预训练 / 后训练 RL、基础设施与评测；宣布 **完整开放权重**。截至入库日 **无 arXiv id**。

## 开源与可用性（与报告同步核查）

- **权重：** 报告脚注指向 HF `moonshotai/Kimi-K3` — **已开源**。
- **报告本身：** 随 [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) **已发布**。
- **训练代码 / 数据：** 报告描述方法与基础设施，**未附可复现训练仓**。

## 核心摘录（归纳，非全文）

### 1) 一句话主张（Abstract）

引入 **2.8T** MoE（**104B** 激活）、原生视觉、**1M** 上下文；架构基于 **KDA** 与 **AttnRes**，配合 **Stable LatentMoE（16/896）** 与训练配方，相对 Kimi K2 约 **2.5× scaling efficiency**。后训练强调 general / agentic / coding 域 RL 与多档 **reasoning effort**。评测达 frontier-level，整体仍落后 Claude Fable 5 与 GPT-5.6 Sol，但优于套件内其余开源与多数闭源对照。**完整权重已释放**。

### 2) 报告结构（目录）

| 章 | 主题 |
|----|------|
| 1 | Introduction |
| 2 | Model Architecture（Hybrid Attention、AttnRes、Stable LatentMoE、MoonViT-V2、Per-Head Muon） |
| 3 | Pre-Training |
| 4 | Post-Training |
| 5 | Infrastructure（KDA 系统共设计、全平衡 EP、百万 token agentic RL、部署） |
| 6 | Evaluations |
| 7 | Case Studies |
| 8 | Conclusion |
| App. | Contributions；SiTU-GLU；Quantile Balancing 推导 |

### 3) 架构要点（§2）

- **块结构：** 每 block **3× KDA + 1× Gated MLA**，每注意力层后接 Stable LatentMoE。
- **AttnRes：** 用可学习 pseudo-query 对 embedding 与前序 block 输出做选择性加权。
- **视觉：** MoonViT-V2 → projector → 共享 embedding。
- **优化：** **Per-Head Muon**（注意力头独立正交化更新）。

### 4) 与机器人研究的映射读法

- 报告主线是 **LLM / agentic coding**，不是具身策略；对本库价值在：**(a)** 长程 research coding agent 后端；**(b)** Muon / MoE 训练方法交叉；**(c)** 1M 上下文下的文献 + 代码闭环案例。

## 对 wiki 的映射

| 目标 | 说明 |
|------|------|
| [Kimi K3](../../wiki/entities/kimi-k3.md) | 用报告规格（104B 激活、69+24 注意力组成、MoonViT-V2）刷新实体页 |
| [Muon](../../wiki/methods/muon.md) | Per-Head Muon 在 2.8T 的官方论述 |
| [ENPIRE](../../wiki/methods/enpire.md) / [autoresearch harness](../../wiki/queries/real-robot-policy-autoresearch-harness.md) | coding / agentic 评测与 harness 语境 |

## 外部参考

- [GitHub MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
- [HF moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- [技术博客](https://www.kimi.com/blog/kimi-k3)
