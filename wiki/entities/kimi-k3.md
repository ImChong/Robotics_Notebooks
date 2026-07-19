---
type: entity
tags: [foundation-model, llm, moonshot, coding-agents, multimodal, moe, open-source]
status: complete
updated: 2026-07-19
related:
  - ../methods/muon.md
  - ../queries/real-robot-policy-autoresearch-harness.md
  - ../methods/enpire.md
  - ./karpathy-autoresearch.md
  - ./paper-muon-scalable-llm-training.md
  - ../concepts/ai-auto-research.md
sources:
  - ../../sources/blogs/kimi_k3_tech_blog.md
  - ../../sources/courses/kimi_k3_api_quickstart.md
summary: "Kimi K3 是月之暗面 2.8T 参数、1M 上下文、原生视觉的旗舰 MoE：KDA + AttnRes + Stable LatentMoE（896→16），面向长程编码与 agentic 知识工作；API 已上线，完整权重计划 2026-07-27 前开源。"
---

# Kimi K3

**Kimi K3** 是 [月之暗面（Moonshot AI）](https://www.kimi.com/) 2026 年发布的旗舰大模型：**2.8 万亿参数** MoE、**100 万 token** 上下文、**原生视觉**（图 / 视频），架构基于 **Kimi Delta Attention（KDA）** 与 **Attention Residuals（AttnRes）**。它是首个达到 **3T-class** 的**计划开源**模型，定位 **long-horizon coding**、**agentic knowledge work** 与 **reasoning**；对本知识库读者，其价值主要在 **研究型 coding agent**（仿真 / 训练脚本 / benchmark 复现）与 **Muon 训练栈** 的规模化验证，而非直接输出机器人电机指令。

## 一句话定义

以 **KDA + AttnRes + Stable LatentMoE** 支撑 **1M 多模态上下文** 的 **2.8T 旗舰模型**，通过 **Kimi API / Kimi Code / Kimi Work** 提供长程编码与知识工作 agent 能力，并将在 **2026-07-27 前** 发布完整开源权重。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| KDA | Kimi Delta Attention | 混合线性注意力机制，利于长上下文与大规模训练 |
| AttnRes | Attention Residuals | 跨深度选择性检索表示，而非均匀累积 |
| MoE | Mixture of Experts | 稀疏专家路由；K3 为 896 experts、激活 16 |
| MLA | Multi-head Latent Attention | 与 Gated MLA 等组件配合的注意力变体 |
| QAT | Quantization-Aware Training | SFT 阶段起的量化感知训练（MXFP4/MXFP8） |
| API | Application Programming Interface | OpenAI 兼容端点 `https://api.moonshot.ai/v1` |

## 为什么重要

- **开源规模前沿：** 首个 **3T-class** 开源叙事；对需要本地部署或微调的研究团队，权重发布后将成为重要基线（截至 **2026-07-19** 权重**尚未**公开，仅 API / 产品可用）。
- **长程 coding agent：** 博客案例覆盖 **GPU kernel 优化**、**从零编译器（MiniTriton）**、**CAD / 前端 vision-in-the-loop**、**科研代码复现**——与机器人研究中「agent 写训练 / 仿真 / 评测脚本」高度同构（参见 [真机策略 autoresearch 闭环](../queries/real-robot-policy-autoresearch-harness.md)）。
- **训练方法交叉：** K3 使用 **[Per-Head Muon](../methods/muon.md)**、Quantile Balancing、Stable LatentMoE 等，是 Moonshot 在 [Muon 规模化 LLM 训练](../entities/paper-muon-scalable-llm-training.md) 之后的工程延续。
- **评测语境：** [ENPIRE](../methods/enpire.md) 的 AutoEnvBench 已跟踪 **Kimi Code** 系列 coding agent；K3 是同一产品线的旗舰推理后端。

## 核心结构

### 架构主干

```mermaid
flowchart TB
  subgraph arch [Kimi K3 主干]
    E[Embedding]
    B[Transformer Blocks × N]
    KDA[Kimi Delta Attention]
    AR[Attention Residuals]
    MoE[Stable LatentMoE<br/>896 experts → 16 active]
    E --> B
    B --> KDA
    B --> AR
    B --> MoE
  end
  subgraph train [规模化训练要点]
    PHM[Per-Head Muon]
    QB[Quantile Balancing]
    QAT[MXFP4 / MXFP8 QAT]
  end
  arch --> train
```

| 组件 | 作用 |
|------|------|
| **KDA** | 高效注意力基础，利于超长上下文与 prefix caching（vLLM 社区实现将随权重发布） |
| **AttnRes** | 跨层选择性检索，缓解深层信息稀释 |
| **Stable LatentMoE** | 极高稀疏度下稳定路由；相对 Kimi K2 约 **2.5× scaling efficiency** |
| **Per-Head Muon** | 注意力头独立 Muon 优化，见 [Muon](../methods/muon.md) |
| **量化** | SFT 起 QAT，MXFP4 权重 + MXFP8 激活，兼顾硬件兼容 |

### 能力分区（产品视角）

| 场景 | 要点 |
|------|------|
| **Coding** | 长时程仓库级工程、终端工具、截图反馈闭环（游戏 / 前端 / CAD） |
| **Knowledge work** | Kimi Work：多轮研究、交互可视化、Widgets / Dashboard |
| **Multimodal** | 文本 + 图像 + 视频统一输入；motion design / 视频剪辑案例 |

## 工程实践

### API 接入（OpenAI 兼容）

- **模型 ID：** `kimi-k3`
- **Endpoint：** `https://api.moonshot.ai/v1`，密钥 `MOONSHOT_API_KEY`
- **Thinking：** **始终开启**；使用 `reasoning_effort="max"`（当前唯一级别），**勿**使用 K2.x 的 `thinking` 参数
- **多轮 / 工具：** 必须把 API 返回的 **完整 assistant message**（含 `reasoning_content`、tool_calls）原样追加到下一轮
- **视觉：** `content` 为对象数组；图像 **base64** 或 `ms://<file_id>`；**不支持公网 image URL**
- **缓存：** 长 system / 知识库前缀 **自动 context caching**；coding 场景官方称 cache hit **>90%**

### 与机器人研究 harness 的衔接

| 需求 | 建议 |
|------|------|
| **仿真 / 训练脚本 autoresearch** | 优先 **Kimi Code** 或自建 harness，确保 **thinking history 完整回传**（官方局限 #1） |
| **长仓库 + 工具环** | API `tool_choice`、dynamic tool loading；参考 [karpathy/autoresearch](./karpathy-autoresearch.md) 的固定 eval 契约 |
| **真机策略 agent** | 先满足 ENPIRE 式 **reset + verify**；再选 coding backend（见 [ENPIRE](../methods/enpire.md)） |
| **行为边界** | K3 可能 **过度主动**；在 system prompt / `AGENTS.md` 写明禁止擅自改实验假设或跳过验证 |

### 部署与推理（权重发布后）

- 博客建议 **≥64 加速器 supernode** 部署；大 expert-parallel 域利于吞吐。
- **KDA prefix caching** 实现将贡献 **vLLM**；自托管前关注社区发布节奏。

## 局限与风险

| 局限 | 说明 |
|------|------|
| **权重未开源（截至 2026-07-19）** | 博客与 API 文档承诺 **2026-07-27 前** 发布 full weights；入库日无 GitHub / HF 链接 → 当前仅 API / 产品可用 |
| **Thinking history 敏感** | 中途换模型或 harness 丢 thinking 会导致质量崩溃；勿在长跑 session 中无验证切换 backend |
| **相对闭源 UX gap** | 官方承认仍落后于 Claude Fable 5、GPT 5.6 Sol 的体验 |
| **Web search 工具** | API 文档称近期更新中，**不建议生产依赖** |
| **非具身动作模型** | K3 不直接输出机器人关节 / 航点；物理任务需接 VLA / 控制栈或专用具身模型 |

## 开源状态

| 项目 | 状态（2026-07-19） |
|------|-------------------|
| **API / Kimi Code / Kimi Work** | 已上线 |
| **完整模型权重** | **待发布**（截止 **2026-07-27**） |
| **技术报告** | 待发布 |
| **vLLM KDA caching** | 计划随权重发布 |
| **训练代码** | 博客未承诺完整训练栈开源 |

## 参考来源

- [Kimi K3 技术博客归档](../../sources/blogs/kimi_k3_tech_blog.md)
- [Kimi K3 API Quickstart 归档](../../sources/courses/kimi_k3_api_quickstart.md)
- [Kimi K3 官方技术博客](https://www.kimi.com/blog/kimi-k3)
- [Kimi K3 API Quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)

## 关联页面

- [Muon](../methods/muon.md) — K3 训练使用 Per-Head Muon
- [Muon is Scalable for LLM Training](./paper-muon-scalable-llm-training.md) — Moonshot Muon 规模化论文
- [真机策略 autoresearch 闭环搭建指南](../queries/real-robot-policy-autoresearch-harness.md) — coding agent 选型与 harness 前提
- [ENPIRE](../methods/enpire.md) — AutoEnvBench 与 Kimi Code 评测语境
- [autoresearch（karpathy/autoresearch）](./karpathy-autoresearch.md) — 固定预算 LLM 实验环结构可迁移
- [AI Auto-Research](../concepts/ai-auto-research.md) — 研究自动化阶段论

## 推荐继续阅读

- [Kimi K3 Pricing（官方）](https://platform.kimi.ai/docs/guide/kimi-k3-pricing)
- [Kimi API Platform 文档索引](https://platform.kimi.ai/docs/llms.txt)
- [Moonshot Muon 规模化论文（arXiv:2502.16982）](https://arxiv.org/abs/2502.16982)
