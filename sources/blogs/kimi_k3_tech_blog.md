# Kimi K3: Open Frontier Intelligence（技术博客）

> 原始资料归档（ingest）

- **标题：** Kimi K3: Open Frontier Intelligence
- **类型：** blog（Moonshot AI / Kimi 官方技术博客）
- **组织：** 月之暗面（Moonshot AI）
- **原始链接：** <https://www.kimi.com/blog/kimi-k3>
- **入库日期：** 2026-07-19
- **一句话说明：** Kimi K3 是 **2.8T 参数**、**1M 上下文**、**原生视觉** 的旗舰 MoE 模型，基于 **KDA + AttnRes + Stable LatentMoE**；面向长程编码、知识工作与推理，并计划 **2026-07-27 前** 开源完整权重。

## 开源与可用性（截至 2026-07-19 核查）

- **API / 产品：** 已在 Kimi.com、Kimi Work、Kimi Code、Kimi API 上线；默认 **max thinking effort**。
- **权重：** 博客与 API 文档均写明 **full model weights will be released by July 27, 2026**；截至入库日 **尚未发布** GitHub / Hugging Face 权重链接。
- **技术报告：** 架构、训练与评测细节将随 **Kimi K3 technical report** 发布。
- **推理生态：** 正与 inference partners 与开源维护者对齐；**KDA prefix caching** 实现将贡献给 **vLLM** 社区，随模型一并发布。

## 核心摘录（归纳，非全文）

### 1) 规模与架构

| 维度 | 要点 |
|------|------|
| **参数量** | **2.8T**（首个开源 **3T-class** 模型） |
| **注意力** | **Kimi Delta Attention（KDA）** + **Attention Residuals（AttnRes）** — 改善长序列与深层信息流动 |
| **MoE** | **Stable LatentMoE**：896 experts，每 token 激活 **16**；相对 K2 约 **2.5× scaling efficiency** |
| **多模态** | 原生视觉；文本 / 图像 / 视频统一建模 |
| **上下文** | **1M tokens** |
| **量化** | SFT 起 **QAT**：**MXFP4 权重 + MXFP8 激活** |
| **训练技巧** | **Quantile Balancing**（专家分配）、**Per-Head Muon**、**SiTU**、**Gated MLA**；全平衡 expert-parallel、静态 shape、关键路径无 host sync |
| **部署建议** | 推荐 **≥64 加速器 supernode**；大 EP 域利于推理效率 |

### 2) 能力叙事（相对竞品）

- 整体仍 **落后于最强闭源**（文内点名 Claude Fable 5、GPT 5.6 Sol），但在评测套件中 **frontier-level**，且 ** consistently outperform other tested models**。
- **Coding：** 长程工程会话、大仓库导航、终端工具编排；**vision-in-the-loop**（截图反馈）用于游戏、前端、CAD。
- **Knowledge work：** Kimi Work 内 agentic 知识工作流；Widgets / Dashboard 持久化交互组件。

### 3) 编码案例研究（博客亮点）

| 案例 | 摘要 |
|------|------|
| **Kernel optimization** | 24h sandbox 内优化 AttnRes / KDA / MLA kernel（H200 + 备选 GPGPU）；与 Fable 5（含 fallback）竞争，显著优于 Opus 4.8 / GPT 5.6 Sol / GPT 5.5 |
| **MiniTriton** | 从零构建类 Triton 编译器（tile IR + MLIR passes + PTX codegen）；roofline 上媲美或超越 Triton / torch.compile；nanoGPT 端到端训练收敛稳定 |
| **Game / digital creation** | 概念 / 图 / 视频 → 可玩交互；代码与实时截图迭代 |
| **Chip design（PoC）** | 48h 自主 run：开源 EDA + Nangate 45nm，4 mm² @ 100 MHz，模拟 **>8,700 tok/s** decode |
| **Coding for research** | ~2h 复现天体物理 I–Love–Q 关系（通常 1–2 周）；20+ 论文、300+ EoS、3000+ 行 Python、交互 HTML dashboard |

### 4) 知识工作案例

- **42 年 ASIC 产业** 交互研究站（120+ 轮自改进、2.8k+ 搜索、11k+ 页 PDF）
- **聚变产业** 咨询式报告 + 交互可视化
- **GWTC-5** 引力波事件分析（20+ 子 agent）
- **视频 editing：** 3Blue1Brown 风格架构讲解动画；56 片段 teaser 精剪（节拍同步、多轮修订）

### 5) 局限（官方）

1. **Thinking history 敏感：** 训练于 preserved thinking history；harness 未完整回传历史 thinking 或中途换模型会导致质量不稳定 → 推荐 **Kimi Code** 等已验证 harness。
2. **过度主动：** 长程难任务训练使 minor issue 时可能替用户做意外决策 → 需在 system prompt / `AGENTS.md` 加行为边界。
3. **UX gap：** 相对 Fable 5 / GPT 5.6 Sol 仍有明显体验差距。

### 6) 定价与 API（博客 Availability 节）

- Kimi API `kimi-k3`：**$0.30/MTok** cache-hit input、**$3.00/MTok** cache-miss input、**$15.00/MTok** output。
- Mooncake 分离式推理；官方 API 在 coding workload 上 cache hit **>90%**。

## 对 wiki 的映射

| 目标 | 说明 |
|------|------|
| [Kimi K3](../../wiki/entities/kimi-k3.md) | 旗舰模型实体页（架构、API、开源状态、机器人研究相关性） |
| [Muon](../../wiki/methods/muon.md) | K3 训练栈使用 **Per-Head Muon** |
| [真机策略 autoresearch 闭环搭建指南](../../wiki/queries/real-robot-policy-autoresearch-harness.md) | Kimi Code / K3 作为 coding agent 选型参考 |
| [ENPIRE](../../wiki/methods/enpire.md) | AutoEnvBench 已评测 Kimi Code 系列 agent |

## 外部参考

- [Kimi K3 技术博客](https://www.kimi.com/blog/kimi-k3)
- [Kimi K3 API Quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)
- [Kimi API Platform](https://platform.kimi.ai/)
