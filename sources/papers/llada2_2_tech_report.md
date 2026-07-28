# LLaDA2.2 Technical Report（官方技术报告）

> 原始资料归档（ingest）

- **标题：** LLaDA2.2: Enabling Agentic Diffusion Language Models via Levenshtein Editing
- **类型：** paper / technical report（官方 PDF；截至入库日未见独立 arXiv id）
- **作者 / 机构：** Inclusion AI · 蚂蚁数字科技（Ant Digital Technologies）· 浙江大学 · 西湖大学 / Westlake Scitrain（通讯作者含 Zhenzhong Lan、Jianguo Li、Junbo Zhao、Da Zheng）
- **PDF：** <https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf>
- **页数 / 体积：** 11 页 · ~686 KB
- **关联权重：** <https://huggingface.co/inclusionAI/LLaDA2.2-flash>
- **关联仓：** <https://github.com/inclusionAI/LLaDA2.X>
- **前序：** LLaDA2.0（arXiv:2512.15745）、LLaDA2.1（arXiv:2602.08676）
- **入库日期：** 2026-07-28
- **一句话说明：** 在 block-diffusion MoE dLLM 上引入 **Levenshtein editing**（KEEP / SUBSTITUTE / DELETE / INSERT）与 **L-EBPO** agentic RL，并把上下文扩到 **128K**、改用 **block routing**；发布 **LLaDA2.2-flash（100B）** 的 agentic 评测与吞吐结果。

## 开源与可用性（与报告同步核查，2026-07-28）

| 项 | 状态 |
|----|------|
| **技术报告** | **已发布** · 仓内 `LLaDA2_2_tech_report.pdf` |
| **权重** | **已开源** · HF / ModelScope `inclusionAI/LLaDA2.2-flash`（Apache-2.0） |
| **入口仓** | [inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X)（README + 多版 tech report；**无训练脚本**） |
| **推理 / 微调生态** | [dInfer](https://github.com/inclusionAI/dInfer)、[dFactory](https://github.com/inclusionAI/dFactory)；HF 卡称 **SGLang 对 2.2「coming soon」** |
| **训练数据 / 全栈 RL** | 报告描述 CPT / SFT / L-EBPO 与 ASystem / AKernel / SGLang 管线，**未随 LLaDA2.X 仓发布可复现训练代码** |

## 核心摘录（归纳，非全文）

### 1) 一句话主张（Abstract）

dLLM 的 block 并行解码在多轮长程 agent 场景易误差累积；LLaDA2.1 的固定长度 T2T 编辑不够用。LLaDA2.2 用 LCS 对齐草稿与真值得到四元编辑标签，并提出 **L-EBPO** 用环境奖励优化编辑决策；同时 **128K** 上下文 + **block routing** 控制 MoE 激活集合。经验上 agentic 基准与 AR 基线 **Ling-2.6-flash** 接近，吞吐更高。

### 2) 训练流水线（Figure 2）

| 阶段 | 内容 |
|------|------|
| **CPT** | 自 LLaDA2.1 8K base：先 **300B / 64K**，再 **200B / 128K**（末段加入 agent 数据）；路由从 token-level → **block routing**（默认块容量 \(C=48\)，\(E=256\)） |
| **SFT** | 块内 Levenshtein editing；DELETE 删位、INSERT 在当前位置前插入 `[MASK]` 槽；块长固定（pad / truncate） |
| **RL** | **L-EBPO**：动作空间 \(A = V \cup \{\mathrm{DELETE}, \mathrm{INSERT}\}\)；外层轨迹 + 内层块内编辑；奖励 = 工具正确性 + 格式 + 任务完成；rollout 经 SGLang，环境经 AKernel 沙箱 |

### 3) 评测要点（§4）

- **Agentic 7 项均分：** LLaDA2.2-flash **53.83** vs Ling-2.6-flash **55.74**；胜出 τ²-Bench / PinchBench / MCP-Atlas；SWE 系列落后（SWE-bench Verified 脚手架不一致：Claude Code vs OpenHands）。
- **General 10 项均分：** **56.81** vs Ling **65.90**；LongBench v2 反而更高（45.13 vs 42.94）。
- **吞吐：** 11 负载 BF16 平均约 **1.64×** Ling；再 FP8 约再 +18.6%。
- **消融：** 同 SWE 轨迹仅开关 Levenshtein → SWE-bench Verified **35.8 → 44.4**（+8.6）。

### 4) 局限（§5）

块内并行仍弱于 AR 的局部一致性（嵌套 JSON / SQL / 多参工具调用）；INSERT 比 DELETE 更难学；EBPO 似然近似与 MoE 训练–推理路由差仍在；通用能力与 agent 数据配比未定。

### 5) 与机器人研究的映射读法

报告主线是 **agentic coding / tool-use dLLM**，不是具身策略；对本库价值在：**(a)** 长程 coding / SWE agent 后端的 **非自回归** 选项；**(b)** 与 [Diffusion Model](../../wiki/concepts/diffusion-model.md) 概念页的 **离散文本扩散** 交叉；**(c)** autoresearch harness 选型时与 AR 旗舰（如 [Kimi K3](../../wiki/entities/kimi-k3.md)）对照吞吐与结构化输出风险。

## 对 wiki 的映射

| 目标 | 说明 |
|------|------|
| [LLaDA2.2-flash](../../wiki/entities/llada2-2-flash.md) | 主升格实体：规格、开源、编辑机制、评测与部署 |
| [Diffusion Model](../../wiki/concepts/diffusion-model.md) | 补充离散 dLLM / block diffusion 指针 |
| [真机策略 autoresearch 闭环](../../wiki/queries/real-robot-policy-autoresearch-harness.md) | coding backend 吞吐型备选 |
| [Kimi K3](../../wiki/entities/kimi-k3.md) | 同为开放权重 coding/agent 后端对照 |

## 外部参考

- [GitHub inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X)
- [HF inclusionAI/LLaDA2.2-flash](https://huggingface.co/inclusionAI/LLaDA2.2-flash)
- [LLaDA2.1 arXiv:2602.08676](https://arxiv.org/abs/2602.08676)
- [LLaDA2.0 arXiv:2512.15745](https://arxiv.org/abs/2512.15745)
