# InternVLA-A1.5：Unifying Understanding, Latent Foresight, and Action for Compositional Generalization

> 来源归档（ingest）

- **标题：** InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2607.04988>
- **PDF：** <https://arxiv.org/pdf/2607.04988>
- **机构：** 上海人工智能实验室（Shanghai AI Laboratory）Physical Intelligence Team；核心贡献者含 Haoxiang Ma、Junhao Cai、Xiaoxu Xu 等；通讯 Weinan Zhang
- **前作：** InternVLA-A1（统一未来视觉状态与动作）、InternVLA-M1（多模态共训语料）
- **骨干：** **Qwen3.5-2B** VLM + **460M** unified expert（MoT 共享 full attention）
- **世界模型教师：** 冻结 **WAN2.2-5B**（训练期 latent foresight 监督，推理丢弃）
- **预训练规模：** **1.2M** 机器人 episode（**861M** 帧）+ **3M** 多模态样本（InternVLA-M1）
- **入库日期：** 2026-07-11
- **一句话说明：** 在原生 VLM chat 模板上 **持续 VQA/子任务共训**，用 **50 个 foresight token** 向冻结视频生成器查询 **紧凑潜式未来**，再以 **flow matching** 输出连续动作块；**训练用世界模型、部署不滚像素**，六套仿真全榜领先，真机 **组合指令 OOD 绑定** 与 **长程 MOF** 显著优于 π₀.₅ / Motus。

## 核心摘录（面向 wiki 编译）

### 1) MoT 统一架构：理解 + 潜式前瞻 + 连续动作

- **链接：** arXiv §2 InternVLA-A1.5 Model Design；Figure 2
- **摘录要点：**
  - **Mixture-of-Transformers（MoT）**：预训练 **Qwen3.5-2B** 作 VLM 骨干（3×Gated DeltaNet + 1×full attention 交错）；**460M unified expert** 同构但更小 hidden dim，仅经 **共享 full attention** 与 VLM 交互。
  - 输入保持 VLM **原生 chat 模板**：多视角图像、语言指令、**离散化本体状态**（256 bin）、**控制模态 token**（`<end_effector>` / `<joint>` 等）。
  - Stage 1：VLM 预测 **子任务描述 + FAST 离散动作 token**（2048 词表并入 VLM vocabulary）。
  - Stage 2：unified expert 追加 **50 个 learnable foresight tokens** + **action query tokens**；动作由 **flow matching** 解码连续 chunk（默认 **H=50**）。
  - **推理：** 视频生成分支 **完全丢弃**；单步约 **0.1s**（RTX 5090，静态图 + SDPA + flash linear attention）。
- **对 wiki 的映射：**
  - [InternVLA-A1.5](../../wiki/entities/paper-internvla-a15-unified-vla.md) — 架构表与 MoT 流程图
  - [VLA](../../wiki/methods/vla.md) — π₀.₅ 式离散 FAST + flow 双阶段对照
  - [π₀.₇](../../wiki/methods/pi07-policy.md) — 同类「VLM 原生模板 + 子任务分解」提示结构

### 2) Latent foresight：冻结 WAN 监督、非像素级自训

- **链接：** arXiv §3.2 Foresight reasoning；Figure 4
- **摘录要点：**
  - 将未来预测改写为 **latent querying**：foresight tokens 经 unified expert 读出 **紧凑潜码** \(C_t^f\)，替换 WAN2.2 原 T5 文本编码器作 **cross-attention 条件**。
  - 对每个 action chunk 均匀采样 **N=4** 未来帧；在 WAN-VAE 潜空间做 **flow-matching 视频损失** \(\mathcal{L}_{video}\)，梯度 **只回传 foresight token 与上游 expert**，WAN **全程冻结**。
  - 与 InternVLA-A1 / Being-H0 等「从零学像素未来」不同：策略 **继承预训练视频生成器的时空动力学先验**，但 **从不学习像素级生成**。
  - 消融：去掉 \(\mathcal{L}_{video}\) 或 foresight tokens 在 LIBERO-Plus / DOMINO 零样本上跌幅最大（Table 8）。
- **对 wiki 的映射：**
  - [InternVLA-A1.5](../../wiki/entities/paper-internvla-a15-unified-vla.md) — foresight 机制 Mermaid
  - [World Action Models](../../wiki/concepts/world-action-models.md) — 「训练耦合世界模型、部署不想象」族谱
  - [Being-H0.7](../../wiki/methods/being-h07.md) — 另一类潜空间未来监督对照

### 3) 三阶段训练与注意力掩码

- **链接：** arXiv §3 Training Recipe；Figure 5
- **摘录要点：**
  - **Stage 1（300K steps, bs=1024）**：机器人 + VQA 混合；联合 CE 监督 **答案 / 子任务 / FAST 动作**（式 1–2，自然自回归条件 \(\hat\ell \rightarrow a\)）。
  - **Stage 2（600K steps）**：\(\mathcal{L}_{stage2} = \mathcal{L}_{stage1} + \alpha \mathcal{L}_{video} + \beta \mathcal{L}_{action}\)，\(\alpha=1, \beta=10\)。
  - **Posttrain（60K steps, bs=128, cosine decay）**：同 Stage 2 配方，下游微调可选保留视频 分支。
  - **Group-wise causal + 组内双向**：foresight 组看 VLM 上下文；action noise 组看 VLM + foresight；**训练时 mask unified expert 对 FAST token 的注意力**，防泄漏与梯度干扰；推理复用 VLM KV cache，flow 去噪只更新 action 组。
  - 机器人:多模态采样比 **0.15:0.85**，强化 VLM 语义不被动作目标侵蚀。
- **对 wiki 的映射：**
  - [InternVLA-A1.5](../../wiki/entities/paper-internvla-a15-unified-vla.md) — 训练阶段表
  - [Action Chunking](../../wiki/methods/action-chunking.md) — H=50 chunk 与异步部署语境

### 4) 数据：1.2M 机器人 episode + InternVLA-M1 3M QA

- **链接：** arXiv §4 Data Recipe；Figure 6
- **摘录要点：**
  - 机器人六源：**InternData-A1**（合成，396M 帧）、AgiBotWorld、UMI、DROID、Galaxea、RoboMind 1.0；统一进 InternVLA-A1 **共享动作空间**（形态槽位 padding）。
  - 每 episode 三重角色：连续动作监督 flow head、未来帧监督 foresight、FAST 离散目标监督 VLM。
  - 多模态 **InternVLA-M1 ~3M**：General QA 637K + Box/Point/Trajectory QA（机器人空间 grounding）。
  - **两级分组采样**：组内 \((\#\text{frames})^\gamma\)，组间 Re-Mix + 手工上调小真机源权重。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md) — 异构数据源与采样权重
  - [RoboTwin](../../wiki/entities/robotwin.md) — 双臂评测基准语境

### 5) 仿真六榜与真机组合泛化

- **链接：** arXiv §5 Experiments；Tables 2–8；Figures 7–9
- **摘录要点：**
  - **仿真（均报告最佳或极强竞争力）：** SimplerEnv avg **80.8%**；LIBERO **98.9%**；LIBERO-Plus total **84.8%**（零样本）；RoboTwin 2.0 avg **93.2%**（clean 93.3 / rand 93.0）；DOMINO 零样本 SR **27.7%**（微调 **29.3%**）；EBench Test SR **35.2%**。
  - **真机四任务：** Sort/Insert/Move Tubes（**held-out 指令绑定** 组合泛化）+ **MOF** 13 步子长程化学流程。
  - vs **π₀.₅ / Motus**：held-out 绑定上 **A1.5 全胜**；MOF **76.4%** vs π₀.₅ **29.3%**（Motus 0%）；Insert/Move 精度任务领先 π₀.₅ **20.8 / 7.8 pt**。
  - **RoboTwin SFT 效率：** 同等 60K 步微调，A1.5 损失下降最快（Figure 10）。
- **对 wiki 的映射：**
  - [InternVLA-A1.5](../../wiki/entities/paper-internvla-a15-unified-vla.md) — 结果表与真机任务设计
  - [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — 同赛道 RoboTwin / LIBERO 对照

### 6) 局限与工程启示

- **链接：** arXiv §6 Conclusion and Limitations
- **摘录要点：**
  - **局限 1：** foresight 仅覆盖 **一个 action chunk 短视界**，尚无长程想象/显式规划。
  - **局限 2：** WAN 冻结且通用，embodied 场景先验上界受预训练覆盖限制。
  - **启示：** 把状态/动作 cast 进 **VLM 原生 chat + 单一 CE** 可显著稳定训练；**少量 latent token 查询冻结生成器** 即可吸收动力学，无需从零学像素视频。
- **对 wiki 的映射：**
  - [World Action Models](../../wiki/concepts/world-action-models.md) — 短视界 latent foresight vs 部署期像素 roll 讨论

## 当前提炼状态

- [x] arXiv HTML 全文已对齐摘录（2607.04988）
- [x] wiki 映射：`wiki/entities/paper-internvla-a15-unified-vla.md` 新建
- [ ] 待官方代码/权重公开后补仓库与 LeRobot 集成链
