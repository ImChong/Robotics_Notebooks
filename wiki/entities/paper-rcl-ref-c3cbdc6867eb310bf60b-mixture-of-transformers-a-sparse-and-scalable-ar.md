---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
venue: "Transactions on Machine Learning Research"
summary: "提出按模态（文本/图像/语音）解耦所有非嵌入 Transformer 参数并保留全局自注意力的稀疏架构 MoT，在等 FLOPs 下将 Chameleon/Transfusion 预训练算力开销降为 33%–55.8%、墙钟时间降为 47.2%–75.6%（TMLR 2025）。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_ref_c3cbdc6867eb310bf60b_mixture-of-transformers-a-sparse-and-sca.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Mixture-of-Transformers

**Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models（MoT）** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 079/564** 篇，分组 **Components of WAMs（Language & vision-language backbones）**。本页基于 TMLR 04/2025 原文全文（`arXiv:2411.04996v2`）完成结构化精读升级。

## 一句话定义

**Mixture-of-Transformers（MoT）** 是一种面向早期融合（early-fusion）多模态基座模型的稀疏 Transformer 架构：按 token 所属模态确定性地解耦全部非嵌入参数（注意力投影 $Q/K/V/O$、FFN 与 LayerNorm），同时保留跨全序列的全局自注意力交互，在维持单 token 推理/训练 FLOPs 与稠密模型完全一致的前提下大幅加速多模态预训练。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoT | Mixture-of-Transformers | 按模态解耦全部非嵌入参数的稀疏多模态 Transformer |
| MoE | Mixture-of-Experts | 基于可学习路由激活部分专家 MLP 的稀疏架构 |
| PpF | Parameter-to-FLOPs Ratio | 参数量与浮点计算量比值，决定分布式通信开销 |
| LOO | Leave-One-Modality-Out | 留一模态分离消融实验，验证各模态解耦收益 |
| WAM | World Action Model | 世界预测与动作生成耦合模型 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- **直击早期融合多模态模型的模态冲突与扩展瓶颈**：原文 PCA 潜空间分析显示，即便稠密多模态模型以统一离散 token 处理图文音，各模态在潜空间中仍自然聚类成不同区域并存在优化冲突；简单引入学习路由 `MoE-4x` 易导致负载不均、分布漂移退化与墙钟时间变慢（图像训练慢 1.7×）。
- **兼顾统一自回归与混合目标（AR + Diffusion）**：MoT 既可直接作为 Chameleon（全离散自回归）及三模态（文本+图像+语音）底座的即插即用替代，又能在 Transfusion（文本自回归 + 图像连续潜块扩散）下以一半 FLOPs 让 760M MoT 超越 1.4B 稠密基线。
- **为 WAM / VLA 多模态底座提供低通信开销设计参考**：对于需要同时处理视觉观察、语言指令与连续/离散动作流的世界–动作模型，MoT 证明了「模态专用权重 + 全局注意力」的范式，相关条目见 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 与 [World Action Models（WAM）](../concepts/world-action-models.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 079/564 |
| 分组 | Components of WAMs |
| 作者与机构 | Weixin Liang*（Stanford University）、Lili Yu†、Liang Luo†、Srinivasan Iyer、Ning Dong、Chunting Zhou、Gargi Ghosh、Mike Lewis、Wen-tau Yih、Luke Zettlemoyer、Xi Victoria Lin（AI at Meta） |
| 出处 | Transactions on Machine Learning Research (TMLR, 04/2025) · `arXiv:2411.04996v2` |
| 论文 | <https://openreview.net/forum?id=Nu6N69i8SB> |
| 官方代码 | <https://github.com/facebookresearch/Mixture-of-Transformers> |
| 子类 / 象限 | Language & vision-language backbones · 语言与VLM Backbone · 已开源（Playbook） |

## 核心机制（归纳）

> 注：以下机制与评测指标基于通读原文全文（TMLR 04/2025 / `arXiv:2411.04996v2`）核对补充。

### 1. 按模态确定性解耦全部非嵌入参数与全局自注意力（§2.2, Algorithm 1）

- **模态索引分组而非学习路由**：给定交错多模态序列 $x = (x_1, \dots, x_n)$（每个 token 带有已知模态标签 $m_i \in \{\text{text}, \text{image}, \text{speech}\}$），MoT 通过确定性索引 $I_m = \{i : m_i = m\}$ 将 token 分组，彻底绕开 MoE 的可学习路由器、负载均衡辅助损失及双层优化不稳定性。
- **全组件解耦与共享注意力空间**：
  1. **模态专用 Q/K/V 投影**：各模态分别通过 $W_Q^m, W_K^m, W_V^m$ 映射 $X_m$，随后按原始序列位置重组为统一的 $Q, K, V$；
  2. **跨模态全局自注意力**：在共享特征空间对全部模态 token 执行因果/混合全局自注意力 $A = \text{softmax}(QK^T / \sqrt{d_k})V$；
  3. **模态专用输出与前馈**：注意力输出按模态拆分，经模态专用 $W_O^m$、$\text{LayerNorm}_{\text{attn}}^m$、$\text{FFN}^m$ 与 $\text{LayerNorm}_{\text{ffn}}^m$ 完成残差更新。
- **等 FLOPs 控制**：对于每个 token，激活的计算路径与标准稠密 Transformer 完全同构，因此训练与推理 FLOPs 与同等激活参数规模的稠密模型完全一致。

### 2. 多范式目标兼容与组件/留一消融（§3.2–§3.5, §4–§5）

- **三种评测范式**：
  1. **Chameleon（离散自回归图文）**：图像经 VQ-VAE/VQGAN 离散化为 1,024 tokens，图文统一自回归训练（37M–7B）；
  2. **Chameleon+Speech（文本+图像+语音）**：加入基于 DinoSR 变体的 25 Hz（40ms/token，词表 500）语义语音离散 token（443M–7B，语音与图文按 1:6 采样）；
  3. **Transfusion（混合目标）**：文本采用离散自回归目标，图像经 8×8 下采样 VAE 编码为连续潜块并在序列中以 BOI/EOI 包裹、外加固定 0.27B U-Net 补丁编解码，采用 250 步扩散目标联合训练（163M–7B，训练 0.5T tokens）。
- **组件解耦与 LOO 发现（§3.5, §4, §5）**：在 880M 模型上，仅解耦 FFN 已大幅降损；再解耦注意力 $Q, K, V$ 可在 Obelisc 上为图像和文本分别进一步节省约 **33.3%** 和 **10%** FLOPs；在此基础上解耦 LayerNorm 的边际影响极小（Fig. 14）。三模态留一消融（LOO，443M）表明任意合并两种模态都会拉高对应损失（非对称模态竞争，Fig. 15）。将 MoT 文本塔的 FFN 替换为 4 专家 MoE（`MoT + Text MoE-4x`，§5）可进一步加速文本收敛而不损害图像优势。

### 3. 系统吞吐与分布式扩展特性（§6）

- **低 PpF 通信优势（§6.1）**：在 SwiGLU FFN 架构下，每层新增参数对比为 MoE $3(E-1)D^2 + ED$ vs MoT $7(K-1)D^2$；由于模态数 $K$ 通常远小于专家数 $E$，MoT 在 FSDP 分布式云训练中的 Parameter-to-FLOPs（PpF）更低，缓解网络带宽瓶颈，且随着 GPU 数量从 16 增至 256 加速比进一步扩大（§6.2.1）。

### 方法流程图

```mermaid
flowchart LR
    IN["交错多模态 Token 序列 (Text / Image / Speech)"] --> IDX["按模态确定性索引分组 I_m"]
    IDX -->|"Text / Image / Speech 各用独立 W_QKV^m"| QKV["拼接还原原序列 Q, K, V"]
    QKV --> ATTN["共享特征空间全局自注意力 GlobalAttn"]
    ATTN -->|"按模态拆分输出"| POST["模态专用 W_O^m + LayerNorm^m + FFN^m"]
    POST -->|"Chameleon: AR 预测 / Transfusion: AR+Diffusion"| OUT["多模态联合生成输出"]
```

## 评测与指标

- **Chameleon 7B（自回归图文，FLOPs 受控，§3.2.2 & Fig. 5）**：
  - 训练损失：MoT 仅用 **45.5%** 训练步数（60k 步即匹配稠密 120k 步）达到稠密基线最终总训练损失；图像模态仅需 **34.8%** 步数。
  - 验证集损失（Obelisc / MS-COCO / Flickr30k / Shutterstock）：MoT 使用 **55.8%** 训练 FLOPs 即可全面匹配或优于稠密模型最终验证损失（节省 **44.2%** FLOPs），且在 37M–7B 全尺度保持一致优势（而 `MoE-4x` 图像优势在 7B 消失，Fig. 6）。
- **Chameleon+Speech 7B（文本+图像+语音，§3.3.2 & Figs. 8–9）**：
  - 语音模态：7B MoT 仅用 **22.9%** 训练步数匹配稠密语音训练损失（443M–1.5B 尺度为 **15.1%–33.6%**），在 LL60K 与 PPL30K 验证集上仅需 **37.2%** FLOPs 即追平稠密基线，同时图像与文本仍保持 **55.8%** 步数收益。`MoE-4x` 在语音验证集上泛化不稳定甚至劣于稠密模型。
- **Transfusion（自回归文本 + 扩散图像，§3.4.2–§3.4.3 & Figs. 10–12）**：
  - **7B 规模**：MoT 用不到 **1/3（~30%）** FLOPs 匹配稠密图像训练/验证损失；在 `CFG=1.6` 下 COCO-30k 零样本 FID 达 **8.14**（对比 1T tokens 训练的稠密模型 **9.22**）；图像描述（Captioning）CIDEr 达 **0.406** vs 稠密 **0.315**（超 3× 加速）。
  - **760M MoT vs 1.4B 稠密（仅一半训练/推理 FLOPs，Fig. 11）**：760M MoT 全面超越 1.4B 稠密基线——CLIP 分数 **0.214** vs **0.206**（↑）、FID **21.145** vs **24.688**（↓）、CIDEr **0.320** vs **0.286**（↑）；163M/760M/1.4B 三尺度上图像训练损失匹配稠密仅需 **17.5%–26.4%** 步数。
- **系统墙钟时间与横向扩展（AWS `p4de.24xlarge` A100，256 GPUs + FSDP，§6.2 & Figs. 18–19）**：
  - **7B Chameleon 墙钟时间（Fig. 19）**：MoT 达到稠密图像训练质量仅耗时 **47.2%**（验证集 **58.4%**），文本仅耗时 **75.6%**（验证集 **74.9%**）；对比之下 `MoE-4x` 在图像训练上反而产生 **1.7×**（验证集 **2.0×**）墙钟减速。
  - **GPU 横向扩展（443M Chameleon，16 → 256 GPUs，Fig. 18）**：MoT 追平稠密 Obelisc 验证损失所需步数比例，图像从 **42.1%** 降至 **21.6%**，文本从 **75.7%** 降至 **50.9%**。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Components of WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页基于原文补全核心机制与评测数字。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**Mixture-of-Transformers（MoT）证明：在原生多模态基座中按已知模态确定性地解耦全部非嵌入权重、并保留全局自注意力，可以在不增加单步推理/训练 FLOPs 的前提下大幅缓解模态冲突，将 7B 多模态预训练的 FLOPs 与墙钟时间压缩至稠密基线的 33%–56% 与 47%–76%。**

- **规则模态路由优于学习路由 MoE**：确定性按模态索引完全避免了 Expert Choice 等学习路由在低频模态（如语音）和分布偏移下的负载失衡、验证过拟合与散射通信开销。
- **FFN 与注意力解耦均具实质收益**：FFN 解耦作为模态专用记忆贡献最大加速，继续解耦 Q/K/V 投影仍可额外节省 ~33.3% 图像 FLOPs，且当三者解耦后 LayerNorm 解耦影响微弱。
- **特别契合异构目标与重计算模态**：在 Transfusion 中，图像扩散模态获益巨大（760M MoT 即可全面超越 2× 计算量的 1.4B 稠密模型），可进一步与文本塔 `MoE-4x` 混合互补（§5）。
- **工程系统与开源参考**：Playbook 与实现见 <https://github.com/facebookresearch/Mixture-of-Transformers>；实际部署需针对各模态分支token 比例不均问题采用 Grouped GEMM、连续批处理或动态序列对齐填充（§6.1, §6.3）。

## 常见误区或局限

1. **误认为 MoT 减少了模型总参数量**：§3.3.3 明确指出 MoT 的效率并非来自削减总参数，而是**保持单 token 激活参数/FLOPs 不变**、总非嵌入参数随模态数 $K$ 线性扩展至 $K$ 套权重，从而用静态内存空间换取收敛速度与精度。
2. **忽视模态分组与负载不均的系统开销**：§6.1 指出掩码索引可能触发 GPU–CPU 同步，且同一局部 batch 中各模态 token 数量失衡会导致依次执行各模态投影时 GPU 利用率下降，需配合索引缓存、Grouped GEMM 或跨请求动态批处理优化。
3. **在 Transfusion 下文本增益有限**：§3.4.2 指出当图像已切换为连续扩散目标、文本单独保留自回归目标时，稠密基线本身的文本冲突已缓解，MoT 在文本困惑度上的加速明显小于 Chameleon 设定（此时需结合 `MoT + Text MoE-4x` 提升文本收益）。
4. **注意 MoE-4x 基线的评测边界**：§3.2.1 说明实验中 `MoE-4x` 采用 Expert Choice 路由在验证集直接评测困惑度（存在未来 token 泄露可能高估或分布偏移低估），未直接用于自回归长文本生成对比。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_ref_c3cbdc6867eb310bf60b_mixture-of-transformers-a-sparse-and-sca.md`](../../sources/papers/rcl_awesome_wam_ref_c3cbdc6867eb310bf60b_mixture-of-transformers-a-sparse-and-sca.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://openreview.net/forum?id=Nu6N69i8SB>

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://openreview.net/forum?id=Nu6N69i8SB)
