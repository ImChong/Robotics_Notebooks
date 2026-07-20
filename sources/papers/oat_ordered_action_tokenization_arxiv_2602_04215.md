# OAT: Ordered Action Tokenization for Autoregressive Action Prediction

> 来源归档

- **标题：** Ordered Action Tokenization for Autoregressive Action Prediction
- **类型：** paper
- **出处：** 2026 · RSS 2026 Finalist · arXiv preprint
- **arXiv：** <https://arxiv.org/abs/2602.04215>
- **论文 HTML：** <https://arxiv.org/html/2602.04215>
- **项目页：** <https://ordered-action-tokenization.github.io/>
- **代码：** <https://github.com/Chaoqi-LIU/oat>（项目页 OAT Code）；VLA 扩展 <https://github.com/Chaoqi-LIU/praxis-vla>
- **作者：** Chaoqi Liu, Xiaoshen Han, Jiawei Gao, Yue Zhao, Haonan Chen, Yilun Du（Harvard + Stanford）
- **入库日期：** 2026-07-20
- **一句话说明：** 三标准（高压缩 / 完全可解码 / 左右因果 token 空间）驱动的连续机器人动作 tokenization 方案；粗到细层级残差编码；AR 策略在 20+ 操作任务超扩散策略基线；RSS 2026 Finalist。

---

## 核心摘录（策展，非全文）

### 问题与动机

- 自回归（AR）策略复用 LLM 的推理效率优势，但关键前提是 **将连续动作离散化为高质量 token**。
- 现有 tokenization 方案（均匀量化、VQ-VAE、FSQ 等）缺少统一设计框架，往往在压缩率、解码精度、token 顺序语义三者间取舍失当。
- 本文提出 **三项标准** 作为评价与设计 tokenization 的统一框架：high compression、total decodability、left-to-right causal token space。

### 关键洞察

1. **有序残差分解：** 将动作逐层分解为粗粒度方向 + 细粒度残差；layer 1 先预测大方向，后续 layer 逐步精化 → 左到右具有语义因果。
2. **三标准的设计推导：** 从三标准出发可唯一推导出「粗到细层级残差量化」的合理结构，不是经验试错而是原理性设计。
3. **AR 超 DP：** 正确 tokenization 下 AR 策略在成功率上超越 Diffusion Policy，说明精度差距来自 tokenization 而非 AR 范式本身。

### 方法要点

| 维度 | OAT |
|------|-----|
| 标准 1：高压缩 | 层级残差量化，单 token 预算下更高精度 |
| 标准 2：完全可解码 | 每层残差精确还原，无码本崩溃 |
| 标准 3：左右因果 | 粗→细顺序 = token 编号 1→N 的语义单调性 |
| 训练 | AR cross-entropy on token sequences |
| 推理 | 自左向右逐 token 采样 → 残差解码 → 连续动作 |

### 实验摘要

- 20+ 操作任务；成功率指标。
- OAT-AR 超越：均匀量化 AR、VQ/FSQ-AR、Diffusion Policy（成功率绝对值 +X%，论文表格详见原文）。
- 消融：三标准各项缺失均导致显著下降。

### 关于 Knowledge-insulated Token Co-training

- 作者在访谈中提及将 OAT 扩展到 VLA 联合训练时，设计了 **知识隔离 token 协同训练** 策略，防止动作 token 污染语言/视觉预训练知识；正式论文尚未发布该扩展细节。

### 局限（论文自述）

- 主仓已开源；权重与 HF 数据入口见 README。
- 高动力学任务（locomotion 等）泛化性未验证。

### 对 wiki 的映射

- [paper-oat-ordered-action-tokenization](../../wiki/entities/paper-oat-ordered-action-tokenization.md)
- [vla](../../wiki/methods/vla.md)
- [diffusion-policy](../../wiki/methods/diffusion-policy.md)
- [unified-multimodal-tokens](../../wiki/methods/unified-multimodal-tokens.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2602.04215>
- 项目页：<https://ordered-action-tokenization.github.io/>

- 报道入口：[`sources/blogs/wechat_qbitai_rss2026_awards_2026-07-16.md`](../blogs/wechat_qbitai_rss2026_awards_2026-07-16.md)
