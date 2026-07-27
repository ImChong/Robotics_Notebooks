# MoLingo: Motion–Language Alignment for Text-to-Human Motion Generation（arXiv:2512.13840）

> 来源归档（ingest）

- **标题：** MoLingo: Motion–Language Alignment for Text-to-Human Motion Generation
- **类型：** paper / text-to-motion / latent-diffusion / autoregressive / rectified-flow / human-motion
- **arXiv abs：** <https://arxiv.org/abs/2512.13840>
- **PDF：** <https://arxiv.org/pdf/2512.13840>
- **项目页：** <https://hynann.github.io/molingo/MoLingo.html> — 归档见 [`sources/sites/molingo-github-io.md`](../sites/molingo-github-io.md)
- **代码：** <https://github.com/hynann/MoLingo> — 归档见 [`sources/repos/molingo.md`](../repos/molingo.md)
- **作者：** Yannan He, Garvita Tiwari, Xiaohan Zhang, Pankaj Bora, Tolga Birdal, Jan Eric Lenssen, Gerard Pons-Moll
- **机构：** University of Tübingen；Tübingen AI Center；Max Planck Institute for Informatics；Imperial College London；Zuse School ELIZA
- **会议：** CVPR 2026
- **入库日期：** 2026-07-27
- **一句话说明：** 在 **连续运动 latent** 上做 **掩码自回归 rectified flow**；用 **BABEL 帧级文本** 训 **语义对齐自编码器（SAE）**，并以 **T5 多 token cross-attention** 条件注入，提升 HumanML3D 上的真实感与指令跟随，并演示 **G1 + PHC 跟踪**。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 训练/评测数据 | HumanML3D（Guo et al., CVPR 2022） | 主基准；29,024 motions / 87,834 texts |
| 帧级语义 | BABEL（CVPR 2021） | SAE 的 frame-level 文本标签来源（与 HumanML3D 交集） |
| 评测协议 | MARDM-67 / MS-272 / TMR-263 | 本文多协议报告；主表用 MARDM-67 |
| 连续 AR 潜扩散对照 | MARDM、MotionStreamer、ACMDM | 项目页定性对比与用户研究基线 |
| 物理跟踪下游 | PHC + Unitree G1 | 项目页：生成 → retarget → 预训练 RL tracker |

## 摘要级要点

- **问题：** 连续 latent 上的 T2M 扩散已有「整段一次」与「多 latent 自回归」两轨；关键是 **何为可扩散的 latent**，以及 **文本如何注入**。
- **SAE：** 因果 1D-CNN encoder–decoder 把 \(N\) 帧压成 \(l=N/h\) 个连续 latent；用 BABEL 帧级标签经冻结文本编码器得到 class token \(\kappa\)，对过滤后的索引集做 cosine \(\mathcal{L}_{\text{sem}}\)（避免连续重复标签过强对齐）；总损失 = recon + \(\lambda_{\text{sem}}\mathcal{L}_{\text{sem}}\) + \(\lambda_{\text{KL}}\mathcal{L}_{\text{KL}}\)。
- **生成：** 冻结 T5-Large → text adapter → 多 token \(\mathbf{w}\)；训练随机 mask latent；decoder-only Transformer（self-attn + **cross-attn 到文本**）出 \(z_i\)，MLP \(v_\theta\) 学 rectified-flow 速度场；推理从全 mask 迭代去噪，CFG 10% 训练 / 推理 scale≈5.5。
- **消融：** CrossAttn+T5 ≫ 单 token AdaLN；SAE 相对 VAE/AE 抬升 R-Precision / CLIP-Score，FID 仍可比；\(\lambda_{\text{sem}}=0.001\) + KL + cosine 优于 InfoNCE。
- **主结果（MARDM-67，Tab.1）：** MoLingo (VAE) FID **0.049**；MoLingo (SAE) R-Precision Top-1 **0.544**、CLIP-Score **0.686**（均值±95% CI，20 runs）。
- **用户研究：** 相对 DisCoRD / MoMask / MotionStreamer，偏好率约 **83.8% / 77.7% / 84.7%**。
- **机器人：** 生成动作经重定向后由 **PHC 风格预训练 RL tracker** 在 G1 上跟踪；相对 MotionStreamer 宣称更稳足地接触（**G1 管线代码截至入库日未随仓发布**）。
- **局限：** 聚焦主躯干，**不生成精细手指**；MModality 低于部分多样性更强的基线。

## 核心摘录（面向 wiki 编译）

### 1) 语义损失（Eq.1 归纳）

过滤连续 class token 相似度过高的对后，最小化 \(1-\cos(m_i,\kappa_i)\)，使运动 latent 软对齐文本语义而不硬推开异类（相对 InfoNCE）。

### 2) 自回归 flow（Eq.6–7）

\(p(m_{1:l})=\prod_i p(m_i\mid c,m_{<i})\)；每步 \(z_i=\Phi(w;m_{<i})\)，对 \(m_i^t=(1-t)m_i+t\epsilon\) 回归 \(v_\theta\to(\epsilon-m_i)\)。

### 3) 与机器人知识库的关系

- **上游人体 T2M：** 与 [HY-Motion](../../wiki/methods/hy-motion-1.md)、[DART](../../wiki/methods/dart-control.md)、[Awesome T2M](../../wiki/entities/awesome-text-to-motion-zilize.md) 同属人体先验；MoLingo 强调 **语义对齐 latent + AR flow + cross-attn**。
- **下游执行：** 项目页走 **人体生成 → retarget → PHC tracker → G1**；与 [PhyGile](../../wiki/entities/paper-phygile.md) 的 **robot-native 262D 生成–GMT 闭环** 形成对照（后者刻意跳过人体 retarget 鸿沟）。
- **数据接口：** 评测锚定 [HumanML3D](../../wiki/entities/paper-notebook-humanml3d.md)；物理跟踪锚 [PHC](../../wiki/entities/phc.md)。

## 对 wiki 的映射

- 沉淀论文实体：[paper-molingo](../../wiki/entities/paper-molingo.md)
- 交叉更新：[`diffusion-motion-generation`](../../wiki/methods/diffusion-motion-generation.md)、[`hy-motion-1`](../../wiki/methods/hy-motion-1.md)、[`dart-control`](../../wiki/methods/dart-control.md)、[`awesome-text-to-motion-zilize`](../../wiki/entities/awesome-text-to-motion-zilize.md)、[`phc`](../../wiki/entities/phc.md)、[`paper-notebook-humanml3d`](../../wiki/entities/paper-notebook-humanml3d.md)、[`paper-phygile`](../../wiki/entities/paper-phygile.md)
