# OracleZoom: On-Policy Self-Distillation Inspired Reference-Constrained Recursive Image Super Resolution（arXiv:2609.06490）

> 来源归档（ingest）

- **标题：** OracleZoom: On-Policy Self-Distillation Inspired Reference-Constrained Recursive Image Super Resolution
- **类型：** paper / recursive-super-resolution
- **来源：** arXiv abs / PDF；项目页、GitHub README、HF 权重交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2609.06490>
  - PDF：<https://arxiv.org/pdf/2609.06490>
  - 项目页：<https://dipta007.github.io/OracleZoom/>
  - 代码：<https://github.com/dipta007/OracleZoom>
  - 权重：<https://huggingface.co/dipta007/OracleZoom>
  - Demo：<https://huggingface.co/spaces/dipta007/OracleZoom>
  - 训练数据：<https://huggingface.co/datasets/dipta007/OracleZoom-4KLSDB-train>
- **作者：** Shubhashis Roy Dipta*, Sourajit Saha*, Shaswati Saha, Nobin Sarwar（*equal contribution）
- **机构：** University of Maryland, Baltimore County（UMBC）
- **venue：** WACV 2027（in submission，项目页 / README 表述）
- **入库日期：** 2026-09-28
- **复核日期：** 2026-09-28（对照 arXiv 摘要、项目页 meta、GitHub README）
- **一句话说明：** 递归 SR 在深层 zoom **无 GT 监督**；OracleZoom **on-policy** 训练自身预测链，用 **最后一档 GT 作跨尺度参考** 约束可验证结构，**TOPIQ-NR 质量 + KL 约束 latent 先验 + EMA** 引导不可验证细节，七数据集 SOTA 级 CLIPIQA 并显著降幻觉。

## 开源状态（项目页 / README 核查 2026-09-28）

- **已开源：** 代码、merged 权重、1k 训练 tier 数据、Gradio Demo Space 均可公开获取（见上链接）。
- **互指：** [`sources/sites/oraclezoom-project.md`](../sites/oraclezoom-project.md) · [`sources/repos/oraclezoom.md`](../repos/oraclezoom.md)

## 核心论文摘录（MVP）

### 1) 监督缺口：递归倍率 vs GT 存储

- **链接：** arXiv §1；Fig. 1
- **摘录要点：**  successive **4×** 递归时，256× 目标对应约 **131072²** 源像素（单张未压缩 RGB ≈ **52 GB**），深层 zoom **无法** 提供逐像素 GT；模型只能依赖自身预测与弱语义引导（如 VLM caption），**无法直接验证** 合成纹理是否与观测一致。
- **对 wiki 的映射：**
  - [OracleZoom（实体）](../../wiki/entities/paper-oraclezoom.md)

### 2) On-policy 训练 + 参考约束分解

- **链接：** arXiv §3.2；Fig. 2
- **摘录要点：** 受 **OPSD（On-Policy Self-Distillation）** 启发，训练轨迹与推理一致（预测作下一步输入并 **反传整条链**）。对 **GT 不可用** 的尺度，将合成拆为 **(1) 仍可投影对齐验证的部分** — 用 **最后一档 GT** 做 cross-scale consistency；(2) **不可由投影确定的细尺度细节** — 冻结 **无参考质量模型** + **KL 先验** 贴近预训练 SR latent；**EMA teacher** 在监督边界稳定训练。
- **对 wiki 的映射：**
  - [OracleZoom（实体）](../../wiki/entities/paper-oraclezoom.md)

### 3) 实现与训练预算

- **链接：** README §4–5；项目页 stats
- **摘录要点：** 共享 **rank-16 LoRA**（**7.1M** 参数）适配冻结多尺度 VLM prompter + latent SR + VAE decoder；论文设置 **1,000** 图、训练至验证平台（README 约 **9,300** step）。推理 **4 次 4×** 至 256×，每步回到 512×512 再 zoom。
- **对 wiki 的映射：**
  - [OracleZoom（实体）](../../wiki/entities/paper-oraclezoom.md)
  - [`sources/repos/oraclezoom.md`](../repos/oraclezoom.md)

### 4) 评测：质量、保真与 VLM 判幻觉

- **链接：** arXiv 摘要；README §3
- **摘录要点：** **七测试集** CLIPIQA 均值 **0.713**；4× **LPIPS 0.199 / DISTS 0.160**（有 GT 的集合）；256× CLIPIQA **0.706**。相对 **Chain-of-Zoom**，InternVL3.5-38B 在 64× / 256× 决定性子比较中偏好 OracleZoom **68% / 78%**； judged hallucination **0.21 / 0.14** vs CoZ **0.55 / 0.70**。消融：去掉 KL 可升 NR 质量但 **投影 DISTS 与幻觉恶化**。
- **对 wiki 的映射：**
  - [OracleZoom（实体）](../../wiki/entities/paper-oraclezoom.md)

### 5) 与 Chain-of-Zoom 的方法对照

- **链接：** arXiv §2 Related Work
- **摘录要点：** **CoZ** 递归固定倍率 SR + 多尺度 **VLM 文本引导**；OracleZoom 聚焦 **GT 边界之外** 仍保留 **可验证视觉证据**，而非仅语义 caption。
- **对 wiki 的映射：**
  - [OracleZoom（实体）](../../wiki/entities/paper-oraclezoom.md)

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-oraclezoom.md`](../../wiki/entities/paper-oraclezoom.md) — 主实体页（arXiv:2609.06490）
- [`wiki/queries/robot-perception-stack-selection-loop.md`](../../wiki/queries/robot-perception-stack-selection-loop.md) — 感知栈中「细节增强 vs 观测一致性」延伸阅读
