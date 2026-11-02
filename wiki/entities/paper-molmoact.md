---
type: entity
project_id: molmoact
tags:
  - paper
  - vla
  - action-reasoning
  - depth
  - open-source
  - ai2
status: complete
updated: 2026-11-02
arxiv: "2508.07917"
code: https://github.com/allenai/molmoact
related:
  - ../methods/vla.md
  - ../entities/paper-molmoact2.md
  - ../entities/molmo2-vlm.md
  - ../entities/paper-pi0.md
  - ../entities/paper-openvla.md
  - ../overview/vla-history-rt1-to-molmoact-technology-map.md
sources:
  - ../../sources/papers/molmoact_arxiv_2508_07917.md
  - ../../sources/blogs/x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.md
summary: "MolmoAct（arXiv:2508.07917）：MolmoAct 提出 Action Reasoning Models（ARM）：把感知、规划与控制放进结构化三阶段——深度感知 token、可编辑的中层空间轨迹、低层动作——使行为可解释、可引导。"
---

# MolmoAct

**MolmoAct**（[arXiv:2508.07917](https://arxiv.org/abs/2508.07917)，[GitHub](https://github.com/allenai/molmoact)）是 [Khushi 的 VLA 演进长文](https://x.com/starlitmatcha/status/2116918927913804223) 中单独成节介绍的模型之一。本页依据 arXiv 摘要页与该长文；技术细节以论文正文为准。

## 一句话定义

**MolmoAct 提出 Action Reasoning Models（ARM）：把感知、规划与控制放进结构化三阶段——深度感知 token、可编辑的中层空间轨迹、低层动作——使行为可解释、可引导。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ARM | Action Reasoning Model | 动作推理模型 |
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| VQ-VAE | Vector-Quantized Variational Autoencoder | 离散化潜变量自编码器 |

## 为什么重要

- 在长文的演进线里属于「显式空间推理（深度 → 轨迹 → 动作）路线」。
- 摘要要点：论文摘要称：MolmoAct 将观测与指令编码为深度感知 token，生成作为可编辑的轨迹的中层空间规划，再预测精确低层动作；MolmoAct-7B-D 在 SimplerEnv Visual Matching 零样本 70.5%，超过闭源 Pi-0 与 GR00T N1.5；LIBERO 平均成功率 86.6%，长时程比 ThinkAct 多 6.3%；真实世界微调较 Pi-0-FAST 单臂多 10%、双臂多 22.7% 任务进度；分布外泛化多 23.3%；并首次发布含 1 万+ 高质量轨迹的 MolmoAct Dataset（中期训练，平均带来 5.5% 提升），开源全部权重、训练代码与数据。

## 长文的讲法（作者表述）

文章称其三阶段为：用 Depth Anything v2 与 VQ-VAE 把深度图压成 10×10 离散 token（作为训练目标，推理时不需再跑深度估计器）→ 预测至多 5 个图像空间点描述末端轨迹 → 输出 8 个离散化动作的 chunk；优点是能检查预测的深度与意图轨迹而非只看最终动作，缺点是生成中间 token 增加推理延迟，且「轨迹看着合理不等于执行正确」。文章称 MolmoAct 2 在其基础上提升空间推理、自适应深度预测并换成连续 flow-matching 动作专家。

## 核对

- **arXiv：** [2508.07917](https://arxiv.org/abs/2508.07917)（提交 2025-08-11），作者：Lee, Duan, Fang, Deng, Liu, Li 等（arXiv 摘要页前 6 位）
- 摘要与文章一致：深度感知 token → 可编辑轨迹 → 动作的三阶段结构。文章的 10×10 token、5 个点、8 个动作 chunk 等细节来自文章，未在摘要中核对。allenai/molmoact 仓库 README 称模型与数据集已于 2025/08/12 发布，训练 / 微调 / 评估代码已释出，并于 2026/5/5 发布 MolmoAct2（allenai/molmoact2）。

## 关联页面

- [vla](../methods/vla.md)
- [paper-molmoact2](../entities/paper-molmoact2.md)
- [molmo2-vlm](../entities/molmo2-vlm.md)
- [paper-pi0](../entities/paper-pi0.md)
- [paper-openvla](../entities/paper-openvla.md)
- [vla-history-rt1-to-molmoact-technology-map](../overview/vla-history-rt1-to-molmoact-technology-map.md)

## 参考来源

- [molmoact_arxiv_2508_07917.md](../../sources/papers/molmoact_arxiv_2508_07917.md)
- [x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.md](../../sources/blogs/x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.md)
- [arXiv:2508.07917](https://arxiv.org/abs/2508.07917)
