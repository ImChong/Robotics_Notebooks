# DiT：用 Transformer 做可扩展扩散骨干（arXiv:2212.09748）

> 论文来源归档（ingest）

- **标题：** Scalable Diffusion Models with Transformers
- **作者：** William Peebles, Saining Xie
- **类型：** paper / generative-model / diffusion / transformer
- **arXiv：** <https://arxiv.org/abs/2212.09748> · PDF：<https://arxiv.org/pdf/2212.09748.pdf>
- **会议：** ICCV 2023
- **官方代码：** <https://github.com/facebookresearch/DiT>
- **入库日期：** 2026-09-21
- **一句话说明：** 在隐空间扩散里用 **patch 化 Transformer** 替换 U-Net，展示清晰的 **Gflops–FID scaling**，成为后续文生图与机器人 flow/DiT 动作头的骨干模板。

## 核心摘录（面向 wiki 编译）

### 1) 扩散骨干可以不是 U-Net

- **要点：** 潜张量被切成 patch token，经 AdaLN 等调制注入时间步/类别条件，Transformer 预测噪声。U 形跳跃不再是必要条件。
- **对 wiki 的映射：** [`wiki/concepts/diffusion-transformer.md`](../../wiki/concepts/diffusion-transformer.md)、[`wiki/concepts/diffusion-model.md`](../../wiki/concepts/diffusion-model.md)

### 2) 前向算力与样本质量可预测缩放

- **要点：** 在控制变量下，更高 Gflops 的 DiT 对应更好 FID。这给「加大动作头」提供了图像域先例；机器人侧仍须另测延迟。
- **对 wiki 的映射：** [`wiki/concepts/diffusion-transformer.md`](../../wiki/concepts/diffusion-transformer.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

### 3) 机器人 VLA 为何爱 DiT

- **要点：** 动作块是短序列，天然适合 Transformer token；与 VLM 骨干同族，便于共享工具链。Xiaomi / π / GR00T 等公开 VLA 的连续动作头大量写 **DiT + flow matching**。
- **对 wiki 的映射：** [`wiki/methods/vla.md`](../../wiki/methods/vla.md)、[`wiki/methods/diffusion-policy.md`](../../wiki/methods/diffusion-policy.md)

## 开源状态（步骤 2.5）

- `facebookresearch/DiT` **已开源**（PyTorch + 预训练权重说明）。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
