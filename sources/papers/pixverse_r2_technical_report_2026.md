# PixVerse R2: Scaling Real-Time Omni World Models

- **类型：** technical report / blog（PixVerse Research）
- **链接：** <https://pixverse.ai/en/blog/pixverse-r2-scaling-real-time-omni-world-models>
- **发布：** 2026-08-23（页面标注）
- **入库日期：** 2026-09-23
- **前置：** PixVerse R1 — 首个公开发布的通用实时音视频世界模型

## 核心摘录

1. **两阶段框架：** 持续预训练 **Omni Causal AR** → 同骨干 **Real-Time Acceleration**（非重学世界）。
2. **Omni Causal AR：** 文本/参考/音频/动作（WASD 等）进入同一运行世界；**Dynamic Chunk Generation** 按控制信号语义边界切分音视频块。
3. **Train–Inference：** Hybrid **Teacher Forcing + Diffusion Forcing**；因果 mask + **Relative Temporal RoPE**。
4. **Multi-Timescale Memory：** Sink Memory（身份/规则）+ Rolling History（近期动力学）+ Object KV Cache。
5. **Error Bank：** 存储代表性失败状态并训练期回放；长序列亮度漂移内部评测 −35.8%。
6. **Real-Time Acceleration：** DDMD + 对抗正则 + **Block-Sparse Attention**（>90% 稀疏）+ **Pyramid Ultra-Few-Step Distillation**。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-pixverse-r2.md`](../../wiki/entities/paper-pixverse-r2.md)
- 方法交叉：[`wiki/methods/generative-world-models.md`](../../wiki/methods/generative-world-models.md)
