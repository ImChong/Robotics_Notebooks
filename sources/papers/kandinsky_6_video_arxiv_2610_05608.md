# Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation

> 来源归档（paper）

- 类型：paper / multimodal generation / diffusion
- arXiv：https://arxiv.org/abs/2610.05608v1
- 作者：Team Kandinsky 等 87 位合作者
- 项目页：https://kandinskylab.ai/models/video/
- 代码：MIT，https://github.com/kandinskylab/kandinsky-6
- 入库日期：2026-10-11

## 摘录与 wiki 映射

- Lite 3B、Pro 29B；支持文本/图像条件生成 5 秒音视频，含 44 kHz 同步音频与 lip-sync；可选超分辨至 1920×1080。
- 双流 CrossDiT 连接预训练视频流与新训练音频流，以双向 cross-attention 对齐时序和语义。
- 训练顺序：音频流预训练、配对音视频联合预训练、SFT、RL 后训练与蒸馏。
- 论文称 Pro 优于 Kandinsky 5.0 Video Pro，且在 speech quality 方面与领先模型具竞争力；结论来自 side-by-side 人评。
- 对 wiki 的映射：[Kandinsky 6.0 Video 实体](../../wiki/entities/paper-kandinsky-6-video.md)；[项目页归档](../sites/kandinsky-6-video.md)；[仓库归档](../repos/kandinsky-6.md)。
