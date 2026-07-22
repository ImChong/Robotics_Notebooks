# Patch Policy: Efficient Embodied Control via Dense Visual Representations（arXiv:2607.18236）

> 来源归档（ingest）

- **论文：** <https://arxiv.org/abs/2607.18236>（2026-07-20 提交）
- **项目页：** <https://patch-policy.github.io/>
- **作者：** Gaoyue Zhou, Zichen Jeff Cui, Ada Langford, Bowen Tan, Yann LeCun, Lerrel Pinto
- **代码：** 截至 2026-07-22，项目页标注 **GitHub (coming soon)**
- **一句话说明：** 用块因果注意力直接消费预训练 ViT 的密集 patch token，在不引入大型 VLM 的前提下保留细粒度空间信息。

## 核心摘录

- **问题：** 全局池化丢失空间细节，大型 VLA 又难满足高频控制的参数量与延迟预算。
- **方法：** 轻量 transformer 直接读取每帧 patch token，并以 block-causal mask 同时保证时间因果与帧内密集注意力。
- **结果：** 四个仿真和三个真实环境套件上，相对全局池化表征提升 **40%**；比微调 OpenVLA-OFT 高 **18%**，参数约为其 **0.7%**。

**对 wiki 的映射：** [`wiki/entities/paper-patch-policy.md`](../../wiki/entities/paper-patch-policy.md)、[VLA](../../wiki/methods/vla.md)、[Vision Transformer](../../wiki/concepts/vision-transformer.md)。

