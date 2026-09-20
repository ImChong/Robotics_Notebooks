# VLA综述：具身智能路线梳理 (一) - 建议收藏

> 来源归档（blog / 微信公众号 · Lumina 机器人技术指南）

- **标题：** VLA综述：具身智能路线梳理 (一) - 建议收藏
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485266&idx=1&sn=c3c42934e34f0d9dabd1a7655e6ec2b2
- **专辑：** [wechat_lumina_embodied_practice_album](../raw/wechat_lumina_embodied_practice_album_4608355279393816579.md)
- **入库日期：** 2026-09-20
- **对照来源：** [Embodied-AI-Guide §5.0–5.1 VLA](https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/topics/algorithm.md#5-vision-language-action-models--vla-模型)
- **一句话说明：** VLA 把视觉–语言能力延伸到**动作空间**；差异来自动作表示（token / 连续 chunk / 扩散）、数据配方与是否分层双系统。

## 核心摘录：经典 VLA 工作 → 独立节点

| 方向 | 工作 | arXiv | 本库节点 |
|------|------|-------|----------|
| Autoregressive | RT-1 | 2212.06817 | [paper-rt-1](../../wiki/entities/paper-rt-1.md) |
| Autoregressive | RT-2 | 2307.15818 | [paper-rt-2](../../wiki/entities/paper-rt-2.md) |
| Autoregressive | OpenVLA | 2406.09246 | [paper-openvla](../../wiki/entities/paper-openvla.md) |
| Autoregressive | RoboFlamingo | 2311.01378 | [paper-pai-2311-01378-roboflamingo](../../wiki/entities/paper-pai-2311-01378-roboflamingo.md) |
| Diffusion/Flow | Octo | 2405.12213 | [paper-octo](../../wiki/entities/paper-octo.md) |
| Diffusion/Flow | π0 | 2410.24164 | [paper-pi0](../../wiki/entities/paper-pi0.md) |
| Diffusion/Flow | CogACT | 2411.19650 | [paper-cogact](../../wiki/entities/paper-cogact.md) |
| Diffusion/Flow | Diffusion-VLA | 2412.03293 | [paper-diffusion-vla](../../wiki/entities/paper-diffusion-vla.md) |
| 3D Vision | 3D-VLA | 2403.09631 | [paper-sa-2403-09631-3d-vla-a-3d-vision-language-action-generative-wo](../../wiki/entities/paper-sa-2403-09631-3d-vla-a-3d-vision-language-action-generative-wo.md) |
| 3D Vision | SpatialVLA | 2501.15830 | [paper-spatialvla](../../wiki/entities/paper-spatialvla.md) |
| 双臂扩展 | RDT-1B | 2410.07864 | [paper-rdt-1b](../../wiki/entities/paper-rdt-1b.md) |

**综述资源（不重复造页）：** Action Tokenization Survey 2507.01925、VLA for Embodied AI Survey 2405.14093 — 见 [vla.md](../../wiki/methods/vla.md) 参考节。

## 对 wiki 的映射

- [embodied-ai-guide-wechat-album-curator.md](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md) § VLA 一
- [VLA 方法页](../../wiki/methods/vla.md)、[VLA 演进纵览](../../wiki/overview/vla-evolution-lineage.md)
