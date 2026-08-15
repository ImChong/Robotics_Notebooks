# EmbodiedVAE: Disentangled Video VAE for Efficient and Controllable Embodied Manipulation（arXiv:2608.02990）

> 来源归档（ingest）

- **标题：** EmbodiedVAE: Disentangled Video VAE for Efficient and Controllable Embodied Manipulation
- **缩写 / 框架：** **EmbodiedVAE**；下游骨干示例 **IRASim-L**
- **类型：** paper / video-vae / world-model / manipulation
- **arXiv：** <https://arxiv.org/abs/2608.02990>
- **代码：** <https://github.com/Mutual-Luo/EmbodiedVAE>（**Coming Soon**；归档见 [`sources/repos/embodiedvae.md`](../repos/embodiedvae.md)）
- **会议：** ECCV 2026（仓库描述）
- **作者：** Jiayi Luo、Hanxin Zhu\*、Chen Gao、Jiankun Wang、Cong Wang、Tianyu He、Jianxin Li†、Zhibo Chen†
- **机构：** 北京航空航天大学（BUAA）；中关村人工智能研究院（ZGCA）；中国科学技术大学（USTC）；新加坡国立大学（NUS）；中国科学院自动化研究所（CASIA）；微软亚洲研究院（Microsoft Research Asia）
- **入库日期：** 2026-08-15
- **一句话说明：** 为操作世界模型重做 video VAE：双编码器单解码器 + 非对称时空压缩，自动解耦机械臂运动与背景，并用最优传输一致性约束运动保真；相对主流 video VAE 约 **+2 dB PSNR** 的动作可控生成。

## 开源状态（步骤 2.5）

- **仓库核查（2026-08-15）：** [Mutual-Luo/EmbodiedVAE](https://github.com/Mutual-Luo/EmbodiedVAE) 仅 README「Coming Soon!」，**无训练 / 推理脚本或权重**。
- **结论：** **宣称开源 / 实现待发布。** 源码运行时序图标 **不适用**。

## 摘录 1：问题与主张

- 现成 LDM 沿用自然场景 VAE：2D VAE 潜空间太大；video VAE 时间压缩常丢掉臂运动语义 → 画面好看但动作控不住（文内 Wan-VAE 对照）。
- **主张：** 训练期用臂 mask 做两阶段解耦，推理**不再需要 mask**；臂编码器更强空间压缩 \(4\times(16\times16)\)，背景更强时间压缩 \(8\times(8\times8)\)，通道均为 4。

## 摘录 2：方法栈

| 模块 | 要点 |
|------|------|
| **臂编码器** | 因果 3D conv + 3D attention；四次空间、两次时间下采样 |
| **背景编码器** | 三次空间 + 三次时间下采样，吃第一人称背景冗余 |
| **统一解码器** | 双头上采样至同分辨率后通道拼接，共享尾部重建 |
| **OT 一致性** | 在臂潜空间最小化跨帧信息传输代价，稳住运动对应 |
| **下游** | 固定 IRASim-L（461M）；两路 latent 共享 DiT，每 2 block 交叉注意 |

## 摘录 3：实验

- 重建：压缩率约 **0.39%**；Agibot-2025 PSNR **31.67**，约为次优 CMD 压缩率的 1/20 量级仍更高保真（Table 1）。
- 动作条件生成：相对次优 video VAE（Wan-VAE）约 **+2.02 PSNR**，并超过无时间压缩的 SDXL 图像 VAE（Table 2）。
- 消融：OT 运动损失在重建与下游 manipulation 均正收益，下游更大。

**对 wiki 的映射：** [`wiki/entities/paper-embodiedvae.md`](../../wiki/entities/paper-embodiedvae.md)；交叉 [世界动作模型](../../wiki/concepts/world-action-models.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（Coming Soon）
