# Diffusion Model入门

> 来源归档（blog / 微信公众号）

- **标题：** Diffusion Model入门
- **类型：** blog
- **作者：** human five（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/P4SxYSBnxDjX5De1jxMxfA
- **发表日期：** 2026-07-04
- **入库日期：** 2026-07-04
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；正文约 1.67 万字 / 17 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [wechat_diffusion_model_intro_2026-07-04.md](../raw/wechat_diffusion_model_intro_2026-07-04.md)
- **一句话说明：** 以工程可读方式拆解扩散模型：GAN/VAE 痛点 → 前向加噪与逆向降噪 → DDPM 噪声预测损失 → U-Net/时间步嵌入/隐扩散/条件控制 → DiT 与 SD3 架构脉络；为机器人侧 [Diffusion Policy](../../wiki/methods/diffusion-policy.md) 与运动生成提供生成式底座。

## 核心摘录（归纳，非全文）

### 为何需要扩散（相对 GAN / VAE）

| 路线 | 优势 | 痛点 |
|------|------|------|
| **GAN** | 高清晰度、可学习真实度判别 | 对抗训练不稳定、模式崩溃、调试难 |
| **VAE** | 连续隐空间、可压缩表征 | 样本偏平滑；隐空间设计影响大 |
| **Diffusion** | 监督目标确定（预测已知噪声）、训练稳定 | 采样需多步前向，算力开销高 |

核心转向：把「一步生成整图」拆成 **大量小幅降噪子任务**；单张训练图可随机时间步 + 随机噪声生成无限监督对。

### 前向 / 逆向与 DDPM 要点

- **前向**：人为固定加噪；任意步一步采样 \(x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon\)
- **逆向**：网络学习从噪声分布向数据分布的局部修正；最简范式预测叠加噪声 \(\epsilon_\theta(x_t, t)\)，损失 \(\|\epsilon - \epsilon_\theta\|^2\)
- **调度表** \(\beta_t\)：控制每步噪声强度；\(\bar\alpha_t\) 为累积信号保留比例
- **时间步嵌入**：标量 \(t\) → 正弦/余弦高维向量，使同一网络适配全噪声强度区间
- **采样器 vs 降噪网络**：网络输出修正方向；采样器决定如何用预测更新到下一状态（DDPM / DDIM / 快速采样器等）

### 工程落地组件

| 组件 | 作用 |
|------|------|
| **U-Net 骨干** | 下采样全局语义 + 跳跃连接保留局部细节；经典图像降噪架构 |
| **隐扩散 (LDM)** | 自编码器压缩至隐空间再降噪，降低高分辨率算力 |
| **条件控制** | 文本/类别/掩码/深度等经交叉注意力或无分类器引导 (CFG) 注入 |
| **Diffusion Transformer** | 隐张量 patchify 为 token；DiT / PixArt / MM-DiT (SD3) 用 Transformer 替代 U-Net |

### 常见误区（文内收束）

1. 网络不记忆每张图的专属还原路径，而是学 **通用降噪梯度规则**
2. 初始噪声不含最终图像；种子与条件共同决定轨迹
3. 提示词是统计约束，非精确指令（计数、空间关系、否定易错）
4. CFG 引导系数是贴合度–自然度权衡，非单纯「画质旋钮」
5. DDPM 只是入门框架；Flow Matching、一致性模型、rectified flow 等同属扩散家族变体

### 文内主要参考论文

1. Denoising Diffusion Probabilistic Models (DDPM)
2. High-Resolution Image Synthesis with Latent Diffusion Models (LDM)
3. Scalable Diffusion Models with Transformers (DiT)
4. PixArt-α
5. Flow Matching for Generative Modeling
6. Scaling Rectified Flow Transformers (SD3 / MM-DiT)

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [扩散模型（概念）](../../wiki/concepts/diffusion-model.md) | **主沉淀页**：通用机制、训练/采样 pipeline、误区与局限 |
| [生成式模型基础](../../wiki/formalizations/generative-foundations.md) | 数学底座；本页补 **工程直觉与架构演进** |
| [Diffusion Policy](../../wiki/methods/diffusion-policy.md) | 机器人操作：把「生成图像」换成「生成动作序列」 |
| [基于扩散的运动生成](../../wiki/methods/diffusion-motion-generation.md) | 人形/全身轨迹的条件扩散规划 |
| [概率流 (Probability Flow)](../../wiki/formalizations/probability-flow.md) | Flow Matching / rectified flow 等形式化对照 |
