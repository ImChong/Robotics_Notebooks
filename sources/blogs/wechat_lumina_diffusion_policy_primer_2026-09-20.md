# Diffusion Policy如何改变机器人学习

> 来源归档（blog / 微信公众号 · Lumina 机器人技术指南）

- **标题：** Diffusion Policy如何改变机器人学习
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485404&idx=1&sn=f0b24d4d70afc9b4800e4c4adc5ac771
- **专辑：** [wechat_lumina_embodied_practice_album](../raw/wechat_lumina_embodied_practice_album_4608355279393816579.md)
- **入库日期：** 2026-09-20
- **对照来源：** [Embodied-AI-Guide §3 Robot Learning 策略基线](https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/topics/algorithm.md#3-robot-learning--机器人学习从控制到策略)
- **一句话说明：** 把扩散去噪从图像生成换成**动作 chunk 生成**，天然表达多峰演示分布；与 ACT 的 Transformer chunk、π0 的 flow expert 构成 IL 三条主基线。

## 核心摘录：三个 IL 基线 → 独立节点

| 基线 | 链接 | 本库节点 | 开源 |
|------|------|----------|------|
| ACT | tonyzhaozh/act | [paper-act](../../wiki/entities/paper-act.md) | **已开源** |
| Diffusion Policy | real-stanford/diffusion_policy | [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md) | **已开源** |
| DP3 (3D Diffusion Policy) | YanjieZe/3D-Diffusion-Policy | [painode-209-3ddiffusionpolicydp3](../../wiki/entities/painode-209-3ddiffusionpolicydp3.md) | **已开源** |

**机制要点（归纳）：** 训练对专家动作加噪、网络预测噪声；推理从 \( \mathcal{N}(0,I) \) 逐步去噪得到 \(T_p\) 长度 chunk；部署常用 receding horizon 只执行前 \(T_e\) 步。详见 [diffusion-policy 方法页](../../wiki/methods/diffusion-policy.md) 与 [receding-horizon 概念](../../wiki/concepts/receding-horizon-policy-execution.md)。

## 对 wiki 的映射

- [embodied-ai-guide-wechat-album-curator.md](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md) § Diffusion Policy
- [diffusion-policy.md](../../wiki/methods/diffusion-policy.md)、[action-chunking.md](../../wiki/methods/action-chunking.md)
