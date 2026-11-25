---
type: tech_map_node
status: complete
topic: [imitation-learning, manipulation]
summary: "Diffusion Policy 技术栈模块：通过条件去噪扩散过程生成多模态动作块（action chunk），克服传统回归式行为克隆的模态平均与长时序抖动问题。"
sources:
  - ../../../sources/papers/diffusion_policy_reading_arxiv_2303_04137.md
  - ../../../sources/papers/diffusion_and_gen.md
updated: 2026-11-24
---

# Diffusion Policy（技术栈模块）

**Diffusion Policy** 把动作生成写成一次去噪过程：不再直接回归单个动作，而是从噪声出发迭代生成一段动作序列（action chunk）。这让策略能表达多模态的动作分布——同一个场景下「从左边绕」和「从右边绕」都是对的，而回归式 BC 往往会把它们平均成一个错误的中间动作。

## 在全栈中的位置

```mermaid
flowchart LR
    OBS["视觉 + 本体状态观测序列"] --> COND["视觉编码器（CNN / ViT）条件嵌入"]
    NOISE["高斯噪声动作块 A_k"] --> DENOISE["迭代去噪网络（1D U-Net / DiT）"]
    COND --> DENOISE
    DENOISE --> CHUNK["多步动作块输出（Receding Horizon 执行）"]
```

## 核心内容与站内入口

- **方法页与论文实体**：[Diffusion Policy 方法页](../../../wiki/methods/diffusion-policy.md) · [原始论文实体](../../../wiki/entities/paper-diffusion-policy.md)
- **操作策略选型**：[操作策略架构选型](../../../wiki/queries/manipulation-vla-architecture-selection.md) · [模仿学习纵深路线](../../../roadmap/depth-imitation-learning.md)

## 上下游模块

- **上游基础**：[Behavior Cloning](behavior-cloning.md) · [全栈技术域总览](../../overview.md)
- **相关模块**：[策略网络架构](policy-network-architecture.md) · [部署（上真机）](../system/deployment.md)

## 参考来源与收录边界

- [Diffusion Policy 原文通读归档（arXiv:2303.04137）](../../../sources/papers/diffusion_policy_reading_arxiv_2303_04137.md)
- [sources/papers/diffusion_and_gen.md](../../../sources/papers/diffusion_and_gen.md) — 生成式策略家族文献归档
- 推理延迟与动作块执行协议见对应 `wiki/` 页面。
