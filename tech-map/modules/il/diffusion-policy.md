# Diffusion Policy

**Diffusion Policy** 把动作生成写成一次去噪过程：不再直接回归单个动作，而是从噪声出发迭代生成一段动作序列（action chunk）。这让策略能表达多模态的动作分布——同一个场景下「从左边绕」和「从右边绕」都是对的，而回归式 BC 往往会把它们平均成一个错误的中间动作。

**站内入口：** [Diffusion Policy 方法页](../../../wiki/methods/diffusion-policy.md) · [原始论文实体](../../../wiki/entities/paper-diffusion-policy.md) · [操作策略架构选型](../../../wiki/queries/manipulation-vla-architecture-selection.md)
