# Imitation Learning

聚焦行为克隆、DAgger、Diffusion Policy、动作块输出与技能嵌入相关论文。

## 关注问题

- 如何解决行为克隆（BC）中的协变量偏移（Covariate Shift）问题？
- 如何通过生成式模型（如 Diffusion）建模多模态专家数据？
- 如何有效压缩长时序动作序列以提升操作稳定性？
- 如何将动捕数据或专家演示转化为可重用的技能先验？

## 代表性论文

### 基础算法

- **DAgger** (Ross et al., 2011) — *A Reduction of Imitation Learning to No-Regret Online Learning*. 系统解决了 BC 的累计误差问题。
- **Diffusion Policy** (Chi et al., 2023) — *Visuomotor Policy Learning via Action Diffusion*. 将扩散模型引入策略建模，显著提升了复杂操作任务的成功率。

### 动作表达与架构

- **ACT** (Zhao et al., 2023) — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. 提出了 Action Chunking，改善了长时序稳定性。

### 技能嵌入

- **ASE** (Peng et al., 2022) — *Adversarial Skill Embeddings for Hierarchical RL*. 学习可组合的隐空间技能表征。

## 关联页面

- [Imitation Learning (Method)](../../wiki/methods/imitation-learning.md)
- [Diffusion Policy (Method)](../../wiki/methods/diffusion-policy.md)
- [Action Chunking (Method)](../../wiki/methods/action-chunking.md)
- [Behavior Cloning (Method)](../../wiki/methods/behavior-cloning.md)
