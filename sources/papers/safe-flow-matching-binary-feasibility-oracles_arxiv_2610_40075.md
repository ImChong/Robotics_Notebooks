# 二值可行性安全流匹配（arXiv:2610.40075）

> 来源归档（paper）

- **标题：** Learning Safe Flow-Matching Robot Policies from Binary Feasibility Oracles
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40075>
- **PDF：** <https://arxiv.org/pdf/2610.40075>
- **作者：** Ma, Jianming, Jin, Rongjun, Chen, Yizhi, Gao, Yue
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 通过二值可行性判定学习安全 Flow Matching 机器人策略。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** Flow Matching 生成的机器人动作可能违反关节限制或碰撞约束，而真实系统中的可行域不一定具有可微解析表达式。
3. **周报所述创新点：** 提出 Oracle-Guided Differentiable Safeguard。前向传播时，通过二值可行性查询修正不安全动作；反向传播时，从局部边界信息构造约束相关梯度，使安全修正能够参与策略训练。在障碍物感知机器人操作实验中，实现 100% 的约束满足率。
4. **arXiv 摘要（英文节选）：** Robot foundation models increasingly rely on flow-based action generation for general-purpose manipulation, yet the generated actions are not guaranteed to satisfy physical constraints such as joint limits and collision avoidance. This problem is especially challenging when the feasible set cannot be described analytically and is accessible only through black-box feasibility oracles. We study this practical oracle-only setting and propose an oracle-guided differentiable safeguard for safe flow-matching robot policies.

**对 wiki 的映射**

- [paper-safe-flow-matching-binary-feasibility-oracles](../../wiki/entities/paper-safe-flow-matching-binary-feasibility-oracles.md)
