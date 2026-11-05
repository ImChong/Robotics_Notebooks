# frax: Fast Robot Kinematics and Dynamics in JAX

- **论文：** https://arxiv.org/abs/2604.04310v2
- **作者：** Daniel Morton、Marco Pavone
- **机构：** 斯坦福大学
- **版本：** arXiv v2，2026-05-15；已阅读全文、实验与附录。
- **代码：** https://github.com/StanfordASL/frax
- **项目文档：** https://stanfordasl.github.io/frax/
- **对 wiki 的映射：** [frax](../../wiki/entities/frax.md)

## 核对摘录

- 祖先掩码把树结构传播改为向量化运算，RNEA 与 CRBA 的渐近计算量增加，但有利 JIT、自动微分及加速器并行。
- 在 i9-14900KF 上，论文表 II 的 G1 差分 IK 控制器为 6.71 微秒、操作空间控制为 24.6 微秒；它们不是后续 `frax.ik` 数据集重定向吞吐量。
- 论文范围是运动学和动力学库；后续 `frax.ik` 重定向功能与计时需参照[官方文档](../sites/frax.md)和[代码](../repos/frax.md)，不能归给论文原实验。
