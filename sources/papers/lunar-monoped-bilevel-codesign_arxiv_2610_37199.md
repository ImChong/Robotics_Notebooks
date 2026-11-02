# 月球单足协同设计（arXiv:2610.37199）

> 来源归档（paper）

- **标题：** A Bi-level Co-design Framework for Lunar Monoped with 2D Jump and Landing Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37199>
- **PDF：** <https://arxiv.org/pdf/2610.37199>
- **作者：** Ramkumar, Sanjay, Kumar, Suraj
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 面向月球单足机器人二维跳跃与着陆的双层协同设计。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 月球低重力会延长腾空时间并减小接触力，使腿部尺寸和跳跃控制器高度耦合。
3. **周报所述创新点：** 外层差分进化搜索腿段长度，内层分别使用 QP 任务空间控制和直接配点轨迹优化评估设计；进一步加入角动量调节和 LQR 落地稳定控制。
4. **arXiv 摘要（英文节选）：** Legged robots for lunar exploration must be designed for low gravity, where reduced contact forces and long flight phases make leg morphology and controller choice interdependent. We study this for a planar monoped in lunar gravity ($1.625$ m/s$^2$) using a bi-level co-design framework: a differential evolution outer loop searches over link lengths, and each candidate is evaluated on a vertical jump by either a quadratic program based task-space controller (QP-TSC) or a direct-collocation based trajectory optimization. The two controllers yield different optimal link lengths, and trajectory optimization outperforms the QP-TSC across most of the design space, with 17% lower Cost of Transport (COT) at their respective optima.

**对 wiki 的映射**

- [paper-lunar-monoped-bilevel-codesign](../../wiki/entities/paper-lunar-monoped-bilevel-codesign.md)
