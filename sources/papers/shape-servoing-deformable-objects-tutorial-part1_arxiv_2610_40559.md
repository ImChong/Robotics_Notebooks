# 形状伺服教程 Part I（arXiv:2610.40559）

> 来源归档（paper）

- **标题：** A Tutorial on Shape Servoing of Deformable Objects-Part I: Fundamentals
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40559>
- **PDF：** <https://arxiv.org/pdf/2610.40559>
- **作者：** Cuiral-Zueco, Ignacio, Caporali, Alessio, Aranda, Miguel, Palli, Gianluca, López-Nicolás, Gonzalo, Mezouar, Youcef
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 可变形物体形状伺服教程（一）：基础原理。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 机器人控制柔性物体达到目标形状时，需要知道机器人运动如何影响物体形变，但该关系依赖材料、几何和抓取边界条件，难以准确建模。
3. **周报所述创新点：** 系统介绍基于形变雅可比矩阵的形状伺服方法，包括形状表示、离线估计、在线 Broyden 更新、闭环控制律及局部稳定性分析。使用双臂操作弹性杆的仿真实例解释完整控制流程。
4. **arXiv 摘要（英文节选）：** In robotics, shape control (or shape servoing) refers to the automatic manipulation of a deformable object to achieve a desired shape. It involves a feedback loop that exploits the relation between robot motion and changes in the measured shape. This relation, known as the interaction Jacobian or, in this context, the deformation Jacobian, depends on the object's mechanical properties, geometry, and boundary conditions (e.g., where the object is grasped).

**对 wiki 的映射**

- [paper-shape-servoing-deformable-objects-tutorial-part1](../../wiki/entities/paper-shape-servoing-deformable-objects-tutorial-part1.md)
