# ppf_contact_solver_tog_cubic_barrier

> 来源归档

- **标题：** A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness
- **类型：** paper
- **venue：** ACM Transactions on Graphics (TOG), Vol.43, No.6
- **DOI：** https://dl.acm.org/doi/abs/10.1145/3687908
- **作者 / 机构：** Ryoichi Ando 等，ZOZO, Inc.（st-tech）
- **实现：** https://github.com/st-tech/ppf-contact-solver
- **入库日期：** 2026-05-25
- **一句话说明：** 提出带弹性刚度项的三次障碍接触模型，使接触力 Jacobian 与 FEM 弹性在同一牛顿步内一致耦合，支撑 shell/solid/rod 大规模无穿透 GPU 仿真。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-ppf-cubic-barrier-contact-solver.md`](../../wiki/entities/paper-ppf-cubic-barrier-contact-solver.md)

---

## 核心贡献（README / 技术材料归纳）

1. **Cubic barrier 接触势：** 相对传统对数/二次障碍，在接近穿透时仍保持可积、可微的刚度行为，利于与弹性项联合牛顿求解。
2. **Elasticity-inclusive dynamic stiffness：** 接触矩阵组装时显式纳入弹性模态对「有效刚度」的贡献，减少接触–弹性分裂迭代中的虚假振荡或穿透。
3. **工程目标：** 单精度 GPU 上 **百万–亿级** 接触、**无穿透**、布料 **应变上限**（非橡胶拉伸）、摩擦与自碰等复杂场景（domino、woven、five-twist 等示例）。

## 复现与代码分支

- **主分支 `main`：** 持续迭代，API 可能 breaking；性能与功能优于论文分支。
- **`sigasia-2024`：** 与 TOG 论文一致的参考实现；仅维护，**不推荐**日常性能使用；排除 [`articles/bug.md`](https://github.com/st-tech/ppf-contact-solver/blob/main/articles/bug.md) 所列后续算法修复。

## 延伸阅读（仓库内）

- [Singular-value eigenanalysis](https://github.com/st-tech/ppf-contact-solver/blob/main/articles/eigensys.md)
- [Hindsight / ACCD 浮点舍入](https://github.com/st-tech/ppf-contact-solver/blob/main/articles/hindsight.md)

## 对 wiki 的映射

| 条目 | 目标 |
|------|------|
| 论文实体页 | [`wiki/entities/paper-ppf-cubic-barrier-contact-solver.md`](../../wiki/entities/paper-ppf-cubic-barrier-contact-solver.md) |
| 开源实现 | [`wiki/entities/ppf-contact-solver.md`](../../wiki/entities/ppf-contact-solver.md) |
| 原始仓库归档 | [`sources/repos/ppf-contact-solver.md`](../repos/ppf-contact-solver.md) |
