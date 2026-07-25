# Variational-Based Optimal Control of Underactuated Balancing for Dynamic Quadrupeds

> 来源归档

- **标题：** Variational-Based Optimal Control of Underactuated Balancing for Dynamic Quadrupeds
- **类型：** paper
- **作者：** Matthew Chignoli, Patrick M. Wensing
- **机构：** University of Notre Dame / MIT（合作语境）
- **链接：** https://ieeexplore.ieee.org/abstract/document/9033976
- **DOI：** https://doi.org/10.1109/ACCESS.2020.2980446
- **OA PDF：** https://ieeexplore.ieee.org/ielx7/6287639/8948470/09033976.pdf
- **期刊：** IEEE Access 2020
- **入库日期：** 2026-07-25
- **一句话说明：** 欠驱动接触（如两点足）下的四足平衡：变分线性化 + 约束最优控制，以凸 QP 近似摩擦约束最优策略；Mini Cheetah 演示两点支撑扰动恢复与 CoM 出支撑域恢复。
- **开源状态：** **未开源**独立官方仓（截至入库日）；OA PDF 可获取。
- **沉淀到 wiki：** [paper-variational-underactuated-balancing-quadruped](../../wiki/entities/paper-variational-underactuated-balancing-quadruped.md)

---

## 核心贡献（摘录）

1. 将 cart-pendulum/acrobot 类欠驱动平衡思想迁移到摩擦受限、流形构型的腿足。
2. 凸 QP 实现，比完整 MPC 更紧凑。
3. 利用躯干角动量做两点支撑恢复。

## 对 wiki 的映射

- [paper-variational-underactuated-balancing-quadruped](../../wiki/entities/paper-variational-underactuated-balancing-quadruped.md)
- [whole-body-control](../../wiki/concepts/whole-body-control.md)
- [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)
