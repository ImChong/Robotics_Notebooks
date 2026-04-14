# contact_dynamics

> 来源归档（ingest）

- **标题：** 接触动力学核心论文
- **类型：** paper
- **来源：** IJRR / IEEE TRO / IROS / RSS
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖刚体接触建模、LCP 互补约束、MuJoCo 接触模型等接触动力学基础

## 核心论文摘录

### 1) Rigid Body Dynamics Algorithms（Featherstone, 2008）
- **链接：** <https://link.springer.com/book/10.1007/978-1-4899-7560-7>
- **核心贡献：** 系统介绍空间向量代数（spatial vector algebra）与 RNEA/CRBA/ABA 等 O(n) 递推算法；第 11 章详细讨论接触约束的 Impulse/LCP 处理
- **对 wiki 的映射：**
  - [contact-dynamics](../../wiki/concepts/contact-dynamics.md)
  - [floating-base-dynamics](../../wiki/concepts/floating-base-dynamics.md)

### 2) Rigid-Body Dynamics with Frictional Contacts（Stewart, 2000）
- **链接：** <https://epubs.siam.org/doi/10.1137/S0036144599360110>
- **核心贡献：** 提出 LCP（线性互补问题）框架对摩擦接触建模；证明在 Coulomb 摩擦下 LCP 解的存在性；奠定现代仿真器接触求解的理论基础
- **对 wiki 的映射：**
  - [contact-dynamics](../../wiki/concepts/contact-dynamics.md)

### 3) MuJoCo: A Physics Engine for Model-Based Control（Todorov et al., 2012）
- **链接：** <https://ieeexplore.ieee.org/document/6386109>
- **核心贡献：** 提出 soft-contact 模型（弹性接触而非 LCP 硬接触）+ 隐式积分器；接触力通过优化求解而非互补约束；显著提升机器人学习仿真速度和稳定性
- **对 wiki 的映射：**
  - [contact-dynamics](../../wiki/concepts/contact-dynamics.md)
  - [contact-estimation](../../wiki/concepts/contact-estimation.md)

### 4) Contact-Consistent Control Barrier Functions（Shirai et al., 2023）
- **链接：** <https://arxiv.org/abs/2311.01273>
- **核心贡献：** 在 CBF 安全约束框架中整合接触一致性；允许在接触切换时保持安全性；应用于双足机器人的接触感知运动规划
- **对 wiki 的映射：**
  - [contact-estimation](../../wiki/concepts/contact-estimation.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

### 5) Learning to Walk in Minutes Using Massively Parallel Deep RL（Rudin et al., 2022）
- **链接：** <https://arxiv.org/abs/2109.11978>
- **核心贡献：** 隐式接触建模（Isaac Gym 软接触）+ RL 策略直接学习接触处理；无需显式接触检测器；训练 10 分钟获得可迁移真机策略
- **对 wiki 的映射：**
  - [contact-estimation](../../wiki/concepts/contact-estimation.md)
  - [sim2real](../../wiki/concepts/sim2real.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
