# contact_planning

> 来源归档（ingest）

- **标题：** 接触规划与接触隐式优化核心论文
- **类型：** paper
- **来源：** ICRA / IROS / IJRR / TRO
- **入库日期：** 2026-04-14
- **最后更新：** 2026-11-22
- **通读：** 已完成
- **核对范围：** 通读 Deits & Tedrake (HUMANOIDS 2014, [MIT CSAIL PDF](https://groups.csail.mit.edu/robotics-center/public_papers/Deits14a.pdf) — 已修正旧版误标的 astrophysics `arXiv:1403.7171` 链接)、Mordatch et al. (SCA 2012, [Eurographics DL Open PDF](https://diglib.eg.org/items/5e09458c-a0c5-4899-a46d-2a233b0f82bd))、Posa et al. (RSS 2013 MIT 开放 PDF `Posa13a.pdf` 与 [MIT DSpace 条目](https://dspace.mit.edu/handle/1721.1/93007))、Dai et al. (ICRA/HUMANOIDS 2014, [MIT CSAIL PDF](https://groups.csail.mit.edu/robotics-center/public_papers/Dai14.pdf))、Tonneau et al. (T-RO 2018 *An Efficient Acyclic Contact Planner for Multiped Robots*, [HAL `hal-01267345`](https://hal.science/hal-01267345))；并核对 Matthew P. Kelly (SIAM Review 2017) 轨迹优化综述以区分旧版标题混写。
- **一句话说明：** 覆盖 MICP 步位规划、接触隐式轨迹优化（CITO）、多接触运动合成等接触规划方法

## 核心论文摘录

### 1) Footstep Planning on Uneven Terrain with Mixed-Integer Convex Optimization（Deits & Tedrake, HUMANOIDS 2014）
- **链接：** <https://groups.csail.mit.edu/robotics-center/public_papers/Deits14a.pdf>（注：旧版曾误连 `arXiv:1403.7171` 为天体物理论文，现已勘误）
- **核心贡献：** 将不平整地形步位规划表述为 MICP：整数变量选择接触区域，连续变量优化落脚点；用 IRIS 提取安全凸多面体；在 Atlas 验证多步规划；为接触规划与凸优化结合奠定工程基础
- **对 wiki 的映射：**
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

### 2) Contact-Invariant Optimization for Hand Manipulation（Mordatch et al., SCA 2012）
- **链接：** <https://diglib.eg.org/items/5e09458c-a0c5-4899-a46d-2a233b0f82bd>（Eurographics DL Open Access `137-144.pdf`）
- **核心贡献：** Contact-Invariant Optimization（CIO）：把接触序列作为软约束优化，而非离散规划；松弛互补条件为光滑连续问题；自动发现合理的接触序列；扩展为手部操作与全身运动合成
- **关键洞见：** 互补约束松弛（ε-complementarity）= 把离散接触决策变为连续变量，可用梯度优化
- **对 wiki 的映射：**
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)

### 3) A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact（Posa et al., IJRR 2014）
- **链接：** <https://journals.sagepub.com/doi/10.1177/0278364913490388> · [MIT DSpace 条目](https://dspace.mit.edu/handle/1721.1/93007) · [同组接触稳定开源 PDF `Posa13a.pdf`](https://groups.csail.mit.edu/robotics-center/public_papers/Posa13a.pdf)
- **核心贡献：** Direct Contact-Implicit Trajectory Optimization（DIRCOL + CI）：将互补约束（LCP）直接嵌入非线性规划；无需接触序列预定义；可优化跳跃/落地/翻滚等复杂接触轨迹
- **关键公式：** $0 \le \phi(q) \perp \lambda \ge 0$（互补条件作为 NLP 约束）
- **对 wiki 的映射：**
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

### 4) An Efficient Acyclic Contact Planner for Multiped Robots（Tonneau et al., TRO 2018）
- **链接：** <https://ieeexplore.ieee.org/document/8276625> · [HAL 开放预印本 `hal-01267345`](https://hal.science/hal-01267345)（注：旧版曾将此处标题误写为 “An Introduction to Trajectory Optimization”，已据 IEEE/HAL 原文修正；后者为 Matthew P. Kelly 2017 *SIAM Review* 综述）
- **核心贡献：** 提出基于可达性条件与接触空间凸分解的非循环多足接触序列规划器：接触序列规划 → 质心轨迹/步态优化 → 全身运动合成层级架构，在 Atlas / HRP-2 等多接触场景验证
- **对 wiki 的映射：**
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

### 5) Whole-Body Motion Planning with Centroidal Dynamics and Full Kinematics（Dai, Valenzuela & Tedrake, HUMANOIDS 2014）
- **链接：** <https://groups.csail.mit.edu/robotics-center/public_papers/Dai14.pdf>（IEEE <https://ieeexplore.ieee.org/document/7041375>）
- **核心贡献：** 结合完整运动学（IK）与质心动量动力学（Centroidal Dynamics）的全身接触/轨迹直接优化，通过质心线性/角动量与支撑点力矩平衡约束在 Atlas 等平台生成动态多接触运动
- **对 wiki 的映射：**
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
