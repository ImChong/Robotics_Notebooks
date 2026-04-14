# contact_planning

> 来源归档（ingest）

- **标题：** 接触规划与接触隐式优化核心论文
- **类型：** paper
- **来源：** ICRA / IROS / IJRR / TRO
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 MICP 步位规划、接触隐式轨迹优化（CITO）、多接触运动合成等接触规划方法

## 核心论文摘录

### 1) Footstep Planning on Uneven Terrain with Mixed-Integer Convex Optimization（Deits & Tedrake, HUMANOIDS 2014）
- **链接：** <https://arxiv.org/abs/1403.7171>
- **核心贡献：** 将不平整地形步位规划表述为 MICP：整数变量选择接触区域，连续变量优化落脚点；用 IRIS 提取安全凸多面体；在 Atlas 验证多步规划；为接触规划与凸优化结合奠定工程基础
- **对 wiki 的映射：**
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

### 2) Contact-Invariant Optimization for Hand Manipulations（Mordatch et al., SCA 2012）
- **链接：** <https://homes.cs.washington.edu/~todorov/papers/MordatchSCA12.pdf>
- **核心贡献：** Contact-Invariant Optimization（CIO）：把接触序列作为软约束优化，而非离散规划；松弛互补条件为光滑连续问题；自动发现合理的接触序列；扩展为全身运动合成
- **关键洞见：** 互补约束松弛（ε-complementarity）= 把离散接触决策变为连续变量，可用梯度优化
- **对 wiki 的映射：**
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)

### 3) A Direct Method for Trajectory Optimization with Discontinuous Motion（Posa et al., IJRR 2014）
- **链接：** <https://journals.sagepub.com/doi/10.1177/0278364913490388>
- **核心贡献：** Direct Contact-Implicit Trajectory Optimization（DIRCOL + CI）：将互补约束（LCP）直接嵌入非线性规划；无需接触序列预定义；可优化跳跃/落地/翻滚等复杂接触轨迹
- **关键公式：** $0 \le \phi(q) \perp \lambda \ge 0$（互补条件作为 NLP 约束）
- **对 wiki 的映射：**
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

### 4) An Introduction to Trajectory Optimization（Tonneau et al., TRO 2018）
- **链接：** <https://ieeexplore.ieee.org/document/8276625>
- **核心贡献：** 综述多接触运动合成方法：接触序列规划 → 质心运动优化 → 全身运动合成三层架构；分类基于图的接触搜索、凸近似和数据驱动方法；覆盖 Atlas/HRP/CENTAURO 等平台验证
- **对 wiki 的映射：**
  - [footstep-planning](../../wiki/concepts/footstep-planning.md)
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

### 5) Efficient Multi-Contact Pattern Generation with Sequential Convex Approximations（Dai et al., ICRA 2014）
- **链接：** <https://ieeexplore.ieee.org/document/6907464>
- **核心贡献：** 用 SCA（Sequential Convex Approximation）迭代求解非凸多接触规划；将质心动量约束凸化；在 Atlas 上实现多步推箱子等全身操作任务
- **对 wiki 的映射：**
  - [contact-complementarity](../../wiki/formalizations/contact-complementarity.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
