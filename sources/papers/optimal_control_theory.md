# 最优控制理论（Optimal Control Theory）

> Ingest 日期：2026-04-15
> 主题：最优控制基础、轨迹优化、数值方法、非线性 MPC

---

## 核心论文 / 教材

### Betts (2010) — Practical Methods for Optimal Control and Estimation Using Nonlinear Programming
- **核心贡献**：系统介绍直接法（direct collocation / shooting）在最优控制中的应用
- **关键洞见**：将连续最优控制问题离散化为非线性规划（NLP），是轨迹优化主流数值方法的基础

### Diehl et al. (2006) — Fast Direct Multiple Shooting Algorithms for Optimal Robot Control
- **核心贡献**：多重打靶（multiple shooting）结合实时迭代（RTI）求解非线性 MPC
- **关键洞见**：RTI 方案将每步 MPC 求解时间降到毫秒级，是快速腿式机器人 MPC 的理论支柱

### Kelly (2017) — An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation
- **核心贡献**：直接配点法（direct collocation）教程，从 ODE 约束到 NLP 建立的完整流程
- **关键洞见**：trapezoidal / Hermite-Simpson 配点方案；适合机器人运动规划入门

### Nocedal & Wright (2006) — Numerical Optimization (2nd Edition)
- **核心贡献**：SQP、内点法、拟牛顿法等 NLP 求解器的理论基础
- **关键洞见**：SNOPT / IPOPT 等机器人常用求解器的算法来源；理解求解器行为的必读教材

### Todorov & Li (2004) — Optimal Control Methods Suitable for Biomechanical Systems
- **核心贡献**：iLQR（迭代线性二次调节器）的实用版本，计算复杂度低
- **关键洞见**：iLQR 是 DDP（差分动态规划）的简化形式；适合高自由度机器人运动规划

---

## Wiki 映射

| 论文 / 概念 | 对应 wiki 页面 |
|-----------|--------------|
| 直接配点 / 多重打靶 | `wiki/concepts/optimal-control.md` |
| 实时迭代 MPC（RTI） | `wiki/methods/model-predictive-control.md` |
| 轨迹优化数值方法 | `wiki/concepts/trajectory-optimization.md` |
| NLP 求解器（IPOPT / SNOPT） | `wiki/concepts/optimal-control.md` |
| iLQR / DDP | `wiki/concepts/trajectory-optimization.md` |

---

## 关键结论

1. **直接法是机器人轨迹优化主流**：直接配点和多重打靶将轨迹优化转化为大规模稀疏 NLP，利用成熟求解器（IPOPT/SNOPT）高效求解。
2. **RTI 是快速 MPC 的关键**：实时迭代方案在每个控制周期内只进行一次 SQP 迭代，牺牲最优性换取实时性，适合腿式机器人 MPC（~10ms 周期）。
3. **iLQR/DDP 是轨迹优化备选**：相比 NLP，iLQR 不需要外部求解器，适合 GPU 加速的批量轨迹优化（如 RL 中的 model-based planning）。
4. **约束处理是核心挑战**：接触约束（互补条件）的光滑化是机器人轨迹优化中最难处理的部分，直接影响 sim2real 效果。

---

## 参考来源
- Betts (2010), SIAM
- Diehl et al., Fast Motions in Biomechanics and Robotics (2006)
- Kelly (2017), SIAM Review
- Nocedal & Wright (2006), Springer
- Todorov & Li (2004), ACC
