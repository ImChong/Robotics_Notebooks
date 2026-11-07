---
type: query
tags: [mpc, solver, osqp, qpoases, acados, optimization, legged-robots]
status: complete
summary: "MPC 求解器选型指南"
updated: 2026-11-07
sources:
  - ../../sources/papers/osqp_admm_qp_arxiv_1711_08013.md
  - ../../sources/papers/hpipm_arxiv_2003_02547.md
  - ../../sources/papers/acados_arxiv_1910_13753.md
  - ../../sources/papers/crocoddyl_mastalli_icra_2020.md
  - ../../sources/repos/osqp.md
  - ../../sources/papers/mpc.md
  - ../../sources/papers/pi_mpc_arxiv_2601_14414.md
original:
  - ../../sources/papers/osqp_admm_qp_arxiv_1711_08013.md
  - ../../sources/papers/hpipm_arxiv_2003_02547.md
  - ../../sources/papers/acados_arxiv_1910_13753.md
  - ../../sources/papers/crocoddyl_mastalli_icra_2020.md
  - ../../sources/repos/osqp.md
---

# MPC 求解器选型指南

> **Query 产物**：本页由以下问题触发：「机器人 MPC 求解器怎么选：OSQP vs qpOASES vs Acados vs FORCES Pro？」
> 综合来源：[model-predictive-control.md](../methods/model-predictive-control.md)、[wbc-vs-rl.md](../comparisons/wbc-vs-rl.md)

---

## TL;DR 选型决策树

> **读法**：下面的分支里，凡没有标注出处的阈值（如"< 50 变量""< 5 ms"）都是经验性说法（推测），已核对的数据点集中写在"已核对的原文数据"一节。

```
你的 MPC 问题是什么类型？
│
├─ 线性/凸 QP（线性动力学，二次代价）
│   ├─ 有多阶段 OCP 结构（线性化 MPC / RTI 的每步 QP）
│   │   └─ 首选结构化内点法 HPIPM（部分凝聚后最快，耗时稳定）；
│   │      OSQP / qpOASES 在难的时段耗时会变长（HPIPM 论文基准）
│   ├─ 一般稀疏 QP，需反复求解（参数化）
│   │   └─ OSQP（分解缓存 + warm start；中等精度）
│   ├─ 小规模稠密 QP，相邻步有效集变化小 → qpOASES（有效集，易 warm start）
│   └─ 学术原型，不在乎速度 → CVXPY / scipy.optimize
│
├─ 非线性 MPC（非线性动力学，通用代价）
│   ├─ 开源 + 嵌入式友好 + RTI → acados（默认 QP 层 HPIPM）
│   ├─ 商业支持 + 代码生成 → FORCES Pro（本页未核对其性能数字）
│   └─ 全身动力学 + 预定接触序列 → Crocoddyl（DDP/FDDP）
│
└─ 凸 MPC（线性化足式机器人动力学）
    → MIT Cheetah 风格：线性化后得到凸 QP，任选上面的 QP 求解器
    → ETH RSL 风格 → OCS2（QP 后端的具体选择本次未核对）
```

---

## 求解器对比表

| 求解器 | 类型 | 实时性（已核对的说法） | 开源 | 代码生成 | 适用场景 |
|--------|------|--------|------|---------|---------|
| **OSQP** | QP（ADMM） | 论文：典型比内点法快约 10 倍，默认中等精度；warm start + 分解缓存对参数化 QP 提速 10 倍以上 | ✅ | ✅（可编译成无库、无除法的 C） | 凸 MPC、WBC QP、通用 QP |
| **qpOASES** | QP（有效集） | 易 warm start；HPIPM 论文基准中小规模 QP 约 30% 最快、其余常慢 10 倍以上；最坏复杂度随约束数指数增长（OSQP 论文综述） | ✅ | — | 小规模 QP、热启动场景 |
| **HPIPM** | 结构化 QP（内点法） | 论文：部分凝聚的 OCP QP 稳定低于 1 ms，耗时几乎恒定 | ✅ | 否（C 库，BLASFEO） | 线性化 MPC / RTI 的每步 QP；acados 的默认 QP 层 |
| **acados** | NLP/OCP（SQP-RTI） | 论文：链式质量块中位约 1 ms；PowerPC 900 MHz 上 N=20 的 NMPC 为毫秒量级 | ✅ | 高层接口生成 C 工程（核心库不依赖代码生成） | 非线性 MPC、嵌入式 |
| **FORCES Pro** | NLP/OCP（内点法） | 商业；acados 论文把它列为结构化内点法代表，但**未做数值对比**，"极快""快 5–20×"等旧说法无原文依据（推测，已删除数字） | ❌ 商业 | ✅（定制） | 商业产品 |
| **IPOPT** | NLP | acados 论文基准：中位约 60 ms，作非嵌入式基线 | ✅ | ❌ | 离线规划、TO |
| **Crocoddyl** | DDP/FDDP（OCP） | 论文：单次迭代最高 859.6 Hz（60 节点，多线程）；整个机动 10–34 次迭代、<0.5 s；**需预定义接触序列，基准未含摩擦锥 / 力矩限** | ✅ | ❌ | 全身 TO、接触序列给定的 MPC |
| **π MPC / πⁿ MPC** | LTV MPC（ADMM） | 技术报告：小规模 OSQP 最快；π MPC 在 GPU 上耗时几乎不随规模增长，适合长时域（仅据摘要与数值实验核对） | ✅ | ❌ | parallel-in-horizon、construction-free |

## OSQP — 通用凸 QP / 参数化 QP

**适用**：凸 QP，机器人 WBC、凸 MPC（MIT/ETH 风格）

**特点**：
- ADMM 算法，对大规模稀疏问题友好
- 暖启动（warm start）显著加速 MPC 迭代
- Python (`osqp`) / C 接口，易集成

```python
import osqp
import numpy as np
from scipy import sparse

prob = osqp.OSQP()
prob.setup(P, q, A, l, u,
           warm_starting=True,
           eps_abs=1e-4, eps_rel=1e-4,
           max_iter=1000,
           verbose=False)
res = prob.solve()
```

**MPC 使用模式**：每个时间步更新 `l`, `u`（约束更新），调用 `prob.update(l=l_new, u=u_new)` + `prob.solve()`；仅向量变化时 KKT 分解可复用（论文 §6，官方 Python 文档 `m.update(q=, l=, u=)`）。官方默认 `eps_abs=1e-3`、`max_iter=4000`、`warm_starting=True`，示例里的 `1e-4 / 1000` 是调紧精度的写法。

---

## qpOASES — 小规模 QP 的老可靠

**适用**：小规模、需要 Active Set 策略（精确活跃约束集）；"≤ 100 变量"这类阈值为经验说法（推测）

**特点**：
- Active Set 方法，易热启动（继承上一步活跃约束集）；acados 论文称其配合凝聚法特别适合 (N)MPC
- 最坏复杂度随约束数指数增长（OSQP 论文综述）；HPIPM 论文基准里它在难题上常被内点法拉开一个数量级
- 旧版称"在 legged_gym / raisin 早期实现中广泛使用"，本次未核对（推测）

```python
import qpoases
QP = qpoases.PyQProblem(n_var, n_constraints)
options = qpoases.PyOptions()
options.setToMPC()  # 预设 MPC 参数（放松精度换速度）
QP.init(H, g, A, lb, ub, lbA, ubA, nWSR)
QP.hotstart(H_new, g_new, A_new, lb_new, ub_new, lbA_new, ubA_new, nWSR)
```

---

## Acados — 非线性 MPC 开源首选

**适用**：非线性 MPC，嵌入式实时控制，需要代码生成

**特点**（对照 acados 论文）：
- 基于 **SQP / RTI**（Real-Time Iteration）算法：每个采样时刻只做一次完整 SQP 迭代；**不含全局化**（线搜索 / 信赖域），因为实时场景无法给出时间上界
- 核心库是模块化 C 库；Python / Matlab 高层接口借 CasADi 生成函数并生成 C 工程，嵌入式友好
- QP 层以 HPIPM 为基础（默认），另有 HPMPC、qpDUNES、qpOASES、OSQP 接口
- Hessian 近似：Gauss-Newton、SCQP、带凸化的精确 Hessian；支持部分凝聚

```python
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# 定义模型（CasADi 符号动力学）
model = AcadosModel()
model.x = cas.vertcat(pos, vel)
model.u = cas.vertcat(forces)
model.f_expl_expr = robot_dynamics(model.x, model.u)

# 定义 OCP
ocp = AcadosOcp()
ocp.model = model
ocp.cost.cost_type = 'LINEAR_LS'
ocp.dims.N = 20  # 预测步数

# 生成并编译 C 求解器
solver = AcadosOcpSolver(ocp, json_file="ocp.json")
```

---

## FORCES Pro — 商业结构化内点法

**适用**：商业授权预算内、需要结构化内点法 NLP/OCP 求解器

**特点**：
- 针对特定问题结构生成求解器代码；acados 论文把它归为"结构化内点法"一类，并指出代码生成的线性代数在 10×10 到 100×100 的典型 MPC 规模上不如优化库（BLASFEO）
- 旧版"比通用求解器快 5–20 倍""学术版免费""集成于多个工业机器人产品"在本次读到的原文里**都没有出现**，已按推测处理，不作为选型依据；需要时请直接评估官方基准与授权条款

---

## 场景→求解器快速匹配

| 场景 | 推荐求解器 | 理由 |
|------|-----------|------|
| 凸 MPC 足式机器人（MIT 风格） | **OSQP / HPIPM / qpOASES** | 线性化后是凸 QP；MIT WBIC 论文中的凸 MPC 以约 30–40 Hz 运行（原文未写所用 QP 求解器） |
| WBC QP（任务空间控制） | **OSQP** | 标准凸 QP |
| 非线性 MPC（关节空间） | **Acados** | 开源 NLP，RTI；论文给出毫秒量级的嵌入式实测（1 kHz 无原文数据） |
| 全身运动生成 / TO | **Crocoddyl** | DDP 特别适合全身轨迹优化 |
| OCS2 框架内 | **HPIPM** | HPIPM 为 MPC 专用结构化内点法；"OCS2 默认后端"本次未核对（推测） |
| 商业产品 | **FORCES Pro** | 商业结构化内点法（性能优劣本次未核对） |
| 代价/动力学不可微（接触开合、稀疏奖励、黑盒仿真） | **[MPPI](../methods/mppi.md)** 等采样式 | 本页其余求解器都要梯度/凸结构；MPPI 只需前向 rollout，代价换成 GPU 并行采样量 |

> **边界提醒：** MPPI 与上表其余求解器不是同一档竞争者——前者是**免梯度采样式**滚动优化，不给 KKT 意义上的最优性，也没有暖启动收敛保证；当模型可微且能写成 QP/NLP 时，OSQP / Acados 一类仍是更稳更省算力的选择。TD-MPC2 系 MBRL（见 [CAST](../entities/paper-cast-mbrl.md)）用的就是 MPPI 族在学习到的潜动力学上做在线规划。

---

## 已核对的原文数据

- **OSQP 论文**（1400 个基准实例，$n$ 到 $10^4$、$m$ 到 $10^5$）：OSQP 相对 ECOS / GUROBI / MOSEK 总体最好；对比对象**不含 qpOASES**。ADMM 默认中等精度，polishing 可提到内点法水平；参数化问题中仅向量变化时可复用分解。
- **HPIPM 论文**（MPC 结构化 QP）：HPIPM（部分凝聚）<1 ms 且稳定；qpOASES / OSQP 在问题难的阶段耗时上升；Maros-Mészáros 43 题上 qpOASES 约 30% 最快。基准机器为单线程 Core i7，不外推到别的硬件。
- **acados 论文**：链式质量块闭环 MPC 中位耗时 acados≈1.05 ms、GRAMPC≈1.06 ms、ACADO≈1.97 ms、IPOPT≈59.84 ms；PowerPC 900 MHz 上 N=20 的 NMPC 毫秒量级；未与 FORCES 对比。
- **Crocoddyl 论文**：FDDP 处理预定义接触序列的多接触 OCP，10–34 次迭代；作者在评测中刻意忽略摩擦锥与力矩限，称可用二次惩罚加入。
- **π MPC 技术报告**：ADMM 型 LTV MPC；小规模 OSQP 最快，π MPC（GPU）规模扩大时耗时几乎不变，CSTR 例子中 π MPC 胜在省去 QP 矩阵构造；仅核对到摘要与数值实验。

> 旧版文中的"MPC 前沿"代码段（OSQP / qpOASES / acados）为用法示意，除 OSQP 的 `setup / update / solve` 与设置名已对照官方文档外，其余接口名未逐一核对。

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| OCP | Optimal Control Problem | MPC 每步求解的有限时域最优控制问题 |
| legged_gym | Legged Gym | 足式机器人 RL 训练的常用开源框架 |
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |

## 参考来源

- [OSQP 论文通读归档](../../sources/papers/osqp_admm_qp_arxiv_1711_08013.md) — Stellato et al. 2020, *OSQP: An Operator Splitting Solver for Quadratic Programs*
- [HPIPM 论文通读归档](../../sources/papers/hpipm_arxiv_2003_02547.md) — Frison & Diehl 2020
- [acados 论文通读归档](../../sources/papers/acados_arxiv_1910_13753.md) — Verschueren et al. 2022
- [Crocoddyl 论文通读归档](../../sources/papers/crocoddyl_mastalli_icra_2020.md) — Mastalli et al. 2020
- [OSQP 仓库归档（README 与文档核对）](../../sources/repos/osqp.md)
- [π MPC 归档](../../sources/papers/pi_mpc_arxiv_2601_14414.md)（通读未完成，仅核对摘要与数值实验）
- [sources/papers/mpc.md](../../sources/papers/mpc.md) — ingest 档案（MPC 核心论文，二手整理）

---

## 关联页面

- [Model Predictive Control](../methods/model-predictive-control.md) — MPC 理论与机器人应用
- [Whole-Body Control](../concepts/whole-body-control.md) — WBC QP 同样需要 QP 求解器
- [Crocoddyl](../entities/crocoddyl.md) — 全身 TO 的 DDP 框架
- [SE(3) 切空间浮动基 TO](../entities/paper-se3-tangent-to.md) — 离线敏捷 TO 用切空间 + Ipopt，不是嵌入式 QP
- [TSID](../concepts/tsid.md) — 任务空间逆动力学，依赖 QP 求解器
- [π MPC](../methods/pi-mpc.md) — parallel-in-horizon ADMM；[MPC-RL](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) 批训练应用
- [MPPI](../methods/mppi.md) — 免梯度采样式滚动优化，本页各求解器的「模型不可微」逃生口

---

## 一句话记忆

> 通用 / 参数化凸 QP 用 OSQP（分解缓存 + warm start），线性化 MPC 的每步 OCP QP 优先试 HPIPM，非线性 MPC 用 acados（RTI），预定接触序列的全身 TO 用 Crocoddyl；商业的 FORCES Pro 本页未做性能核对。

## 推荐继续阅读

- OSQP：<https://arxiv.org/abs/1711.08013> 与官方文档 <https://osqp.org/>
- HPIPM：<https://arxiv.org/abs/2003.02547>
- acados：<https://arxiv.org/abs/1910.13753>
