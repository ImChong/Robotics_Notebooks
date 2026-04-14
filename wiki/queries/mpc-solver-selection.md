---
type: query
tags: [mpc, solver, osqp, qpoases, acados, optimization, legged-robots]
status: stable
---

# MPC 求解器选型指南

> **Query 产物**：本页由以下问题触发：「机器人 MPC 求解器怎么选：OSQP vs qpOASES vs Acados vs FORCES Pro？」
> 综合来源：[model-predictive-control.md](../methods/model-predictive-control.md)、[wbc-vs-rl.md](../comparisons/wbc-vs-rl.md)

---

## TL;DR 选型决策树

```
你的 MPC 问题是什么类型？
│
├─ 线性/凸 QP（线性动力学，二次代价）
│   ├─ 需要热启动 + 实时控制（< 5ms）
│   │   ├─ 小规模（< 50 变量）→ qpOASES
│   │   └─ 中大规模 → OSQP（暖启动效果好）
│   └─ 学术原型，不在乎速度 → CVXPY / scipy.optimize
│
├─ 非线性 MPC（非线性动力学，通用代价）
│   ├─ 需要商业支持 + 代码生成 → FORCES Pro
│   ├─ 开源 + 高性能 + 嵌入式友好 → Acados
│   └─ 全身动力学 + 接触规划 → Crocoddyl（DDP/FDDP）
│
└─ 凸 MPC（线性化足式机器人动力学）
    → MIT Cheetah 风格 → OSQP / qpOASES 均可
    → ETH RSL 风格 → OCS2 (Hpipm/OSQP 后端)
```

---

## 求解器对比表

| 求解器 | 类型 | 实时性 | 开源 | 代码生成 | 适用场景 |
|--------|------|--------|------|---------|---------|
| **OSQP** | QP（ADMM） | ✅ 极快 | ✅ | ✅（Python/C） | 凸 MPC、WBC QP、标准工具 |
| **qpOASES** | QP（Active Set） | ✅ 快 | ✅ | ✅ | 小规模 QP、热启动场景 |
| **Acados** | NLP/OCP | ✅ 很快 | ✅ | ✅（C） | 非线性 MPC，嵌入式 |
| **FORCES Pro** | NLP/OCP | ✅ 极快 | ❌ 商业 | ✅（定制） | 商业产品，最快 NLP |
| **IPOPT** | NLP | ⚠️ 慢 | ✅ | ❌ | 离线规划、TO |
| **Crocoddyl** | DDP/OCP | ✅ 快 | ✅ | ❌ | 全身 TO，接触规划 |
| **HPIPM** | QP（结构化） | ✅ 极快 | ✅ | ✅ | 线性化 MPC，内嵌 OCS2 |

---

## OSQP — 凸 MPC 首选

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

**MPC 使用模式**：每个时间步更新 `l`, `u`（约束更新），调用 `prob.update(l=l_new, u=u_new)` + `prob.solve()`。

---

## qpOASES — 小规模 QP 的老可靠

**适用**：小规模（≤ 100 变量）、需要 Active Set 策略（精确活跃约束集）

**特点**：
- Active Set 方法，在约束数量少时极快
- 热启动支持（继承上一步活跃约束集）
- 在 legged_gym / raisin 早期实现中广泛使用

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

**特点**：
- 基于 **SQP / RTI**（Real-Time Iteration）算法
- **代码生成**：将 OCP 编译为纯 C 代码，嵌入式友好
- 支持多种后端：HPIPM（默认）、OSQP、qpOASES
- `acados_template`：Python 问题定义 → 自动生成 C 求解器

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

## FORCES Pro — 商业最快选项

**适用**：要求极限实时性（< 1ms）、有商业授权预算

**特点**：
- 针对特定问题结构生成高度优化的求解器代码
- 比通用求解器快 5-20x
- 学术版免费，商业版付费
- 集成于多个工业机器人产品

---

## 场景→求解器快速匹配

| 场景 | 推荐求解器 | 理由 |
|------|-----------|------|
| 凸 MPC 足式机器人（MIT 风格） | **OSQP** | 稀疏 QP，暖启动，50Hz 可达 |
| WBC QP（任务空间控制） | **OSQP** | 标准凸 QP |
| 非线性 MPC（关节空间） | **Acados** | 开源 NLP，代码生成，1kHz 可达 |
| 全身运动生成 / TO | **Crocoddyl** | DDP 特别适合全身轨迹优化 |
| OCS2 框架内 | **HPIPM** | OCS2 默认后端，已针对 MPC 优化 |
| 商业产品 / 极限实时 | **FORCES Pro** | 工业级最快 |

---

## 参考来源

- [sources/papers/mpc.md](../../sources/papers/mpc.md) — ingest 档案（MPC 核心论文）
- OSQP 论文：Stellato et al., *OSQP: An Operator Splitting Solver for Quadratic Programs* (2020)
- Acados 论文：Verschueren et al., *acados — a Modular Open-Source Framework for Fast Embedded Optimal Control* (2021)

---

## 关联页面

- [Model Predictive Control](../methods/model-predictive-control.md) — MPC 理论与机器人应用
- [Whole-Body Control](../concepts/whole-body-control.md) — WBC QP 同样需要 QP 求解器
- [Crocoddyl](../entities/crocoddyl.md) — 全身 TO 的 DDP 框架
- [TSID](../concepts/tsid.md) — 任务空间逆动力学，依赖 QP 求解器

---

## 一句话记忆

> 凸 MPC/WBC 用 OSQP（稀疏+暖启动），非线性 MPC 用 Acados（代码生成+嵌入式），全身 TO 用 Crocoddyl（DDP），要最快的商业项目上 FORCES Pro。
