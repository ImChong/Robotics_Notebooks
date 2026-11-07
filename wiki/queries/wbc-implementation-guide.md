---
title: WBC 工程实现指南
type: query
status: complete
created: 2026-04-14
updated: 2026-11-07
summary: 从零实现 Whole-Body Control 的工程步骤：URDF 建模 → 动力学计算 → QP 求解 → 任务配置 → 实机调试，附工具链选型和常见问题。
sources:
  - ../../sources/papers/delprete_tsid_arxiv_1410_3863.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
  - ../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md
  - ../../sources/papers/osqp_admm_qp_arxiv_1711_08013.md
  - ../../sources/repos/pinocchio.md
  - ../../sources/repos/tsid.md
  - ../../sources/repos/osqp.md
  - ../../sources/papers/whole_body_control.md
  - ../../sources/papers/null_space_control.md
original:
  - ../../sources/papers/delprete_tsid_arxiv_1410_3863.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
  - ../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md
  - ../../sources/papers/osqp_admm_qp_arxiv_1711_08013.md
  - ../../sources/repos/pinocchio.md
  - ../../sources/repos/tsid.md
  - ../../sources/repos/osqp.md
---

> **Query 产物**：本页由以下问题触发：「如何从零搭建一个 WBC 控制器？」
> 综合来源：[Whole-Body Control](../concepts/whole-body-control.md)、[TSID](../concepts/tsid.md)、[HQP](../concepts/hqp.md)、[MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md)

# WBC 工程实现指南

## 实现路线概览

```
Step 1: 机器人模型准备（URDF/MJCF → Pinocchio）
Step 2: 动力学计算（M, C, g, J）
Step 3: 任务/约束定义
Step 4: QP 构建与求解
Step 5: 力矩映射与输出
Step 6: 实机调试
```

---

## Step 1：机器人模型准备

### 工具选择：Pinocchio（推荐）

```python
import pinocchio as pin

model = pin.buildModelFromUrdf("robot.urdf")
data  = model.createData()

# 加载 meshcat 可视化（可选）
from pinocchio.visualize import MeshcatVisualizer
```

**关键接口：**
```python
# 更新运动学
pin.computeAllTerms(model, data, q, v)

# 获取质量矩阵 M(q)
M = data.M  # shape: (nv, nv)；注意：CRBA 只填上三角，用前须对称化
M = np.triu(M) + np.triu(M, 1).T

# 获取非线性力 h(q,v) = C(q,v)v + g(q)
h = data.nle  # shape: (nv,)

# 获取末端执行器雅可比 J_ee（须先算过关节雅可比；computeAllTerms 已包含）
J = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
```

> **API 核对（Pinocchio 头文件）**：`computeAllTerms` 等价于一次性调用 FK、CRBA、`nonLinearEffects`、`computeJointJacobians`、质心等；`crba` 文档明确只计算 $M$ 的**上三角**；`getFrameJacobian` 文档要求先调用 `computeJointJacobians`，参考系有 LOCAL / LOCAL_WORLD_ALIGNED / WORLD；偏置项 $h$ 存在 `data.nle`。`\dot J \dot q` 项可用 `getFrameClassicalAcceleration`（零加速度前向运动学后读取）取得，比用 $\dot J$ 乘 $\dot q$ 更高效。

**模型质量检查清单：**
- [ ] 质量、惯量、质心位置正确（与 CAD/实体一致）；浮动基座机器人用 `buildModelFromUrdf` 时需显式指定 `JointModelFreeFlyer()`（推测，按 Pinocchio 常规用法，未在本次核对范围）
- [ ] 关节限制（位置/速度/力矩）已填写
- [ ] 接触脚的 frame_id 已确认
- [ ] 无悬空/孤立 link

---

## Step 2：动力学方程

WBC 基于刚体动力学方程求解：

$$M(q)\ddot{q} + h(q,\dot{q}) = S^T \tau + J_c^T f_c$$

其中：
- $M$：质量矩阵（Pinocchio: `data.M`）
- $h = C\dot{q} + g$：科氏力 + 重力（`data.nle`）
- $S^T$：驱动矩阵（浮基机器人前 6 行为零）
- $\tau$：关节力矩（决策变量）
- $J_c$：接触雅可比（`pin.getFrameJacobian`）
- $f_c$：接触力（决策变量）

---

## Step 3：任务与约束定义

### 常见任务

| 任务 | 变量 | 参考量 | 权重/优先级 |
|------|------|--------|------------|
| 质心位置控制 | $\ddot{c}$ | CoM 轨迹 | P1（高） |
| 躯干姿态控制 | 躯干角加速度 | 期望姿态 | P1 |
| 摆动脚追踪 | 足端加速度 | 步态规划 | P2 |
| 关节阻尼 | $\ddot{q}$ | 0 | P3（低） |

### 常见约束

| 约束 | 表达式 |
|------|--------|
| 摩擦锥 | $|f_x|, |f_y| \leq \mu f_z$，$f_z \geq 0$ |
| 关节力矩限制 | $\tau_{min} \leq \tau \leq \tau_{max}$ |
| 接触无滑动 | $J_c \ddot{q} + \dot{J}_c \dot{q} = 0$（刚性接触） |
| 关节速度限制 | 通常由上层规划保证 |

---

## Step 4：QP 构建与求解

### 决策变量

$$x = [\ddot{q}^T, \tau^T, f_c^T]^T$$

（Saab 2013 取 $u=(\tau,\ddot q,f)$；`tsid` 库只取 $[\ddot q, f]$，$\tau$ 事后由驱动行动力学解出。）

（Saab 2013 取 $u=(\tau,\ddot q,f)$；`tsid` 库只取 $[\ddot q, f]$，$\tau$ 事后由驱动行动力学解出。）

### 目标函数（加权任务）

$$\min_x \sum_i w_i \|J_i \ddot{q} + \dot{J}_i \dot{q} - \ddot{x}_{i,ref}\|^2 + w_\tau \|\tau\|^2$$

### QP 标准形式

$$\min_x \frac{1}{2} x^T H x + g^T x$$
$$\text{s.t.} \quad A_{eq} x = b_{eq}, \quad A_{ineq} x \leq b_{ineq}$$

### 求解器选型

| 求解器 | 类型 | 已核对的特点 | 推荐场景 |
|-------|------|------|---------|
| **OSQP** | QP（ADMM 一阶法） | 论文：典型比内点法快约 10 倍；默认精度中等（`eps_abs=1e-3`）；`update(q,l,u)` 可复用分解 + warm start | 稀疏 / 中大型 QP、需要参数化反复求解 |
| **qpOASES** | QP（有效集） | OSQP 论文：有效集法易 warm start，但最坏复杂度随约束数指数增长；HPIPM 论文：在 43 个小规模 QP 中约 30% 最快、其余常慢 10 倍以上 | 小规模、相邻周期有效集变化小 |
| **eiquadprog（Goldfarb–Idnani 对偶法）** | 稠密 QP | WBIC 论文（Kim 2019）用 Goldfarb–Idnani 求 WBC 的 QP，称适合小问题；`tsid` 库默认求解器同族 | 变量几十到一两百的稠密 WBC QP |
| ECOS / Gurobi | SOCP / QP·MIP | 作为内点法 / 商业基线出现在 OSQP 基准；本次未核对其在 WBC 中的表现 | 需二阶锥摩擦锥精确表达时再评估（推测） |

> 旧版表中"OSQP 50~200 μs""≥500 Hz"等具体数字没有原文依据，已删除；实际耗时取决于变量数、约束数与硬件，须自己测量。

```python
import osqp
import numpy as np

solver = osqp.OSQP()
solver.setup(P=H_sparse, q=g, A=A_sparse, l=lb, u=ub,
             warm_starting=True, verbose=False, eps_abs=1e-4)
result = solver.solve()
tau = result.x[nv:nv+nu]   # 提取关节力矩
f_c = result.x[nv+nu:]      # 提取接触力
```

---

## Step 5：力矩映射与输出

从 WBC 输出到实机执行：

```
WBC 输出: τ_wbc [关节力矩, Nm]
↓
关节摩擦补偿: τ_out = τ_wbc + τ_friction(dq)
↓
力矩限幅: τ_out = clip(τ_out, τ_min, τ_max)
↓
发送至电机驱动器 (CAN / EtherCAT)
```

**PD 辅助（可选）：**
在不确定 WBC 输出正确之前，可叠加弱 PD 控制作为安全网：
```
τ_final = τ_wbc + Kp*(q_ref - q) + Kd*(dq_ref - dq)
```
调试期 Kp/Kd 设高，收敛后减小至接近 0。

---

## Step 6：实机调试清单

### Phase 1：静止测试（上电，机器人架起来）
- [ ] 零力矩测试：输出全零，检查机器人自重姿态
- [ ] [重力补偿](../concepts/gravity-compensation.md)：输出 $\tau = g(q)$，机器人应悬空保持静止
- [ ] 各关节单独测试：阶跃指令，观察响应

### Phase 2：站立（支撑状态）
- [ ] 质心高度控制：CoM Z 轴跟踪 ±5cm 阶跃
- [ ] 躯干姿态：Roll/Pitch ±5° 响应
- [ ] 接触力可视化：确认支撑力在摩擦锥内

### Phase 3：动态行走
- [ ] 摆动脚轨迹跟踪误差 < 2cm
- [ ] 步态相位切换平滑（无冲击力）
- [ ] 带扰动测试（轻推躯干）

---

## 原文里的工程经验（已核对）

- **总要加一个姿态（postural）任务**：Del Prete 2014 证明加入最低优先级的关节姿态任务后，解唯一且与权重矩阵无关，并稳定关节空间零空间；否则带名义速度的机器人在零空间里会漂移到奇异构型或关节限位。
- **阻尼伪逆**：原文用阻尼系数 $\lambda=0.02$（最大增益 $(2\lambda)^{-1}=25$），零空间投影矩阵里把小于 $2.5\times10^{-8}$ 的奇异值置零，两者需满足 $\sigma_{min}^2(\sigma_{min}^2+\lambda^2)^{-1}<z$。
- **接触可行性**：Saab 2013 指出只加"接触点不动"约束而不加"法向力非负"，会生成不可行的动态运动；两者都要进 QP。
- **MIT WBIC（Kim 2019）实机配置**：WBIC 500 Hz、关节 PD 40 kHz；任务 PD 增益 $k_p=100\,s^{-2}$、$k_d=10\,s^{-1}$（姿态 / 位置 / 足端相同）；QP 权重反力项 1、浮动基座松弛项 0.1；所有测试共用一组参数。它输出的不只有力矩，还有关节位置 / 速度指令交给关节 PD。
- **`tsid` 库的用法**：见 [TSID](../concepts/tsid.md) 工程实践；优先级 0 为硬约束层，其后为加权任务层，默认求解器至多 2 层。

## 常见问题诊断

| 症状 | 可能原因 | 排查方向 |
|------|---------|---------|
| 机器人上电抖动 | 重力补偿计算错误 | 检查 $g(q)$ 输出大小量级；见 [重力补偿](../concepts/gravity-compensation.md) |
| QP 无解（infeasible） | 任务冲突或摩擦锥过紧 | 放宽摩擦系数，降低任务权重 |
| 步态时机器人向前倾 | CoM X 轴跟踪滞后 | 增加前馈，减小 CoM 追踪延迟 |
| 关节力矩剧烈震荡 | QP 求解数值不稳定 | 检查 H 矩阵条件数，增加正则项 |
| 仿真正常，实机漂移 | 模型参数不准（SysID） | 进行关节摩擦/惯量辨识 |

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| HQP | Hierarchical Quadratic Programming | 按优先级分层堆叠任务与约束的二次规划 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| CAD | Computer-Aided Design | 计算机辅助设计，硬件结构建模 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| Kp | Proportional Gain | PD 控制的位置误差增益，影响刚度与响应 |
| Kd | Derivative Gain | PD 控制的速度误差增益，抑制振荡 |
| SysID | System Identification | 系统辨识，估计物理/动力学参数 |

## 参考来源

- [Del Prete et al. 2014 — TSID 原论文通读归档](../../sources/papers/delprete_tsid_arxiv_1410_3863.md)
- [Saab et al. 2013 — 刚性接触下动态全身运动生成通读归档](../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md)
- [Kim et al. 2019 — WBIC + MPC（Mini Cheetah）通读归档](../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md)
- [Stellato et al. — OSQP 论文通读归档](../../sources/papers/osqp_admm_qp_arxiv_1711_08013.md)
- [Pinocchio 仓库归档（README 与头文件核对）](../../sources/repos/pinocchio.md)
- [TSID 仓库归档](../../sources/repos/tsid.md)、[OSQP 仓库归档](../../sources/repos/osqp.md)
- Escande et al. (2014), *Hierarchical QP* — 见 [HQP 归档](../../sources/papers/escande_hqp_ijrr_2014.md)
- Pinocchio 文档: <https://github.com/stack-of-tasks/pinocchio>
- OSQP 文档: <https://osqp.org/>
- [零空间控制论文簇](../../sources/papers/null_space_control.md)

---

## 关联页面

- [Whole-Body Control](../concepts/whole-body-control.md) — WBC 理论基础
- [TSID](../concepts/tsid.md) — WBC 的典型实现框架
- [HQP](../concepts/hqp.md) — 层级 QP 任务优先级框架
- [零空间控制](../concepts/null-space-control.md) — 单臂 7 轴显式投影；全身再升级 HQP
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md) — WBC 任务中的 CoM 动力学
- [Contact Estimation](../concepts/contact-estimation.md) — WBC 需要接触状态作为输入
- [MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md) — WBC 作为 MPC 下层执行器的架构
- [重力补偿](../concepts/gravity-compensation.md) — 悬空 $\tau=g(q)$ 验收

## 推荐继续阅读

- Del Prete et al., *Task Space Inverse Dynamics*：<https://arxiv.org/abs/1410.3863>
- Kim et al., *Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control*：<https://arxiv.org/abs/1909.06586>
- TSID 仓库 exercises：<https://github.com/stack-of-tasks/tsid>
