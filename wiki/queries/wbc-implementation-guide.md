---
title: WBC 工程实现指南
type: query
status: complete
created: 2026-04-14
updated: 2026-04-14
summary: 从零实现 Whole-Body Control 的工程步骤：URDF 建模 → 动力学计算 → QP 求解 → 任务配置 → 实机调试，附工具链选型和常见问题。
sources:
  - ../../sources/papers/whole_body_control.md
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
M = data.M  # shape: (nv, nv)

# 获取非线性力 h(q,v) = C(q,v)v + g(q)
h = data.nle  # shape: (nv,)

# 获取末端执行器雅可比 J_ee
J = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
```

**模型质量检查清单：**
- [ ] 质量、惯量、质心位置正确（与 CAD/实体一致）
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

### 目标函数（加权任务）

$$\min_x \sum_i w_i \|J_i \ddot{q} + \dot{J}_i \dot{q} - \ddot{x}_{i,ref}\|^2 + w_\tau \|\tau\|^2$$

### QP 标准形式

$$\min_x \frac{1}{2} x^T H x + g^T x$$
$$\text{s.t.} \quad A_{eq} x = b_{eq}, \quad A_{ineq} x \leq b_{ineq}$$

### 求解器选型

| 求解器 | 类型 | 速度 | 稳定性 | 推荐场景 |
|-------|------|------|--------|---------|
| **OSQP** | QP | 极快（50~200μs） | 高 | 实时控制（≥500Hz） |
| **qpOASES** | QP | 快 | 高 | 活跃集方法，热启动效果好 |
| **ECOS** | SOCP | 中 | 高 | 需要二阶锥约束（摩擦锥精确） |
| Gurobi | QP/MIP | 快 | 很高 | 商业，研究用免费 |

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
- [ ] 重力补偿：输出 $\tau = g(q)$，机器人应悬空保持静止
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

## 常见问题诊断

| 症状 | 可能原因 | 排查方向 |
|------|---------|---------|
| 机器人上电抖动 | 重力补偿计算错误 | 检查 $g(q)$ 输出大小量级 |
| QP 无解（infeasible） | 任务冲突或摩擦锥过紧 | 放宽摩擦系数，降低任务权重 |
| 步态时机器人向前倾 | CoM X 轴跟踪滞后 | 增加前馈，减小 CoM 追踪延迟 |
| 关节力矩剧烈震荡 | QP 求解数值不稳定 | 检查 H 矩阵条件数，增加正则项 |
| 仿真正常，实机漂移 | 模型参数不准（SysID） | 进行关节摩擦/惯量辨识 |

---

## 参考来源

- Del Prete et al., *Task Space Inverse Dynamics* (2014) — TSID 参考实现
- Escande et al., *Hierarchical QP* (2014) — HQP 框架
- Pinocchio 文档: <https://github.com/stack-of-tasks/pinocchio>
- OSQP 文档: <https://osqp.org/>

---

## 关联页面

- [Whole-Body Control](../concepts/whole-body-control.md) — WBC 理论基础
- [TSID](../concepts/tsid.md) — WBC 的典型实现框架
- [HQP](../concepts/hqp.md) — 层级 QP 任务优先级框架
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md) — WBC 任务中的 CoM 动力学
- [Contact Estimation](../concepts/contact-estimation.md) — WBC 需要接触状态作为输入
- [MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md) — WBC 作为 MPC 下层执行器的架构
