---
type: query
tags: [pinocchio, dynamics, kinematics, wbc, python, quick-start]
status: stable
summary: "Pinocchio 快速上手：最小可运行动力学示例"
updated: 2026-04-25
sources:
  - ../../sources/papers/robot_kinematics_tools.md
---

# Pinocchio 快速上手：最小可运行动力学示例

> **Query 产物**：本页由以下问题触发：「用 Pinocchio 做机器人动力学计算的最小可运行示例？」
> 综合来源：[pinocchio.md](../entities/pinocchio.md)、[wbc-implementation-guide.md](./wbc-implementation-guide.md)、[tsid.md](../concepts/tsid.md)

---

## TL;DR

Pinocchio 的核心是三步：**加载 URDF → 更新运动学/动力学 → 调用算法**。90% 的 WBC/TO 工作围绕这三步展开。

```bash
pip install pin  # 或 conda install pinocchio -c conda-forge
```

---

## 最小示例：加载 URDF + 正运动学

```python
import pinocchio as pin
import numpy as np

# 1. 加载 URDF
model = pin.buildModelFromUrdf("robot.urdf")
data = model.createData()

# 2. 随机配置（也可以用 pin.neutral(model) 得到默认姿态）
q = pin.randomConfiguration(model)  # shape: (nq,)
v = np.zeros(model.nv)              # 广义速度，shape: (nv,)

# 3. 更新正运动学（更新所有关节的 SE(3) 变换）
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# 4. 读取末端执行器位姿
frame_id = model.getFrameId("r_ankle_roll_link")  # 替换为你的 frame 名
T_world_foot = data.oMf[frame_id]  # pin.SE3 对象
print("位置:", T_world_foot.translation)
print("旋转:", T_world_foot.rotation)
```

---

## 逆运动学（雅可比 + 阻尼最小二乘）

```python
# 计算空间雅可比（世界坐标系，6D）
J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
# J.shape: (6, nv)

# 阻尼伪逆 IK（dq = J† * dx）
dx_desired = np.array([0.01, 0, 0, 0, 0, 0])  # 希望末端沿 x 方向移动 1cm
damping = 1e-6
J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
dq = J_pinv @ dx_desired
q_new = pin.integrate(model, q, dq)  # 正确处理 SE3/SO3 流形积分
```

---

## 动力学：RNEA（逆动力学）

```python
# RNEA: 给定 q, v, a → 计算所需关节力矩 τ
a = np.zeros(model.nv)  # 广义加速度
tau = pin.rnea(model, data, q, v, a)
# tau.shape: (nv,)

# 附加：计算惯性矩阵 M（正动力学基础）
M = pin.crba(model, data, q)  # Composite Rigid Body Algorithm
# M.shape: (nv, nv)
```

---

## 动力学：ABA（正动力学）

```python
# ABA: 给定 q, v, τ → 计算加速度 a
tau_input = np.zeros(model.nv)
a_out = pin.aba(model, data, q, v, tau_input)
# a_out.shape: (nv,)
```

---

## 浮动基机器人（legged robots）

腿式机器人需要浮动基，配置向量包含 SE(3) 基座状态：

```python
# 加载浮动基 URDF（在 buildModelFromUrdf 中指定 freeFlyer）
model = pin.buildModelFromUrdf("robot.urdf", pin.JointModelFreeFlyer())

# 配置向量结构：[基座位置(3) + 基座四元数(4) + 关节位置(n)] → nq = 7 + n
# 速度向量结构：[基座线速度(3) + 基座角速度(3) + 关节速度(n)] → nv = 6 + n
q = pin.neutral(model)  # 默认站立位姿

# 设置基座位置（第0-2分量）
q[:3] = [0, 0, 0.85]  # 基座高度

# 设置基座方向（四元数，第3-6分量，xyzw 顺序）
q[3:7] = [0, 0, 0, 1]  # 单位四元数（无旋转）
```

---

## 接触约束 + WBC 框架

```python
# 计算接触雅可比（用于 WBC 约束）
contact_frame = model.getFrameId("left_foot")
Jc = pin.computeFrameJacobian(model, data, q, contact_frame, 
                               pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]  # 取位置部分

# 典型 WBC QP 约束形式（参考 wbc-implementation-guide.md）：
# min  ||τ - τ_des||² + λ ||ddq||²
# s.t. M(q) ddq + h(q,v) = Sᵀ τ + Jcᵀ λ   (动力学方程)
#      Jc ddq = -dJc/dt * v                  (接触加速度为零)
#      τ_min ≤ τ ≤ τ_max
```

---

## 常用 API 速查

| 函数 | 用途 |
|------|------|
| `pin.forwardKinematics(model, data, q)` | 更新所有关节位姿 |
| `pin.updateFramePlacements(model, data)` | 更新 frame 位姿 |
| `pin.computeFrameJacobian(...)` | 计算 frame 雅可比 |
| `pin.rnea(model, data, q, v, a)` | 逆动力学（关节力矩） |
| `pin.crba(model, data, q)` | 惯性矩阵 |
| `pin.aba(model, data, q, v, tau)` | 正动力学（关节加速度） |
| `pin.computeCoriolisMatrix(model, data, q, v)` | 科里奥利矩阵 C |
| `pin.computeGeneralizedGravity(model, data, q)` | 重力项 g(q) |
| `pin.integrate(model, q, dq)` | 流形上的配置积分 |

---

## 调试提示

1. **frame 名检查**：`print([f.name for f in model.frames])` 列出所有 frame
2. **坐标系检查**：`LOCAL` = 末端局部系，`LOCAL_WORLD_ALIGNED` = 末端原点 + 世界方向（WBC 常用）
3. **nq vs nv**：浮动基下 nq ≠ nv（四元数 4 个参数但只有 3 个自由度），永远用 nv 分配速度/力向量
4. **更新顺序**：`forwardKinematics` 必须在读取 `data.oMf` 前调用

---

## 参考来源

- [sources/papers/robot_kinematics_tools.md](../../sources/papers/robot_kinematics_tools.md) — ingest 档案（Pinocchio 2019 / Crocoddyl 2020）
- [Pinocchio 官方文档](https://stack-of-tasks.github.io/pinocchio/)
- [Pinocchio 示例](https://github.com/stack-of-tasks/pinocchio/tree/master/examples)

---

## 关联页面

- [Pinocchio](../entities/pinocchio.md) — Pinocchio 框架详细介绍
- [WBC Implementation Guide](./wbc-implementation-guide.md) — 基于 Pinocchio 的完整 WBC 实现
- [TSID](../concepts/tsid.md) — TSID 框架使用 Pinocchio 作为底层引擎
- [Crocoddyl](../entities/crocoddyl.md) — 在 Pinocchio 之上的最优控制框架

---

## 一句话记忆

> Pinocchio 三步：加载 URDF → `forwardKinematics(q)` → 调算法（RNEA/CRBA/ABA/Jacobian）；浮动基要记住 nq ≠ nv，坐标系用 `LOCAL_WORLD_ALIGNED`。
