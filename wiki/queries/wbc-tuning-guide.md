---
type: query
tags: [wbc, control, optimization, tuning, humanoid, qp]
status: complete
updated: 2026-04-20
related:
  - ../concepts/whole-body-control.md
  - ../concepts/tsid.md
  - ./wbc-implementation-guide.md
  - ../formalizations/friction-cone.md
sources:
  - ../../sources/papers/whole_body_control.md
summary: "WBC 调参指南：通过合理的任务权重分配、约束松弛、正则化技巧与求解器热启动配置，实现高性能全身控制的实践经验。"
---

# WBC 调参指南：从公式到实机表现

> **Query 产物**：本页由以下问题触发：「WBC QP 求解器怎么调参？权重矩阵、约束松弛、热启动有哪些技巧？」
> 综合来源：[Whole-Body Control](../concepts/whole-body-control.md)、[TSID](../concepts/tsid.md)、[WBC Implementation Guide](./wbc-implementation-guide.md)

---

## 1. 权重矩阵的设计 (The "Art of Weights")

WBC 通常是一个多目标优化问题 $\min \|Ax - b\|_W^2$。权重 $W$ 决定了任务的优先级。

### 调参顺序
1. **主优先级任务 (High Priority)**：
   - **躯干姿态 (Floating Base Orientation)**：这是平衡的核心。如果不稳，机器人会迅速侧翻。
   - **支撑足加速度 (Support Foot Consistency)**：必须设为极高权重或作为硬等式约束，否则脚会滑移。
2. **次优先级任务**：
   - **质心轨迹 (CoM Tracking)**：决定了运动的总体走向。
   - **摆动足轨迹 (Swing Foot Tracking)**：确保迈步准确。
3. **低优先级任务 (Regularization)**：
   - **关节姿态正则化 (Joint Posture)**：给出一个倾向的“基础姿势”（如微蹲），防止奇异点。

### 权重比例技巧
- 不要使用绝对值，使用**相对比例**。
- 典型的比例：`Orientation : Support_Foot : CoM : Swing_Foot : Joint_Reg = 1000 : 1000 : 100 : 100 : 1`。

---

## 2. 约束松弛与惩罚 (Relaxation)

硬约束（如 $Ax=b$）在实际物理碰撞中容易导致 QP 无解（Infeasible）。

### 处理技巧
- **等式约束转化为惩罚项**：将 $Ax=b$ 变为 $\min \lambda \|Ax - b\|^2$。
- **引入松弛变量 (Slack Variables)**：对于摩擦锥约束 $\sqrt{f_x^2 + f_y^2} \leq \mu f_z$，如果一定要违反，引入 $\xi \geq 0$：$\sqrt{f_x^2 + f_y^2} \leq \mu f_z + \xi$。
- **权重策略**：对松弛变量 $\xi$ 施加极大的线性惩罚权重。

---

## 3. 正则化 (Regularization)

为了提高数值稳定性，尤其是当雅可比矩阵 $J$ 接近奇异时。

- **力矩正则化**：惩罚 $\|\tau\|^2$。防止 QP 求解器为了微小的跟踪精度提升而输出巨大的瞬时力矩。
- **阻尼项**：在 $A^T A$ 中加入 $\epsilon I$。这能保证 Hessian 矩阵始终正定。

---

## 4. 求解器热启动 (Hot Start)

对于 1kHz 的控制环，每一微秒都很珍贵。

- **重复利用上一时刻解**：将 $t-1$ 时刻的解 $x^*$ 作为 $t$ 时刻的初始猜测。
- **求解器策略**：
  - **OSQP**：开启 `warm_starting=True`，能显著减少 ADMM 迭代次数。
  - **qpOASES**：非常适合热启动，因为它基于 Active Set 策略，相邻帧的有效约束集往往变化很小。

---

## 5. 实机调试排障

| 症状 | 可能原因 | 解决对策 |
|------|----------|----------|
| **关节高频震荡** | PD 增益太高或力矩正则化太弱 | 增加 $\|\tau\|^2$ 权重，减小任务空间 $K_p$ |
| **支撑脚打滑** | 摩擦系数 $\mu$ 设得太高或摩擦锥约束被违反 | 调低 $\mu$（保守估计），加强摩擦锥惩罚 |
| **Base 姿态下沉** | CoM 权重太低，无法克服重力 | 增加 CoM 任务权重，检查动力学 $g(q)$ 补偿是否正确 |
| **QP 频繁无解** | 任务冲突严重或硬约束太多 | 检查是否有相互排斥的等式约束，增加松弛变量 |

---

## 关联页面
- [Whole-Body Control 概念](../concepts/whole-body-control.md)
- [TSID 方法](../concepts/tsid.md)
- [WBC 实现指南](./wbc-implementation-guide.md)
- [Friction Cone 形式化](../formalizations/friction-cone.md)

## 参考来源
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md)
- Kim et al., *Highly Dynamic Quadrupedal Locomotion via Whole-Body Impulse Control* (2019).
