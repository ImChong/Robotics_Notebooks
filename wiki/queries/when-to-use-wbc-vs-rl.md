---
type: query
tags: [wbc, rl, locomotion, decision, architecture]
related:
  - ../comparisons/wbc-vs-rl.md
  - ../concepts/whole-body-control.md
  - ../methods/policy-optimization.md
  - ../tasks/locomotion.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/policy_optimization.md
  - ../../sources/papers/whole_body_control.md
---

# 决策树：何时使用 WBC vs RL？

> Query 类型：决策指南
> 生成日期：2026-04-15
> 问题：面对具体机器人控制任务，应该选择全身控制（WBC）、强化学习（RL），还是两者结合？

---

## 快速决策树

```
任务是否需要精确接触力控制？
├── 是 → 优先考虑 WBC
│        任务是否高度重复且有明确数学模型？
│        ├── 是 → 纯 WBC（如工业抓取、已知地形行走）
│        └── 否 → WBC + RL（如 loco-manipulation、不规则地形）
└── 否 → 任务是否在仿真中容易设计 reward？
          ├── 是 → 优先 RL（如 locomotion、障碍跑、运动技能）
          └── 否 → 需要演示数据？
                    ├── 是 → IL（BC / Diffusion Policy）
                    └── 否 → 重新分解任务
```

---

## 详细对比

### 选 WBC 的信号

| 信号 | 说明 |
|------|------|
| 有精确动力学模型 | URDF + 精确惯量参数，接触约束可数学建模 |
| 实时力矩/接触力反馈 | SEA 或力矩传感器可用；接触面已知 |
| 任务有明确约束层级 | 如"保持平衡 > 完成末端任务 > 关节极限"的优先级明确 |
| 需要精确末端轨迹 | 机械臂操作、精密装配 |
| 计算资源充足 | WBC 的 QP 求解每步约 1–5ms；需要实时算力 |

**典型场景**：Atlas 行走（官方控制器）、ANYmal 地形跟随、工业机械臂装配

---

### 选 RL 的信号

| 信号 | 说明 |
|------|------|
| 动力学模型不精确 | 接触摩擦、软体、未知地形难以建模 |
| 任务 reward 可设计 | 前进速度、能量、存活时间等标量 reward 自然 |
| 需要探索未知动作空间 | 人形机器人跑跳、翻滚等非结构化运动 |
| 需要应对大量扰动 | 推力、不规则地形；RL 通过 domain randomization 自然鲁棒化 |
| 快速原型验证 | 仿真环境下数小时可训出策略，无需建模 |

**典型场景**：Unitree H1 locomotion、ANYmal 野外行走（Learning to Walk in Minutes）、人形跑跳

---

### 选 WBC + RL 结合的信号

| 信号 | 说明 |
|------|------|
| 高层目标 + 低层力控 | RL 输出期望速度/接触时序，WBC 执行精确关节指令 |
| 需要接触多样性 + 鲁棒性 | loco-manipulation 任务：RL 处理探索，WBC 处理力控 |
| 从仿真迁移到真机 | RL 提供鲁棒性，WBC 减小 sim2real gap（力矩层面精确） |
| 任务层级明确分离 | 如 MPC（中层轨迹规划）+ WBC（低层力控）+ RL（高层策略） |

**典型场景**：Unitree H1 loco-manipulation、MIT Cheetah 3 跑跳（MPC + WBC 分层）

---

## 实现复杂度 vs 性能权衡

| 方案 | 实现难度 | 峰值性能 | 鲁棒性 | sim2real |
|------|---------|---------|--------|---------|
| 纯 WBC | ★★★★ | ★★★★ | ★★★ | ★★★ |
| 纯 RL | ★★ | ★★★★★ | ★★★★★ | ★★★ |
| WBC + RL 分层 | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ |
| MPC + RL | ★★★★ | ★★★★★ | ★★★★ | ★★★★ |

---

## 常见误区

**误区 1**：*"WBC 太慢，不适合快速运动"*
→ 错误。现代 WBC（TSID/HQP）可以 1kHz 运行；瓶颈通常是感知延迟，不是 QP 求解。

**误区 2**：*"RL 可以完全替代 WBC"*
→ 不准确。RL 在接触力精确控制上仍弱于 WBC；高精度操作任务中两者互补。

**误区 3**：*"有了 RL 不需要建模"*
→ 危险。无精确模型的 RL 在真机上可能学到依赖仿真 artifact 的策略；sim2real gap 可能来自未建模的执行器动态。

---

## 关联页面
- [WBC vs RL 对比](../comparisons/wbc-vs-rl.md)
- [全身运动控制](../concepts/whole-body-control.md)
- [策略优化方法](../methods/policy-optimization.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)

## 参考来源
- [policy_optimization.md](../../sources/papers/policy_optimization.md)
- [whole_body_control.md](../../sources/papers/whole_body_control.md)
