# MPC 与 WBC 集成：人形机器人 locomotion 的典型控制架构

> **Query op 产物** · 2026-04-13
> 问题：MPC 和 WBC 在人形机器人 locomotion 里是怎么配合工作的？
> 综合来源：MPC page + WBC page + locomotion page + optimal-control page

## 一句话总结

**MPC 负责"大尺度规划"（质心往哪走、落脚点放哪），WBC 负责"全身执行"（怎么协调关节力矩来跟踪 MPC 发出的指令）**——两者分层配合，组成当前人形机器人 locomotion 最主流的控制架构。

## 问题背景

人形 locomotion 面临两个层次的控制问题：

| 层次 | 核心问题 | 时间尺度 | 典型方法 |
|------|---------|---------|---------|
| **高层规划** | 接下来几步怎么走、落脚点在哪 | 100ms ~ 1s | MPC |
| **底层执行** | 每个关节发多少力矩才能跟踪规划 | 1ms | WBC (QP) |

如果只用 MPC 直接控制关节——计算量太大（30+ 自由度），实时不可行。
如果只用 WBC 不用 MPC——没有前瞻规划，只能被动响应，无法处理复杂地形和扰动。

## 分层控制架构

```
感知层（IMU、编码器、视觉等）
    ↓
┌─────────────────────────────────────┐
│  高层：MPC（模型预测控制）              │
│  - 基于 centroidal dynamics / LIP 模型 │
│  - 预测未来 N 步（100ms~1s）          │
│  - 输出：下一步质心位置 / 落脚点计划    │
└─────────────────────────────────────┘
    ↓  task-space 指令
┌─────────────────────────────────────┐
│  低层：WBC（全身控制 / QP 优化）        │
│  - 接收 MPC 发来的任务空间目标          │
│  - 求解关节力矩分配 QP                  │
│  - 输出：各关节力矩指令                │
└─────────────────────────────────────┘
    ↓
电机驱动 → 机器人执行
```

## 典型实例：Walking Control

### Phase 1：MPC 生成支撑多边形和质心计划

MPC 以 **Centroidal Dynamics** 或 **LIP** 为预测模型：

```
输入：
  - 当前状态（位置、速度、姿态）
  - 参考步态（步长、步频）
  - 约束（摩擦锥、关节限位）

求解：
  - 未来 N 步的质心轨迹
  - 未来 N 步的落脚点序列（Contact Force Plan）

输出：
  - 质心参考轨迹 (CoM position / velocity)
  - 落脚点位置 + 期望接触力
```

### Phase 2：WBC 跟踪 MPC 指令

WBC 接收 MPC 的输出，将其转化为关节力矩：

```
任务层（Task Space）：
  - 跟踪质心轨迹  → 质心任务
  - 跟踪落脚点    → 足端位置任务
  - 跟踪姿态      → 姿态任务

优先级调度：
  1. 平衡任务（最高优先级）
  2. 落脚点跟踪
  3. 姿态跟踪
  4. 关节限位（硬约束）

QP 优化层：
  - 最小化任务跟踪误差
  - 满足关节力矩/速度限位
  - 满足摩擦锥约束

输出：
  - 各关节力矩指令（1kHz）
```

### 关键设计点

**1. MPC 和 WBC 的模型一致性**
- 如果 MPC 用 Centroidal Dynamics，WBC 需要把 centroidal 目标映射到关节空间
- 常用方法：先在质心空间跟踪，再在关节空间用 WBC 补偿
- **常见坑**：两层模型不一致会导致跟踪不稳定

**2. 时域分离**
- MPC 以 10-50Hz 运行（每步计算量太大，无法 kHz 级别）
- WBC 以 1kHz+ 运行（纯 QP，计算快）
- MPC 的输出作为 WBC 的"参考轨迹"输入

**3. 接触序列决定控制模式**
- 双脚支撑期：两个接触点，MPC 分配地面反力
- 单脚支撑期：一个接触点，WBC 处理力矩平衡
- **扰动恢复**：MPC 重新规划落脚点，WBC 快速响应

**4. 真实部署例子**

| 机器人 | MPC 层 | WBC 层 |
|--------|--------|--------|
| MIT Mini Cheetah | Convex MPC (100Hz) | QP WBC (1kHz) |
| Unitree H1 | NMPC | WBC (Hierarchical QP) |
| ANYmal | Hierarchical NMPC | Whole-body QP |
| Atlas (Boston Dynamics) | MPC (Model Predictive Planning) | WBC |

## 为什么不用 RL 替代 MPC？

| 维度 | MPC + WBC | RL 策略 |
|------|-----------|---------|
| 实时性 | ✅ 确定性，毫秒级响应 | ⚠️ 需要推理速度足够快 |
| 扰动恢复 | ✅ MPC 重新规划，适应性很强 | ⚠️ 取决于训练分布 |
| 样本效率 | ✅ 无需训练 | ❌ 需要大量训练 |
| 约束处理 | ✅ 硬约束自然表达 | ⚠️ 需要 reward shaping |
| 泛化到新地形 | ⚠️ 需要重新规划 | ✅ 有一定泛化能力 |

**主流趋势**：MPC + WBC 作为 baseline，RL 作为提升层（学更好的策略 prior 或补偿误差）。

## 关联页面

- [Model Predictive Control (MPC)](../methods/model-predictive-control.md) — MPC 层
- [Whole-Body Control (WBC)](./whole-body-control.md) — WBC 层
- [Centroidal Dynamics](./centroidal-dynamics.md) — MPC 常用的人形简化动力学模型
- [Locomotion](../tasks/locomotion.md) — locomotion 任务层
- [Optimal Control (OCP)](./optimal-control.md) — 理论层，MPC 是 OCP 的在线求解
