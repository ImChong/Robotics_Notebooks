# Dynamic Humanoid Arm-Swing Motion via Trajectory Optimization Leveraging Speed–Torque Mode Switching of a Variable Chain Motor（IROS 2026）

- **标题：** Dynamic Humanoid Arm-Swing Motion via Trajectory Optimization Leveraging Speed–Torque Mode Switching of a Variable Chain Motor
- **类型：** paper（会议）
- **会议：** IEEE/RSJ IROS 2026
- **项目页：** <https://hit4752.github.io/projects/202609_iros26-arm-swing/>
- **机构：** 东京大学 — Hiromi Tada, Takuma Hiraoka, Jin Hirai, Kunio Kojima, Kei Okada
- **入库日期：** 2026-09-25
- **代码：** **无**（项目页 2026-09-25 核查，见 [hit4752_iros26_vc_motor_arm_swing.md](../sites/hit4752_iros26_vc_motor_arm_swing.md)）
- **一句话说明：** 针对 VC 电机 **离散切换 speed–torque 约束** 带来的规划难题，在轨迹优化中用 **全模式可行域并集** 约束 \(v,\tau\)，执行时再按轨迹选模式；在 JAXON 肘部实现 **9.01 m/s** 末端峰值，优于任一固定模式。

## 摘要级要点

- **问题：** 高竞技人形动作需要 **高关节速度 + 高关节扭矩**；变传动可扩大包络但对人形 **重量尺寸** 不友好；**输出特性切换** 导致 admissible speed–torque 约束 **不连续**，动态运动规划研究不足。
- **方法：** 轨迹优化中把关节 \(v,\tau\) 约束在 **多模式 speed–torque 可行范围之并集**；优化阶段 **不显式引入离散模式变量**；真机执行每步由优化轨迹的 speed–torque 剖面 **决定模式**。
- **约束：** \(|v|\le v_{\max,C}\) 且 \(|\tau|\le \min_i(\min(\tau_{\max,i}, -\alpha(|v|-v_{\max,i})))\) 类形式（三模式 A/B/C）；用 **LSE log-sum-exp** 平滑 max/min。
- **任务：** JAXON [2] 左肘 VC 电机；优化 **最大向前末端速度** 的甩臂；对比固定高扭矩 / 中间 / 高速与 **提出切换** 四条件。
- **结果（真机峰值）：** 末端 **9.01 m/s**、肘 **−10.23 rad/s**；固定最快高速模式 (c) 为 8.24 m/s / −9.19 rad/s。优化轨迹上切换相对 (c) 末端 **+7.7%**、肘 **+10.2%**。
- **模式序列（切换条件 d）：** 优化轨迹沿 (a)→(b)→(c)→(b)→(a)：**高扭矩加速 → 切高速模式**；电压日志显示切高速模式后所需电压仍可控。

## 核心论文摘录（MVP）

### 1) 并集约束轨迹优化

- **链接：** 项目页 § Trajectory Optimization over All Feasible Modes
- **摘录要点：** 橙色边界为组合可行域；并集约束让优化同时「看到」高扭矩与高速度角，无需混合整数模式变量。
- **对 wiki 的映射：**
  - [IROS 2026 VC 动态甩臂](../../wiki/entities/paper-iros26-vc-motor-dynamic-arm-swing.md) — 流程 Mermaid 与公式。
  - [轨迹优化（Trajectory Optimization）](../../wiki/methods/trajectory-optimization.md) — 作动器约束类 OCP。

### 2) 真机模式切换与电压

- **链接：** 项目页 § Faster Humanoid Arm Swings
- **摘录要点：** 切换使高速段所需电机电压不随速度单调爆炸，支撑动态甩臂。
- **对 wiki 的映射：**
  - [可变链电机](../../wiki/entities/variable-chain-motor.md) — 电气模式与部署读法。

### 3) 与固定特性对照

- **摘录要点：** 固定高扭矩 (a) 末端仅 6.20 m/s；仅选中间或高速模式无法在全程同时满足加速与峰值速度需求。
- **对 wiki 的映射：**
  - [queries/actuator-drive-chain-selection-loop](../../wiki/queries/actuator-drive-chain-selection-loop.md) — 选型：单点 speed–torque 折中 vs 可切换包络。

## 参考文献（项目页）

1. H. Tada et al., IROS 2025 — VC motor 开发（见 [iros25_variable_chain_motor_tada.md](./iros25_variable_chain_motor_tada.md)）
2. K. Kojima et al., Humanoids 2015 — JAXON：<https://doi.org/10.1109/HUMANOIDS.2015.7363459>
