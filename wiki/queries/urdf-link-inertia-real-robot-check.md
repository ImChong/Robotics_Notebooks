---
type: query
tags: [urdf, inertia, sysid, sim2real, pinocchio, mujoco, cad, dynamics]
status: complete
updated: 2026-09-15
summary: "URDF 连杆惯量对照真机的分层抽检：约定与书桌物理一致性 → 称重/质心 → 静力学 g(q) → 动力学回归；CAD 数字不是力矩真值，10 参数也不能全部独立辨识。"
related:
  - ../concepts/urdf-robot-description.md
  - ../concepts/robot-link-and-rotor-inertia.md
  - ../concepts/system-identification.md
  - ../concepts/gravity-compensation.md
  - ../concepts/humanoid-closed-loop-inertia-calibration.md
  - ../methods/joint-actuator-parameter-identification.md
  - ../methods/sim2real-joint-sysid-experiment-design.md
  - ../entities/pinocchio.md
  - ../entities/flobaroid.md
  - ../entities/mujoco.md
  - ./simulation-physics-fidelity.md
  - ./pinocchio-quick-start.md
sources:
  - ../../sources/papers/urdf_link_inertia_real_robot_check.md
  - ../../sources/papers/robot_link_rotor_inertia_primary_refs.md
  - ../../sources/papers/system_identification.md
  - ../../sources/repos/pinocchio.md
  - ../../sources/repos/flobaroid.md
---

> **Query 产物**：本页由以下问题触发：「URDF 中的机器人连杆惯量如何与真机对比检查？」
> 综合来源：[URDF 描述](../concepts/urdf-robot-description.md)、[连杆惯量与转子惯量](../concepts/robot-link-and-rotor-inertia.md)、[System Identification](../concepts/system-identification.md)、[重力补偿](../concepts/gravity-compensation.md)、[人形整机闭环惯量标定](../concepts/humanoid-closed-loop-inertia-calibration.md)、[Pinocchio](../entities/pinocchio.md)

# URDF 连杆惯量对照真机检查

把 URDF `<inertial>` 当成「仿真里能跑」不够；要分层核对 **约定是否写对、数字是否像刚体、整机质量/质心是否与台秤一致、静止力矩是否等于 $g(q)$、运动段预测力矩是否贴真机**。CAD 导出是初值，不是验收。

## 一句话定义

> **先查 URDF 写的是不是刚体 10 参数，再用秤、静力学和动力学残差对照真机；对不上时改连杆惯量，不要把转子 `armature` 塞进质量。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| URDF | Unified Robot Description Format | 本页检查对象：每个 `link` 的 `<inertial>` |
| CoM | Center of Mass | 质心；URDF `<origin xyz>` 相对 link 系 |
| RNEA | Recursive Newton–Euler Algorithm | $g(q)=\mathrm{RNEA}(q,0,0)$，静力学对照入口 |
| SysID | System Identification | 用运动数据估 10 参数或其可辨识组合 |
| LMI | Linear Matrix Inequality | 把「像刚体」写成凸约束，避免辨识出非物理张量 |
| CAD | Computer-Aided Design | 给出密度积分初值；力矩预测不必优于实测辨识 |
| MJCF | MuJoCo XML Format | 可用等效惯量盒目视检查 URDF 导入结果 |

## TL;DR：五层对照，越往下越贵

| 层 | 问什么 | 真机量 | 通过标准 | 失败时改哪 |
|----|--------|--------|----------|------------|
| 0 约定 | 坐标系、惯量积符号、单位 | 无（书桌） | 张量在质心系；CAD 非对角已按 URDF 负号约定 | XML / 导出脚本 |
| 1 物理一致性 | 这 10 个数能否由一块刚体生成 | 无 | $m>0$，主惯量 $>0$ 且满足三角不等式；等效惯量盒不飞出外壳 | 填错的 $I$、单位阵占位 |
| 2 称重 / 质心 | 整机（或可拆连杆）质量与 CoM | 台秤；可选悬挂 / 测力板 | $\sum m$ 与实称偏差通常 $< 3\%$；CoM 落在几何内 | 漏装电池/线缆/外壳，或 CAD 密度 |
| 3 静力学 | 多组静止构型的关节力矩 | 电流/力矩计（悬空、低速） | $\tau_{\mathrm{meas}} \approx g(q)+$ 库仑；改 $I_a$ 不应大改 $g(q)$ | 质量与 CoM；不要改 armature |
| 4 动力学 | 激励轨迹上的力矩预测 | 力矩 + $q,\dot q,\ddot q$ | 预测残差优于 CAD 基线；只写回 **可辨识组合** | 最小参数集 / SDP；转子另记账 |

**总原则：** 层 0–2 不过，不要做层 4。层 3 过了，全身 WBC 才有地板；层 4 用来收动态误差，且 **不能** 唯一还原每个 link 的 10 个数。

```mermaid
flowchart TD
  start[拿到厂商或 CAD 导出的 URDF]
  start --> c0{0 约定: 质心系 / 惯量积符号 / kg·m²?}
  c0 -->|否| fix0[对齐主轴或取反 CAD 非对角元]
  c0 -->|是| c1{1 书桌: m>0, I_C 正定+三角不等式<br/>惯量盒不超出几何?}
  fix0 --> c1
  c1 -->|否| fix1[丢掉单位阵占位 / 从 mesh 密度重算]
  c1 -->|是| c2{2 台秤: Σm 与整机称重?}
  fix1 --> c2
  c2 -->|差>数个百分点| fix2[补漏件: 电池 线缆 外壳 末端工具]
  c2 -->|对齐| c3{3 悬空静止: τ ≈ g(q)?}
  fix2 --> c3
  c3 -->|系统偏置| fix3[改质量/CoM; 查重力方向与电流-力矩标定]
  c3 -->|过关| c4{4 有运动+力矩?}
  fix3 --> c4
  c4 -->|固定基 / 有关节力矩| dyn[Atkeson 回归 + Traversaro/Wensing 物理一致约束]
  c4 -->|浮动基 / 无力矩计| fb[Ayusawa 基座方程 或 PRIME 接触联合估计]
  c4 -->|高减速比振荡仍在| arm[转子惯量进 armature 不是 URDF 质量]
  dyn --> done[写回 URDF 可辨识项; 绑定机身序列号]
  fb --> done
  arm --> done
```

## 为什么重要

仿真物理保真度的第 ① 层是几何/惯量，误差会被动力学、接触和执行器逐级放大，见 [仿真物理保真度链路](./simulation-physics-fidelity.md)。常见事故不是「没填惯量」，而是：

- CAD 与 URDF **惯量积符号相反** → 侧向/扭转耦合方向反了，$g(q)$ 在某些构型突然偏。
- 公开模型填了 **单位阵或过大主惯量**（MuJoCo 官方点名）→ 等效刚体比外壳还大，落地冲击和摆动周期都假。
- 把电机转子折算惯量 **写进 link 质量** → 重力项被污染，见 [连杆 vs 转子](../concepts/robot-link-and-rotor-inertia.md)。
- 一次 OLS 估完全部 10 参数 → 得到不可实现或不可辨识的数，控制器里 $M(q)$ 仍是错的。

Atkeson 等人在直接驱动臂上已经给出工程结论：**用运动数据估出的参数，力矩预测可以优于 CAD**。对照真机的目标是「预测 $\tau$ / 平衡重力」，不是「XML 里每个数都有唯一真值」。

## 核心原理：URDF 写的到底是哪 10 个数

每个 `link` 的 `<inertial>` 规定刚体相对 **质心系 C** 的空间惯量（ROS Wiki [urdf/XML/link](http://wiki.ros.org/urdf/XML/link)）：

| 字段 | 物理量 | 真机对应 |
|------|--------|----------|
| `<mass>` | $m$ | 该连杆（含灌胶、螺丝、线束归属）的质量 |
| `<origin xyz>` | 质心 $c$ 在 link 系 | 悬挂/平衡/测力板得到的 CoM |
| `<origin rpy>` | $\hat{C}$ 相对 $\hat{L}$ | 通常取主轴使惯量积为 0 |
| `<inertia>` | $I_C$（6 元） | 绕质心的惯性张量，**不是**零件原点张量 |

标准 10 参数（Atkeson 1986 / Wensing 2018）为

$$
\pi = \bigl[m,\; mc_x,\; mc_y,\; mc_z,\; I_{xx},\; I_{xy},\; I_{xz},\; I_{yy},\; I_{yz},\; I_{zz}\bigr]^\top
$$

动力学对 $\pi$ 线性：$\tau = Y(q,\dot q,\ddot q)\,\pi$。因此「对照真机」在层 4 比较的是 **$Y\pi$ 与实测力矩**，不是逐分量相等。

Gautier & Khalil (1990)：**并非 10 个数都能独立辨识**。基座附近、轴向与重力平行、或传感器不足时，只能恢复线性组合。写回 URDF 时保留 CAD 先验去填不可观分量，只覆盖可观组合。

### 书桌必须过的物理一致性

仅 $m>0$、$I_C\succ 0$ 是 **半一致性**（保证 $M(q)\succ 0$）。Traversaro et al. (IROS 2016) 给出 **完全物理一致性**：主惯量 $J_1,J_2,J_3$ 还须满足三角不等式

$$
J_1+J_2 \ge J_3,\quad J_2+J_3 \ge J_1,\quad J_3+J_1 \ge J_2
$$

否则不存在密度 $\rho\ge 0$ 能生成这组参数。Wensing et al. (RA-L 2018) 把该条件写成 LMI，并可加「质量落在 CAD 包围盒内」。

对已对角化的 $I_C$，三角不等式即 $I_{xx}+I_{yy}\ge I_{zz}$ 等；一般对称矩阵则对 **特征值** 做同样检查。

## 工程实践：分层对照清单

### 第 0 层 — 约定与解析（不上真机）

1. 确认惯性表达在 **质心**，不是零件原点。MATLAB `rigidBodyTree` 把惯量存在关节原点，数值会与 URDF 不同，差一个平行轴定理，动力学应仍等价。
2. SolidWorks / 部分 CAD：非对角元与 URDF **差一个负号**。官方建议把 C 系对齐主轴，令 `ixy=ixz=iyz=0`。
3. Pinocchio `convertFromUrdf`：`Inertia(m, com, R I_C Rᵀ)`。缺 `<inertial>` → **零惯量**，仿真发飘。
4. 单位：kg 与 kg·m²。毫米 CAD 未换算时 $I$ 会差 $10^6$。

### 第 1 层 — 书桌尺度与可视化

- Gazebo：`View → Wireframe` + `Center of Mass`；质量必须 $>0$，主惯量为 0 会无限加速（[OSRF URDF in Gazebo](https://github.com/osrf/gazebo_tutorials/blob/master/ros_urdf/tutorial.md)）。
- MuJoCo：打开 **equivalent inertia box**。盒子远大于外壳 = 公开 URDF 的典型病（官方 `inertiafromgeom` 文档）。可用 `inertiafromgeom="true"` 从 geom 覆盖坏数字，但那是均匀密度近似，**不能替代称重**。
- ROS 教程：不要用单位阵当占位。

可辨识的启发式：均匀密度连杆的主惯量量级约 $m r^2$。大腿质量 2 kg、特征尺寸 0.3 m，则 $I$ 应在 $10^{-2}$–$10^{-1}$ kg·m²，而不是 1 或 100。

### 第 2 层 — 台秤与整机质心

这是最便宜的真机实验。

| 做法 | 测什么 | 和 URDF 比什么 |
|------|--------|----------------|
| 整机上台秤 | 总质量 | `pin.computeTotalMass(model)` |
| 可拆连杆单独称 | 单 link $m$ | 对应 `<mass>`（注意螺丝/线束归属） |
| 水平刀口 / 悬挂铅垂 | 某构型整机 CoM 投影 | `pin.centerOfMass(model, data, q)` |
| 六维力板静立 | 压力中心 | 浮动基 $g(q)$ 与接触力平衡 |

漏项几乎总是：**电池、外壳、线缆、末端工具、灌胶**。这些会改分布式质量，单关节空转台架补不回来，见 [整机闭环惯量标定](../concepts/humanoid-closed-loop-inertia-calibration.md)。

### 第 3 层 — 静力学：$g(q)$ 对照

固定基、悬空、多组静止姿态：

$$
\tau_{\mathrm{meas}}(q) \;\stackrel{?}{\approx}\; g(q) + \tau_{\mathrm{coulomb}}\,\mathrm{sign}(0^\pm)
$$

Pinocchio：`pin.computeGeneralizedGravity(model, data, q)`。验收口径与 [重力补偿](../concepts/gravity-compensation.md) 相同：悬空输出 $\tau=g(q)$ 应能停住。

**读残差：**

- 所有构型同一方向偏 → 某 link 质量或 CoM，或电流–力矩系数 $k_t$。
- 只在大展臂时偏 → 远端质量/CoM。
- 改仿真 `armature` 残差几乎不变，但阶跃变快慢 → 你在看转子惯量，不是连杆惯量。

站立接触会把惯性「吞」进地面力矩，不要在四脚着地时做这一层，见 [关节动力学辨识实验设计](../methods/sim2real-joint-sysid-experiment-design.md)。

### 第 4 层 — 动力学回归（才叫 SysID）

有关节力矩（或可靠电流）时：

1. 设计持续激励（Gautier & Khalil 1992；FloBaRoID 用 Fourier 级数）。
2. 组回归 $\min_\pi \|Y\pi-\tau\|^2$，加 Traversaro/Wensing 物理一致约束（FloBaRoID 的 SDP 路径）。
3. 只把 **可辨识组合** 写回 URDF；弱可观项拉回 CAD（FloBaRoID 显式支持）。
4. 转子/摩擦 **不要** 放进这 10 参数，见 [关节执行器参数辨识](../methods/joint-actuator-parameter-identification.md)。

浮动基、无力矩计：用 **基座 wrench / 足底力 + 运动学**（Ayusawa 2014；[FloBaRoID](../entities/flobaroid.md) 两步法；量产接触闭环见 [PRIME](../entities/prime-system-id.md)）。

### Pinocchio 书桌审计脚本

把厂商 URDF 换成你的路径即可跑层 0–1，并打印总质量供层 2 对照。

```python
import numpy as np
import pinocchio as pin

model = pin.buildModelFromUrdf("robot.urdf")
data = model.createData()

print("total mass [kg] =", pin.computeTotalMass(model))
q0 = pin.neutral(model)
print("CoM @ neutral [m] =", pin.centerOfMass(model, data, q0).T)

for i, name in enumerate(model.names):
    if i == 0:
        continue
    Y = model.inertias[i]
    m = Y.mass
    Ic = Y.inertia  # 3x3 about CoM, link orientation
    w = np.linalg.eigvalsh(Ic)
    tri = (w[0] + w[1] >= w[2] - 1e-12
           and w[1] + w[2] >= w[0] - 1e-12
           and w[2] + w[0] >= w[1] - 1e-12)
    ok = m > 0 and np.all(w > 0) and tri
    print(f"{name:24s} m={m:8.4f}  eig(I)={w}  fully-consistent={ok}")
```

层 3 把多组真机静止 $(q,\tau)$ 与 `pin.computeGeneralizedGravity` 做差；层 4 用 `pin.computeJointTorqueRegressor(model, data, q, v, a)` 组 $Y_{\mathrm{rb}}$（仍不含 armature）。

## 局限与风险

- **CAD 赢视觉、辨识赢力矩。** Atkeson 1986：辨识模型对 $\tau$ 的预测可以优于 CAD。不要因为「STEP 很精」就跳过层 3–4。
- **可辨识性。** 最小参数集以外的分量对力矩无贡献；硬估会把噪声写进 URDF。
- **半一致性不够。** 只保证 $I_C\succ 0$ 仍可能违反三角不等式，密度不可实现。
- **接触污染。** 站立/撑地时地面力矩与惯性纠缠；悬空或基座方程优先。
- **执行器不等于连杆。** 高减速比下 $J_r G^2$ 常大于连杆投影惯量；写进 `<mass>` 会毁掉 $g(q)$。
- **单位阵与 `inertiafromgeom`。** 后者用均匀密度几何近似，适合扔掉明显错误的公开模型，不适合当出厂真值。
- **公众号/产线叙事。** 整机闭环体检见 [人形闭环惯量标定](../concepts/humanoid-closed-loop-inertia-calibration.md)；本页不把 ISO 安全标准解读成惯量强制条款。

## 关联页面

- [URDF 描述](../concepts/urdf-robot-description.md) — 字段与工作流；本页是「惯量如何验收」
- [连杆惯量与转子惯量](../concepts/robot-link-and-rotor-inertia.md) — `<inertial>` vs `armature`
- [System Identification](../concepts/system-identification.md) — 刚体 / 执行器 / 摩擦分层
- [重力补偿](../concepts/gravity-compensation.md) — 层 3 的控制侧读法
- [人形整机闭环惯量标定](../concepts/humanoid-closed-loop-inertia-calibration.md) — 量产四张单子；本页是 URDF 抽检前半段
- [关节执行器参数辨识](../methods/joint-actuator-parameter-identification.md) / [实验设计](../methods/sim2real-joint-sysid-experiment-design.md) — $I_a$ 与摩擦，不是 link 10 参数
- [Pinocchio](../entities/pinocchio.md) / [Pinocchio 快速上手](./pinocchio-quick-start.md) — `computeTotalMass` / `computeGeneralizedGravity` / 回归矩阵
- [FloBaRoID](../entities/flobaroid.md) — 激励 → 物理一致辨识 → 写回 URDF
- [仿真物理保真度链路](./simulation-physics-fidelity.md) — 本页落实第 ① 层「真机称重 + CAD 复核 + SysID」

## 参考来源

- [URDF 连杆惯量与真机对照检查（一手资料索引）](../../sources/papers/urdf_link_inertia_real_robot_check.md) — 本页编译依据
- [连杆惯量与转子惯量一手索引](../../sources/papers/robot_link_rotor_inertia_primary_refs.md) — URDF `<inertial>`、Modern Robotics Ch.8、Gautier–Khalil、MuJoCo `armature`
- [系统辨识论文摘录](../../sources/papers/system_identification.md) — 最小参数集与激励轨迹
- [Pinocchio 仓库归档](../../sources/repos/pinocchio.md)
- [FloBaRoID 仓库归档](../../sources/repos/flobaroid.md)

## 推荐继续阅读

- ROS Wiki：[urdf/XML/link — inertial](http://wiki.ros.org/urdf/XML/link)
- Traversaro et al., IROS 2016：[arXiv:1610.08703](https://arxiv.org/abs/1610.08703)
- Wensing, Kim, Slotine, RA-L 2018：[arXiv:1701.04395](https://arxiv.org/abs/1701.04395)
- Atkeson, An, Hollerbach, IJRR 1986：DOI [10.1177/027836498600500306](https://doi.org/10.1177/027836498600500306)
- MuJoCo：[compiler inertiafromgeom](https://mujoco.readthedocs.io/en/latest/XMLreference.html#compiler-inertiafromgeom)

## 一句话记忆

> URDF 惯量对照真机：先过符号和刚体不等式，再对总质量与 $g(q)$，最后才用激励轨迹估可辨识组合——CAD 数字不是力矩真值，转子惯量也不是 link 质量。
