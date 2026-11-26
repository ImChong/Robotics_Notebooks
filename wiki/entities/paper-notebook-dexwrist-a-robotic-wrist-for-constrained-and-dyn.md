---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-11-26
venue: curated
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_dexwrist-a-robotic-wrist-for-constrained-and-dyn.md
summary: "DexWrist（IEEE RA-L 2026）：将 2-(R, RR) 运动解耦球面并联机构与 13:1 准直驱执行器结合的紧凑 2-DoF 机械腕（0.97 kg，3.75 Nm，0.33 Nm 反驱力矩），显著提升受限空间可达性与接触动态操作学习性能。"
---

# DexWrist

**DexWrist: A Robotic Wrist for Constrained and Dynamic Manipulation**（Martin Peticco, Gabriella Ulloa, John Marangola, Nitish Dashora, Pulkit Agrawal，MIT Improbable AI Lab，IEEE RA-L 2026 / arXiv:2507.01008v4）提出了一种可即插即用替换商用机械臂末端腕部的 2-DoF 紧凑机器人手腕。条目收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：12_Hardware_Design）。

## 一句话定义

一种将**速度约束雅可比对角化的 2-(R, RR) 球面并联机构（PKM）**与**前臂内嵌 13:1 准直驱（QDD）执行器**相结合的 2 自由度机械腕，以 0.97 kg 紧凑包络实现共轴心大转角变姿、低机械阻抗碰撞耐受与 1 对 1 电机–自由度映射。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| PKM | Parallel Kinematic Mechanism | 并联运动学机构，可将电机置于近端从而大幅降低末端惯量 |
| QDD | Quasi-Direct-Drive | 准直驱执行器，采用低减速比传动以获得高反驱性与低等效转子惯量 |
| F/E & R/U | Flexion/Extension & Radial/Ulnar | 对应手腕的俯仰（pitch）与偏航（yaw）两个输出自由度 |
| BC | Behavior Cloning | 从人类遥操作演示数据中监督学习动作策略的模仿学习方法 |

## 为什么重要

- 现有商用 6/7-DoF 机械臂（如 UR3e、Franka FR3、AgileX PiPER）在受限杂乱空间与高速接触任务中存在明显腕部短板：要么采用高减速比（80:1–160:1）谐波减速器导致**极难反驱、碰撞易报错停机**；要么采用串联错开布置的腕关节导致**末端微调姿态需大幅摆动整条机械臂**；而传统并联腕又存在**非对角雅可比耦合**，控制建模复杂。
- DexWrist 证明仅通过腕部硬件升级（不改学习算法），就能在杂乱冰箱取物、白板擦除等任务上直接带来 **50–76%** 的策略成功率相对提升与 **3–5×** 的任务提速，说明**末端硬件阻抗与运动学共轴心**对下游模仿学习与遥操作效率具有决定性影响。

## 核心信息

| 字段 | 内容 |
|------|------|
| 作者与机构 | Martin Peticco, Gabriella Ulloa, John Marangola, Nitish Dashora, Pulkit Agrawal（MIT Improbable AI Lab） |
| 出处 | IEEE Robotics and Automation Letters (RA-L), 2026 / arXiv:2507.01008v4 |
| 项目页 / 原文 | <https://martinpeticco.com/dexwrist> / <https://arxiv.org/abs/2507.01008> |
| 驱动与机构 | 2-(R, RR) 解耦球面 PKM（pitch+yaw，假设上游提供 roll）+ 2× 定制 13:1 QDD（CubeMars GL40 + 单级复合行星轮） |
| 控制接口 | moteus-n1（16 V CAN 菊花链，1 kHz 力矩/电流闭环）+ AS5047P 磁编码器 |
| 分类 | 12_Hardware_Design（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md) 待深读） |
| 计划文件夹 | `papers/12_Hardware_Design/dexwrist-a-robotic-wrist-for-constrained-and-dyn` |

## 核心机制

1. **2-(R, RR) 速度解耦球面并联机构**：两个驱动轴 $e_1$（yaw 支链 R）和 $e_2$（pitch 支链 RR）固定在**前臂侧**而非随末端转动，所有轴线交汇于腕心（距输出法兰 48.75 mm）。连杆偏转角均为 $90^\circ$，仅 $\alpha_2 = \angle(e_2, e_4) = 95^\circ$（预留 $5^\circ$ 避让角使两轴均可无干涉到达 $\pm 40^\circ$）。由闭环约束 $e_3^\top v = 0$ 与 $e_4^\top e_5 = 0$ 微分得到速度约束雅可比矩阵 $K\dot{\theta} = J_0 \omega$，其中 **$K = \text{diag}(v^\top(e_1\times e_3),\ e_5^\top(e_2\times e_4))$ 严格对角**，实现单电机对应单约束的 1 对 1 速度解耦映射；在 $|\theta_i|\le 40^\circ$ 范围内 $\partial q_2/\partial \theta_2 \ge 0.98$ 且输出雅可比条件数 $\le 1.7$，正逆运动学均有解析闭式解。
2. **静力学可补偿上三角力矩映射**：由于 $q_1 = \theta_1$ 而 $q_2(\theta_1, \theta_2)$ 依赖双轴，$J^\top$ 呈上三角形式：$\tau_1 = M_1 + M_2 \frac{\partial q_2}{\partial \theta_1}$、$\tau_2 = M_2 \frac{\partial q_2}{\partial \theta_2}$。yaw 外力矩永远不会传递至 pitch 电机，而 pitch 力矩会按解析比例反映到 yaw 电机（$\theta_2=0$ 时为 0，在 $(20^\circ,20^\circ)/(30^\circ,30^\circ)/(40^\circ,40^\circ)$ 处分别为 15% / 31% / 50%），且可精确前馈补偿（台架实测未建模残差最大仅 0.044 Nm，占额定力矩 1.2%）。
3. **前臂集成 13:1 准直驱（QDD）执行器**：选用 CubeMars GL40 无刷云台电机（$J_m = 74\text{ g}\cdot\text{cm}^2$，额定约 0.25 Nm）搭配定制单级复合行星齿轮箱（13:1 减速比，1045 钢齿轮，Lewis 弯曲安全系数 $\ge 3$）。相比 80:1–160:1 谐波传动，其等效转子反射惯量 $N^2 J_m$ 降低 **40–150×**（5 kg@70 mm 负载下 $J_{\text{ref}}/J_L \approx 5\times 10^{-2} \ll 1$），无需复杂导纳控制即可在高速碰撞时靠低机械阻抗被动限幅接触力，并使 PD 位置控制自然产生可观测稳态接触偏转 $e \approx \tau_{\text{ext}}/K_p$。

### 方法流程图

```mermaid
flowchart LR
  Arm["机械臂上游 Roll 轴 (P/S)"] --> Forearm["前臂机架固定双 13:1 QDD 执行器 (1 kHz CAN)"]
  Forearm -->|"theta_1 (Yaw R 支链)"| PKM["2-(R, RR) 球面并联机构 (共轴心 +/-40 度)"]
  Forearm -->|"theta_2 (Pitch RR 支链, 95 度偏置)"| PKM
  PKM -->|"对角约束雅可比 K + 解析静力补偿"| EE["末端夹爪 / 输出法兰 (低远端惯量)"]
  EE -->|"被动低阻抗接触与偏转本体感知"| Policy["遥操作数据采集 & Diffusion Policy (30-50 Hz)"]
```

## 实验与评测

### 1. 机械与执行器实测指标（Table II）

| 指标 | 设计目标（参考人类 ADL / NASA） | DexWrist 实测 | 达标情况 |
|------|-------------------------------|--------------|---------|
| 额定持续力矩（60 s @ 70 mm） | $\ge 3$ Nm | **3.75 ± 0.05 Nm** | 达标 |
| 反驱力矩（$K_p=0$ 无前馈） | $\le 0.4$ Nm | **0.33 ± 0.06 Nm** | 达标 |
| 限位静态载荷 / 轴向载荷 | $\ge 9$ Nm / $\ge 15$ kg | **≥ 14 Nm** / **≥ 100 kg**（测试值） | 达标 |
| 空载最高转速 / 力矩带宽 | $\ge 53.3$ rpm / $\ge 10$ Hz | **96.6 ± 9.4 rpm** / **10.15 ± 1.34 Hz** | 达标 |
| 重复角定位精度 / ROM | $\le 3.47^\circ$ / F/E $\pm 40^\circ$, R/U $[-10^\circ, 30^\circ]$ | **1.65°** SD / 双轴 **[−40°, 40°]** | 达标 |
| 尺寸（长 × 宽 × 高）与质量 | $\le 195.5\times 61.4\times 61.4$ mm, $\le 1.0$ kg | **178.2 × 64 × 66.5 mm**, **0.97 kg** | 长/重达标（宽/高超 4% / 8%） |

### 2. 系统级下游任务表现（仿真、遥操作与 Diffusion Policy）

- **PyBullet 受限深柜可达工作空间（Fig. 6）**：在保持末端朝向固定时，替换 AgileX PiPER 原装串联腕后，无碰撞可达目标点数量提升 **88%**。
- **人类遥操作采集效率（4 项真实任务，共 500+ 条演示，Table IV & Fig. 7）**：
  - 成功演示平均时长缩短 **1.3–2.2×**（AgileX/UR3e；Franka 上缩短 **1.2×**）；包含重置与失败在内的操作员总耗时降低 **39.0–69.3%**（冰箱取罐 63.5→38.7 s，拔线缆 76.3→28.0 s，深抽屉取杯 57.2→29.6 s，UR3e 擦白板 21.5→6.6 s；Franka 24.2→22.4 s），重置次数全面下降（冰箱 1.7→1.0，UR3e/Franka 擦白板 0.2→0.0）。
- **Diffusion Policy 自主闭环评测（每配置 $\ge 15$ 次测试，Fig. 8 & Table V）**：
  - **AgileX + 杂乱冰箱取罐**（30 Hz 末端位姿控制）：成功率相对提升 **50%**，平均完成时间加速 **3.24×**（**91.0 ± 7.9 s → 28.1 ± 2.2 s**）。
  - **UR3e + 白板擦除**（原装腕须用笛卡尔导纳控制，DexWrist 直接关节位置控制 50 Hz）：成功率相对提升 **76%**，平均完成时间加速 **4.92×**（**21.2 ± 10.5 s → 4.3 ± 1.2 s**）。
  - **Franka FR3 + 白板擦除**（同等关节阻抗控制器与关节空间动作）：成功率均达饱和（原装 **19/20** vs. DexWrist **20/20**），完成时间仍加速 **1.43×**（**6.1 ± 1.4 s → 4.3 ± 1.3 s**，单侧 Mann-Whitney $p \approx 9\times 10^{-5}$）。

## 与其他工作对比

| 腕部架构 | 共轴心 | 速度解耦 | 非弹性高刚度可控 | 低反射转子惯量 | 低远端运动惯量 |
|----------|--------|----------|------------------|----------------|----------------|
| 串联腕（UR3e、Franka） | ✗ | ✓ | ✓ | 高 (✗) | 高 (✗) |
| 万向节 U-joint（OpenArm 等） | ✓ | ✓ | ✓ | 低 (✓) | 高 (✗)（第二电机随动） |
| 耦合并联 PKM（[ByteWrist](./paper-notebook-bytewrist-a-parallel-robotic-wrist-enabling-flex.md) 等） | ✓ | ✗（非对角雅可比） | ✓ | 低 (✓) | 高/中 (✗) |
| 变刚度/柔顺腕（Milazzo 等） | ✗/✓ | ✗ | ✗（弹性元件限制带宽/负载） | 高 (✗) | 低 (✓) |
| 高减速比解耦 PKM（Negrello） | ✓ | ✓ | ✗（350:1 重齿轮） | 高 (✗) | 低 (✓) |
| **DexWrist（本文）** | **✓** | **✓** | **✓** | **低 (✓)** | **低 (✓)** |

## 常见误区或局限

1. **「完全解耦」仅针对速度约束，不代表力矩与载荷分担解耦**：虽然约束雅可比 $K$ 对角、单电机控制单约束，但双电机**无法共享同一轴的负载**（每个电机必须独立承受单轴全部额定力矩）；且在静力学层面 pitch 外力矩仍会按 $\partial q_2/\partial \theta_1$ 比例耦合到 yaw 电机（极端拐角 $(40^\circ, 40^\circ)$ 达 50%），动态交叉耦合留待后续研究。
2. **机械复杂度与回程间隙**：相比串联关节，并联连杆与球铰引入一定背隙（实测角精度 SD $1.65^\circ$），因此更适合接触丰富、顺应性优先的操作任务，而不适用于微米级高精度刚性定位。
3. **工作范围、尺寸与封装限制**：$\pm 40^\circ$ 活动范围满足 ADL 基准，但小于人类手腕完整的最大屈伸幅度；宽/高（64/66.5 mm）略超 95 分位男性人体参考（61.4 mm）4% 与 8%；目前仅提供 2-DoF（假设上游机械臂带 roll 轴），且尚未配备连杆防夹护罩与内置走线设计。

## 结论

**DexWrist 证明了将「共轴心解耦球面 PKM」与「前臂 QDD 驱动」结合，能够同时释放遥操作效率与模仿学习策略的潜力——在极难碰撞的硬质谐波臂（UR3e）和长连杆串联臂（AgileX）上带来数倍提速与大幅成功率跃升。**

1. **硬件顺应性可直接替代复杂的软件导纳调参**：40–150× 的反射惯量降幅让 UR3e 无需力矩传感器与导纳控制即可用普通关节位置 PD 稳定擦白板，且 PD 跟踪误差天然提供接触本体信号。
2. **更短更紧凑的演示是 BC 提效的关键桥梁**：共轴心变姿减少了整臂在狭小空间内的迂回动作，将遥操作耗时削减 39–69%，同时缩短轨迹步长 $T$ 直接收紧了行为克隆 $O(\epsilon T^2)$ 的误差累积上界。
3. **即便已有高质量关节阻抗控制（Franka FR3），更轻的远端腕部仍可提速 1.43×**：在相同控制器和动作空间下隔离验证了纯硬件低惯量/紧凑几何的独立收益。
4. **选型边界清晰**：适合需要碰撞容忍、高速动态接触或狭小空间变姿的操作平台；若任务要求极高绝对定位精度或需要独立 3-DoF 旋转且上游无 roll 轴，则需额外集成第 3 轴或改进连杆消隙。

## 与其他页面的关系

- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 关联页面

- [Manipulation](../tasks/manipulation.md)：对照相关方法与适用场景。
- [机器人操作系统与末端执行器](../methods/robotic-manipulation-systems.md)：对照相关方法与适用场景。

## 参考来源

- [humanoid_pnb_dexwrist-a-robotic-wrist-for-constrained-and-dyn.md](../../sources/papers/humanoid_pnb_dexwrist-a-robotic-wrist-for-constrained-and-dyn.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文原文：Martin Peticco et al., *DexWrist: A Robotic Wrist for Constrained and Dynamic Manipulation*, IEEE RA-L 2026, [arXiv:2507.01008](https://arxiv.org/abs/2507.01008)

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- [DexWrist 项目主页](https://martinpeticco.com/dexwrist)
