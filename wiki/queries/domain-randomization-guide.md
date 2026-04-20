---
type: query
tags: [sim2real, domain-randomization, rl, locomotion, training, parameters]
status: complete
summary: "如何为 sim2real 任务设计 domain randomization 参数与分布范围：涵盖物理参数、延迟、感知噪声和几何随机化的完整指导手册。"
sources:
  - ../../sources/papers/locomotion_rl.md
  - ../../sources/papers/privileged_training.md
related:
  - ../concepts/domain-randomization.md
  - ../concepts/sim2real.md
  - ../queries/sim2real-deployment-checklist.md
---

# Query：sim2real Domain Randomization 参数设计指南

> **Query 产物**：本页由以下问题触发：「做 sim2real，如何设计 domain randomization 的参数与分布范围？」
> 综合来源：[Domain Randomization](../concepts/domain-randomization.md)、[Sim2Real](../concepts/sim2real.md)、[Sim2Real 部署检查清单](../queries/sim2real-deployment-checklist.md)

## TL;DR 决策规则

| 参数类别 | 典型随机化维度 | 推荐分布 | 推荐范围 |
|---------|-------------|---------|---------|
| **物理-质量** | 全身连杆质量 | Uniform | ±20% 标称值 |
| **物理-摩擦** | 地面摩擦系数 | Uniform | 0.3 ~ 1.5 |
| **物理-阻尼** | 关节阻尼系数 | Uniform | ±50% 标称值 |
| **物理-刚度** | 关节 PD 刚度 Kp | Uniform | ±30% 标称值 |
| **物理-质心** | 连杆质心偏移 | Uniform | ±5 cm |
| **延迟-动作** | Action delay | Uniform | 0 ~ 40 ms |
| **延迟-观测** | Observation delay | Uniform | 0 ~ 20 ms |
| **噪声-IMU** | 线加速度偏置 | Normal(0, σ) | σ = 0.1 m/s² |
| **噪声-IMU** | 角速度偏置 | Normal(0, σ) | σ = 0.05 rad/s |
| **噪声-编码器** | 关节位置噪声 | Normal(0, σ) | σ = 0.01 rad |
| **噪声-编码器** | 关节速度噪声 | Normal(0, σ) | σ = 0.05 rad/s |
| **几何** | 地形高低起伏 | Uniform | 0 ~ 3 cm（平地任务），0 ~ 15 cm（地形自适应） |
| **几何** | 地形坡度 | Uniform | 0° ~ 10° |

**决策优先级**：延迟 > 摩擦系数 > 质量/质心 > 阻尼/刚度 > 传感器噪声

---

## 详细内容

### 1. 物理参数随机化

#### 1.1 质量与惯量

质量误差是 sim2real 中最常见的 gap 来源之一，主要原因是：
- 仿真 URDF/MJCF 文件中的质量通常来自 CAD 模型，未经实测
- 实际机器人有线缆、连接件等附加质量

推荐做法：
```
m_sim ~ Uniform(0.8 × m_nominal, 1.2 × m_nominal)  # ±20%
com_offset ~ Uniform(-0.05, 0.05) m  # 每个轴独立随机化
```

惯量矩阵可以通过对角元素分别随机化 ±30%，非对角项保持不变。对于人形机器人，躯干（torso）的惯量随机化权重应设更高，因为其对平衡影响最大。

#### 1.2 摩擦系数

地面摩擦系数是足式机器人 sim2real 中影响最显著的参数，现实地面材质差异大（木地板 μ ≈ 0.6，混凝土 μ ≈ 0.8，光滑瓷砖 μ ≈ 0.3~0.4）。

推荐做法：
```
μ_ground ~ Uniform(0.3, 1.5)  # 覆盖从光滑地面到粗糙橡胶地垫
μ_foot   ~ Uniform(0.5, 1.0)  # 足底摩擦，与地面摩擦组合
```

注意：过低的摩擦系数（< 0.2）会使策略无法维持直立，应与课程学习结合使用，初期设定为 0.5~1.5，后期才引入低摩擦场景。

#### 1.3 关节参数

关节阻尼（damping）和刚度（stiffness）的实际值与仿真模型常有 20%~50% 偏差：

```
damping_j   ~ Uniform(0.5, 1.5) × damping_nominal_j
stiffness_j ~ Uniform(0.7, 1.3) × stiffness_nominal_j
```

电机力矩常数（motor torque constant）影响执行效果，推荐 ±10%~20% 随机化。

#### 1.4 外力干扰

在训练中加入随机外力扰动是物理随机化的有效补充：
```
F_ext_body  ~ Uniform(-30, 30) N，持续 0.1~0.5 s
τ_ext_joint ~ Uniform(-2, 2) N·m（各关节独立）
```

---

### 2. 延迟随机化

延迟是 sim2real 中最容易被忽视的 gap，也是最致命的一类。

#### 2.1 Action Delay（动作延迟）

从策略输出动作到电机真正执行之间，真实系统存在以下延迟来源：
- 通信总线延迟（CAN/EtherCAT）：1~5 ms
- 上位机计算延迟：5~20 ms
- 电机驱动器响应：1~3 ms
- 控制周期离散化：取决于控制频率（50Hz → 20ms 步长）

推荐做法：
```
action_delay ~ Uniform(0, 40) ms  # 整体延迟
# 在仿真中实现时，通常离散化为控制步长的整数倍：
n_delay_steps ~ RandInt(0, 4)  # 在 50Hz 下等价于 0~80ms
```

实现方式：维护一个动作历史队列，将当前时刻的动作延迟 `n_delay_steps` 步后再喂给仿真执行器。

#### 2.2 Observation Delay（观测延迟）

传感器数据到达策略输入端也有延迟：
- IMU 数据：通常 1~5 ms
- 关节编码器：1~3 ms
- 相机/深度传感器：33~100 ms（取决于帧率和处理延迟）

推荐做法：
```
obs_delay ~ Uniform(0, 20) ms  # 本体感知观测延迟
```

注意：同时存在 action delay 和 observation delay 时，有效控制延迟会叠加，总延迟应控制在合理范围内，避免策略在训练中完全无法利用当前状态信息。

---

### 3. 感知噪声随机化

#### 3.1 IMU 噪声

IMU（惯性测量单元）噪声模型包含：
- **白噪声（随机游走）**：每个时间步独立采样
- **偏置漂移（bias drift）**：缓慢变化的系统误差
- **尺度误差（scale factor）**：传感器增益偏差

推荐训练时随机化：
```python
# 线加速度
acc_noise  = N(0, 0.1)   # m/s², 白噪声
acc_bias   = Uniform(-0.2, 0.2)  # m/s², 每个 episode 固定
# 角速度
gyro_noise = N(0, 0.05)  # rad/s, 白噪声
gyro_bias  = Uniform(-0.1, 0.1)  # rad/s, 每个 episode 固定
```

偏置在每个训练 episode 开始时重新采样并保持固定，模拟真实传感器的慢漂移特性。

#### 3.2 关节编码器噪声

关节位置和速度噪声：
```python
q_noise   = N(0, 0.01)   # rad，位置量化噪声（12 bit encoder）
dq_noise  = N(0, 0.05)   # rad/s，速度差分噪声
```

对于低精度编码器（如磁编码器），位置噪声可增至 σ = 0.02~0.05 rad。

速度通常由位置差分计算，噪声会被差分放大，实际配置时 `dq_noise` 往往比 `q_noise` 大一个量级。

#### 3.3 状态估计误差

如果策略输入包含估计的基座速度或姿态（而非直接测量值），应对这些估计量加噪声：
```python
base_vel_noise  = N(0, 0.1)   # m/s，估计速度噪声
base_ori_noise  = N(0, 0.05)  # rad，姿态估计噪声（roll/pitch）
```

---

### 4. 几何与环境随机化

#### 4.1 地形随机化

针对地形自适应任务，将地形高度场参数化随机化：

| 地形类型 | 随机化范围 | 适用阶段 |
|---------|---------|---------|
| 平地（平坦地面） | 高低差 0~1 cm | 初始训练 |
| 粗糙地面 | 高低差 0~3 cm，空间频率高 | 中期 |
| 台阶 | 高度 0~15 cm，深度 20~60 cm | 地形泛化 |
| 斜坡 | 倾角 0°~15° | 地形泛化 |
| 离散障碍 | 高度 0~10 cm，间距 30~60 cm | 挑战期 |

#### 4.2 初始状态随机化

策略的初始状态分布对训练稳健性影响很大：
```python
init_joint_pos ~ Uniform(default_pos - 0.1, default_pos + 0.1)  # rad
init_base_vel  ~ Uniform(-0.5, 0.5)  # m/s，各方向独立
init_base_ori  ~ Uniform(-0.05, 0.05)  # rad，小倾斜扰动
```

---

### 5. 分布类型选择

#### 5.1 Uniform vs Normal

| 分布类型 | 适用场景 | 优缺点 |
|---------|---------|-------|
| **Uniform(a, b)** | 参数范围明确，希望覆盖极端值 | 极端值被均等采样，有助于鲁棒性；但可能过度训练边界情况 |
| **Normal(μ, σ)** | 标称值附近最常见，极端情况少见 | 更自然地反映现实分布；需设置截断范围（clip）避免极端值 |
| **LogUniform** | 跨量级参数（如刚度系数 10~1000）| 在对数尺度上均匀，更好覆盖低值和高值 |

**实践建议**：
- 延迟和噪声：使用 Uniform，确保策略在最坏情况下也能工作
- 物理参数（质量、摩擦）：Uniform 是主流，简单有效
- 传感器偏置（每 episode 固定）：Normal 更接近真实传感器特性

#### 5.2 Curriculum 与范围动态调整

固定随机化范围的问题：训练初期范围太大可能导致任务无法学习。推荐分阶段扩大：

```
阶段 1：μ_ground ∈ [0.6, 1.2]，无延迟，无噪声
         → 验证：策略可以在平地行走
阶段 2：μ_ground ∈ [0.4, 1.5]，action_delay ∈ [0, 20ms]
         → 验证：策略在轻度扰动下稳定
阶段 3：完整 DR 范围，加入地形
         → 验证：策略在完整随机化下成功率 > 80%
```

---

### 6. 常见误区

#### 误区 1：随机化范围越大越好

范围过大会使任务变得无法学习，尤其在训练早期。典型症状：
- 策略 reward 长时间不上升
- 策略退化为保守的不动策略（站立不走）

**修复**：用 curriculum 从小范围开始，确认策略收敛后再扩大范围。

#### 误区 2：忽略延迟随机化

许多论文和开源实现只随机化物理参数，忽略延迟。这会导致仿真中策略表现优秀但真机失败，因为真实系统总存在不可忽视的通信和计算延迟。

**修复**：至少添加 action_delay ~ Uniform(0, 3) 步（在 50Hz 下即 0~60ms）。

#### 误区 3：所有参数重要性相同

对足式运动来说，参数影响排序通常是：
```
地面摩擦 > 动作延迟 > 质量/质心 > 关节阻尼 > 传感器噪声
```

应优先花时间调整高影响参数的随机化范围。

#### 误区 4：用仿真成功率衡量 DR 效果

仿真成功率高不代表 sim2real 成功。正确的验证方法是在真机上测试，或使用系统辨识（SysID）测量真实参数是否落在随机化范围内。

#### 误区 5：过小的随机化范围导致过拟合仿真

如果所有参数都锁定在标称值附近 ±5%，策略会学到依赖仿真特定的物理特性，真机部署时遇到轻微参数差异就失败。

---

## 参考来源

- [sources/papers/locomotion_rl.md](../../sources/papers/locomotion_rl.md) — 足式运动 RL 训练与 DR 设计实践
- [sources/papers/privileged_training.md](../../sources/papers/privileged_training.md) — Teacher-Student 训练框架中的 DR 配合策略
- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* (IROS 2017)
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (RSS 2021) — DR 配合自适应模块的完整流程

## 关联页面

- [Domain Randomization（概念）](../concepts/domain-randomization.md) — DR 核心概念与主要类型
- [Sim2Real（概念）](../concepts/sim2real.md) — Sim2Real 全景：DR 在其中的位置
- [Query：Sim2Real 部署检查清单](../queries/sim2real-deployment-checklist.md) — 部署前的逐项验证流程
