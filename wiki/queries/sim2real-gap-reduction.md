---
type: query
tags: [sim2real, domain-randomization, locomotion, deployment, training]
status: complete
summary: "Sim2Real Gap 缩减实战指南"
updated: 2026-04-25
sources:
  - ../../sources/papers/sim2real.md
  - ../../sources/papers/privileged_training.md
  - ../../sources/papers/simulation_tools.md
  - ../../sources/papers/system_identification.md
---

# Sim2Real Gap 缩减实战指南

> **Query 产物**：本页由以下问题触发：「sim2real transfer 失败的根因分类，以及对应的缩减策略？」
> 综合来源：[sim2real](../concepts/sim2real.md)、[domain-randomization](../concepts/domain-randomization.md)、[privileged-training](../concepts/privileged-training.md)、[sim2real-checklist](./sim2real-checklist.md)

## TL;DR 决策路径

```
sim2real 迁移失败？先定位根因：

1. 动力学模型误差（Model Error）
   └─ 修复：系统辨识 + 物理参数 DR + ActuatorNet

2. 执行器动力学误差（Actuator Dynamics）
   └─ 修复：ActuatorNet / 执行器延迟建模 + 低通滤波

3. 感知误差（Sensing Gap）
   └─ 修复：传感器噪声 DR + 状态估计 + 视觉 sim2real

4. 控制频率不匹配
   └─ 修复：对齐仿真与真机控制频率（通常 50-200Hz）

5. 策略过拟合仿真（Simulation Overfitting）
   └─ 修复：加强 DR 范围 / 特权信息蒸馏 / 在线适应（RMA）
```

## Gap 分类与对应修复策略

### Gap 1：模型动力学误差

**表现**：策略在仿真中稳定行走，但真机出现持续漂移或抖动。

**根因**：惯性参数（质量/质心位置/转动惯量）不准确，关节摩擦系数差异。

**修复工具包**：
| 方法 | 描述 | 复杂度 |
|------|------|--------|
| 物理参数 DR | 随机化质量/摩擦系数/阻尼 ±20-50% | 低 |
| 系统辨识 | 实测物理参数，替换 URDF | 中 |
| Push Randomization | 训练时随机施加外力 | 低 |
| 质心位置偏差 | 随机化 CoM 位置 ±3cm | 低 |

**典型配置**（legged_gym 风格）：
```python
randomize_friction: [0.5, 1.25]    # 地面摩擦系数
randomize_base_mass: [-1, 3]       # 基座质量偏差 (kg)
randomize_com_pos: [-0.05, 0.05]   # CoM 位置偏差 (m)
push_robots: True                   # 随机推力 [0, 150] N
```

### Gap 2：执行器动力学误差

**表现**：策略输出的关节力矩与真机执行有延迟，或力矩响应曲线不符合仿真。

**根因**：仿真中执行器通常是理想力矩/速度源；真机有电机惯性、减速器弹性、控制器延迟（通常 10-30ms）。

**修复工具包**：
| 方法 | 描述 |
|------|------|
| ActuatorNet（Hwangbo 2019） | 用 MLP 学习执行器的输入-输出映射（从真机数据） |
| 延迟建模 | 在仿真中加入随机延迟 1-5 个控制步 |
| PD 控制模式 | 用 PD 目标角度代替直接力矩（更鲁棒） |
| 低通滤波 | 对策略输出的动作做一阶低通滤波 |

### Gap 3：感知误差

**表现**：IMU 数据有噪声漂移，接触检测不准确，视觉 sim2real 纹理/光照差异。

**修复工具包**：
```python
# 仿真 IMU 噪声配置
imu_noise_vel:  [0.0, 0.1]     # 速度估计噪声 (m/s)
imu_noise_grav: [0.0, 0.05]    # 重力分量噪声
contact_noise:  0.05            # 接触检测误报率
```

**特权信息蒸馏（推荐）**：
- Teacher 用完美状态（地形高度图、精确速度）训练
- Student 用历史观测历史（IMU + 关节编码器）模仿 teacher
- 真机只跑 student → 无需视觉 sim2real

### Gap 4：控制频率与时序

**真机 vs 仿真频率对齐 checklist**：

- [ ] 仿真控制频率 = 真机控制频率（通常 50-200 Hz）
- [ ] 策略推理延迟 < 一个控制周期（1/freq）
- [ ] 通信延迟（ROS2 / EtherCAT）已在仿真中建模
- [ ] 关节位置/速度的读取顺序与仿真一致

### Gap 5：策略过拟合仿真

**表现**：增大 DR 范围后仿真性能下降，减小后真机又失败。

**修复方法**：

1. **在线适应（RMA）**：训练适应模块实时估计环境参数
2. **历史观测作为输入**：用过去 N 步的观测（而非单帧）隐式估计环境参数
3. **domain randomization 双阶段训练**：
   - Phase 1：在仿真（强 DR）训练 base policy
   - Phase 2：在真机（少量数据）fine-tune

## 完整 Sim2Real Pipeline Checklist

### 阶段 1：仿真配置
- [ ] URDF 参数准确（可用 system identification 工具）
- [ ] 执行器模型：ActuatorNet 或 PD + 延迟
- [ ] 控制频率与真机一致
- [ ] DR 范围：物理参数 ±30%，推力随机化

### 阶段 2：训练配置
- [ ] 地形课程从简单到复杂
- [ ] 策略输入包含历史观测（3-5 步）
- [ ] 惩罚项防止过激动作（关节速度/力矩限制）
- [ ] 若有示范数据：加 AMP 或 teacher 蒸馏

### 阶段 3：部署验证
- [ ] 先在仿真中跑满参数化范围测试
- [ ] 首次真机测试：低速 + 安全装置 + 人工监护
- [ ] 监控量：关节温度、电流峰值、控制延迟
- [ ] 逐步提升速度/地形难度

## 参考来源

- [sources/papers/sim2real.md](../../sources/papers/sim2real.md) — DR / RMA / InEKF ingest 摘要
- [sources/papers/privileged_training.md](../../sources/papers/privileged_training.md) — teacher-student / RMA 适应模块
- [sources/papers/simulation_tools.md](../../sources/papers/simulation_tools.md) — 仿真平台原论文
- [sources/papers/system_identification.md](../../sources/papers/system_identification.md) — 执行器建模 ActuatorNet

## 关联页面

- [Sim2Real](../concepts/sim2real.md) — sim2real 概念综述
- [Domain Randomization](../concepts/domain-randomization.md) — DR 方法详解
- [Privileged Training](../concepts/privileged-training.md) — teacher-student 蒸馏
- [Sim2Real Checklist](./sim2real-checklist.md) — 部署前完整检查清单

## 一句话记忆

> Sim2Real Gap = 模型误差 + 执行器延迟 + 感知噪声；对症下药：系统辨识 + ActuatorNet + DR + 特权信息蒸馏。
