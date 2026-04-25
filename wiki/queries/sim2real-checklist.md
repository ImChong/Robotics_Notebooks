---
type: query
tags: [sim2real, deployment, locomotion, humanoid, rl]
status: complete
summary: "> **Query 产物**：本页由以下问题触发：「从仿真到真机部署，有哪些必须检查的工程事项？」"
updated: 2026-04-25
sources:
  - ../../sources/papers/sim2real.md
---

> **Query 产物**：本页由以下问题触发：「从仿真到真机部署，有哪些必须检查的工程事项？」
> 综合来源：[Sim2Real](../concepts/sim2real.md)、[Domain Randomization](../concepts/domain-randomization.md)、[System Identification](../concepts/system-identification.md)、[Privileged Training](../concepts/privileged-training.md)、[Locomotion](../tasks/locomotion.md)

# Sim2Real 工程 Checklist

从仿真训练到真实机器人部署的工程清单。每个阶段都有容易踩的坑，按顺序逐项核查。

---

## 阶段 0：仿真建模（训练之前）

### 机器人模型
- [ ] URDF/MJCF 模型质量参数是否精确（质量、惯性张量、质心位置）
- [ ] 关节限位是否与真实硬件一致（角度范围 + 速度限制）
- [ ] 足端接触几何是否合理（球形/胶囊体/真实形状）
- [ ] 电机模型是否包含力矩限制和速度-力矩曲线

### 仿真器选择
- [ ] 接触稳定性：MuJoCo（稳定）> PyBullet（不推荐用于精细接触）> Isaac Gym/Lab（速度最快）
- [ ] 控制频率：建议 ≥ 100Hz；真实机器人低于 50Hz 难以稳定
- [ ] 渲染（视觉策略）：IsaacLab 支持大规模视觉训练

---

## 阶段 1：域随机化配置（核心）

### 物理参数随机化
- [ ] **质量随机化**：基座 ±20%，末端 ±30%（含负载）
- [ ] **惯性张量随机化**：±20%
- [ ] **关节阻尼/摩擦随机化**：±50%
- [ ] **地面摩擦系数随机化**：0.3 ~ 1.2（覆盖木地板到地毯）
- [ ] **地面弹性系数随机化**：0.0 ~ 0.3

### 执行器随机化
- [ ] **控制延迟随机化**：1~3 个控制周期（真实硬件常有 5~20ms 延迟）
- [ ] **动作噪声**：高斯噪声加在输出上，σ = 0.01~0.05 rad
- [ ] **电机力矩误差**：±5%（建模误差）
- [ ] **PD 增益随机化**（如果用位置控制）：Kp/Kd ±20%

### 传感器随机化
- [ ] **IMU 噪声**：加速度计 / 陀螺仪白噪声
- [ ] **关节编码器噪声**：±0.01 rad
- [ ] **观测延迟**：1 步延迟（真实控制环路中常见）
- [ ] **观测 Dropout**（可选）：随机置零部分观测

### 地形随机化（locomotion 专项）
- [ ] 平地基线（必须先能在平地上成功）
- [ ] 随机高度场（uniform noise ±3cm）
- [ ] 斜坡（±15°）
- [ ] 台阶（高度 0.1~0.2m）
- [ ] 离散障碍（石块、砖头）

---

## 阶段 2：策略验证（上机之前）

### 仿真内验证
- [ ] 训练曲线：reward 稳定收敛，无明显下降
- [ ] 对照实验：取消域随机化后策略是否过拟合（验证随机化的必要性）
- [ ] 鲁棒性测试：在训练范围边界（最大随机化）下成功率 > 80%
- [ ] 扰动测试：施加 50N 推力后能恢复（见 [Balance Recovery](../tasks/balance-recovery.md)）
- [ ] 速度跟踪：0 ~ 1.5 m/s 范围内跟踪误差 < 15%

### 策略输出检查
- [ ] 动作幅度：关节位置目标是否在安全范围内（不超关节限位）
- [ ] 动作平滑性：相邻两步动作差 < 0.1 rad（过大则真实关节损耗高）
- [ ] 站立测试：在零速度指令下能稳定站立 > 30s

---

## 阶段 3：系统辨识（上机前）

- [ ] **执行器辨识**：用 chirp 信号测量实际力矩-速度曲线
- [ ] **关节摩擦辨识**：匀速运动下测量摩擦力矩
- [ ] **质量辨识**：用 IMU + 已知激励估计实际质量分布
- [ ] **控制延迟测量**：发送阶跃指令，测量实际响应延迟
- [ ] 将辨识结果用于缩小域随机化范围（集中在真实参数附近）

---

## 阶段 4：真机初次部署

### 安全措施
- [ ] **吊绳/支架保护**：首次上机必须有物理安全保护
- [ ] **急停机制**：硬件级急停按钮，软件级力矩截断
- [ ] **力矩上限**：在策略输出上叠加硬性力矩限制（通常为额定力矩 80%）
- [ ] **站立测试先行**：先测试静止站立，再测试行走

### 第一步测试顺序
1. 站立（双脚支撑，零速度指令）→ 观察稳定性，≥ 30s
2. 原地踏步（足端抬起 5cm，步频 1Hz）
3. 极慢速直走（0.1 m/s）→ 确认方向正确
4. 正常速度（0.5 m/s）
5. 转弯
6. 复杂地形

### 观察项
- [ ] 关节力矩是否异常（过热、振荡）
- [ ] 步态是否与仿真一致（不对称、异常姿态）
- [ ] IMU 读数是否合理（无毛刺）

---

## 阶段 5：Gap 诊断与闭环

### 常见 Sim2Real Gap 及修复

| 现象 | 原因 | 修复 |
|------|------|------|
| 步态明显抖动 | 控制延迟未建模 | 在仿真中增大延迟随机化范围 |
| 向一侧偏转 | 质量不对称/摩擦差异 | 辨识真实质量参数，收窄随机化 |
| 关节力矩过大 | 电机模型不准 | 辨识真实力矩-速度曲线 |
| 速度比仿真慢 | 摩擦系数偏高 | 在仿真中提高地面摩擦随机化上界 |
| 遇到小障碍就摔倒 | 足端感知不足 | 加入足端力传感器观测/域随机化 |
| 停止后向后倒 | 站立策略不够鲁棒 | 增加站立稳定性 reward 项 |

### RMA 在线适应（可选）
如果使用 RMA 框架，Adaptation Module 会在真机运行中自动估计环境参数，逐渐缩小 gap。见 [Privileged Training](../concepts/privileged-training.md)。

---

## 关联页面

- [Sim2Real](../concepts/sim2real.md) — 理论方法总览
- [Domain Randomization](../concepts/domain-randomization.md) — 随机化策略详解
- [System Identification](../concepts/system-identification.md) — 参数辨识方法
- [Privileged Training](../concepts/privileged-training.md) — Teacher-Student / RMA 框架
- [Balance Recovery](../tasks/balance-recovery.md) — 扰动测试与恢复策略
- [Locomotion](../tasks/locomotion.md) — locomotion 任务完整描述

## 参考来源

- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021) — RMA 两阶段框架
- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* (2017) — 域随机化基础
- Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain* (Science Robotics, 2020) — 足式 sim2real 完整 pipeline
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md) — DR / RMA / InEKF ingest 摘要
