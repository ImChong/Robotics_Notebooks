---
type: query
tags: [sim2real, deployment, locomotion, humanoid, rl]
status: complete
summary: "从仿真到真机部署的完整工程清单（含快速部署检查 3 分钟版）。"
updated: 2026-11-07
related:
  - ./sim2real-gap-reduction.md
  - ./sim2real-closed-loop-engineering.md
  - ./robot-policy-debug-playbook.md
  - ../concepts/sim2real.md
original:
  - ../../sources/papers/rma_reading_arxiv_2107_04034_pt3.md
  - ../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md
  - ../../sources/papers/rapid_locomotion_rl_arxiv_2205_02824.md
  - ../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md
  - ../../sources/papers/extreme_parkour_arxiv_2309_14341.md
  - ../../sources/papers/domain_randomization_tobin_reading_arxiv_1703_06907.md
sources:
  - ../../sources/papers/sim2real.md
  - ../../sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md
---

> **Query 产物**：本页由以下问题触发：「从仿真到真机部署，有哪些必须检查的工程事项？」（含原「真机部署 RL 策略前后要检查什么？」）
> 综合来源：[Sim2Real](../concepts/sim2real.md)、[Domain Randomization](../concepts/domain-randomization.md)、[System Identification](../concepts/system-identification.md)、[Privileged Training](../concepts/privileged-training.md)、[Locomotion](../tasks/locomotion.md)

# Sim2Real 工程 Checklist

从仿真训练到真实机器人部署的工程清单。每个阶段都有容易踩的坑，按顺序逐项核查。

## 快速部署检查

> **3 分钟版**：上机前只看本节 + [阶段 4：真机初次部署](#阶段-4真机初次部署)。迁移失败根因分类见 [Gap 缩减指南](./sim2real-gap-reduction.md)；已上机后的症状排查见 [真机调试 Playbook](./robot-policy-debug-playbook.md)。

### 训练端准备

| 检查项 | 典型范围 | 通过标准 |
|-------|---------|---------|
| 质量/惯量随机化 | ±20% | 仿真中性能稳定 |
| 关节摩擦/阻尼随机化 | ±30% | 无明显策略退化 |
| 延迟随机化 | 0~40ms | 覆盖真实延迟上限 |
| 观测噪声 | 参照真机规格 | 噪声量级与真机一致 |

- [ ] 关节力矩限制在仿真中强制执行（与真机规格一致）
- [ ] 策略输出频率与真机控制频率匹配（50Hz 或 100Hz）
- [ ] 观测归一化参数在导出时固定（不依赖运行时统计）

### 部署端检查

- [ ] 模型文件版本与训练代码一致
- [ ] 观测空间维度和顺序与真机接口匹配（逐字段核对）
- [ ] 动作空间映射：仿真关节顺序 ↔ 真机 CAN ID 顺序
- [ ] 推理延迟测试 < 5ms
- [ ] 初始姿态与仿真初始化姿态一致（误差 < 5°）

### 调试端常见问题

| 现象 | 可能原因 | 排查方向 |
|-----|---------|---------|
| 上机立刻摔倒 | 观测顺序错误 | 核对 obs/action 映射 |
| 原地抖动 | 控制频率不匹配 | 检查 PD 增益 |
| 向一侧漂移 | IMU 安装偏差 | 校准 IMU offset |
| 步伐极小 | 命令归一化错误 | 打印原始策略输入 |
| 关节力矩饱和 | 力矩映射错误 | 核查 scale 因子 |

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| MJCF | MuJoCo XML Format | MuJoCo 的模型与场景描述格式 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真训练环境 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| Kp | Proportional Gain | PD 控制的位置误差增益，影响刚度与响应 |
| Kd | Derivative Gain | PD 控制的速度误差增益，抑制振荡 |
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| RMA | Rapid Motor Adaptation | 从历史轨迹隐式估计环境参数的快速运动自适应 |
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |

## 已发表随机化范围对照（2026-11-07 核对原文）

本页阶段 1 的百分比范围（质量 ±20%、延迟 0~40 ms、PD ±20% 等）是**工程经验值**（推测，非论文原文）。下表给出一手论文实际使用的范围，用于校准：

| 参数 | RMA（A1，训练范围） | Margolis 2022（Mini Cheetah） | DreamWaQ++（Go1） | Hwangbo 2019（ANYmal） |
|------|--------------------|-------------------------------|-------------------|------------------------|
| 摩擦 | 0.05 – 4.5 | 0.05 – 4.00 | 0.2 – 1.25 | —（接触模型为硬接触 + 库仑摩擦） |
| 负载 | 0 – 6 kg | −1.0 – 3.0 kg | −1 – 2 kg | 质量 U(−15, 15)% |
| 质心 | ±0.15 m（表 I 单位以原文为准） | ±0.10 m | ±50 mm | U(−2, 2) cm（连杆质心与关节位置各自） |
| 电机强度 | ×0.90 – 1.10 | 90% – 110% | ×0.9 – 1.1 | 用执行器网络建模，不随机化 |
| PD 增益 | Kp 50–60，Kd 0.4–0.8 | 固定 Kp=20、Kd=0.5 | Kp、Kd 因子 ×0.9 – 1.1 | 固定 kp=50、kd=0.1 |
| 延迟 | —（异步结构，未单独随机化延迟） | —（本文未报告延迟随机化） | 本体 0 – 15 ms，外感知三档噪声 + 安装偏置 | 由执行器网络吸收通信与机械延迟 |
| 观测噪声 | 表 I 未列 | —（仅含地形噪声消融） | 关节位置 ±0.01 rad、关节速度 ±1.5 rad/s、体线速度 ±0.1 m/s、角速度 ±0.2 rad/s、重力向量 ±0.05 | 关节速度 U(−0.5, 0.5) rad/s，体线速度 ±0.08 m/s，角速度 ±0.16 |
| 其他 | 重采样概率 0.004/步 | 地面恢复系数 0 – 1 | Roll-Drop；外感知安装位姿偏置 | 30 个随机惯性模型；施加扰动 |

读表结论：

- 论文里的摩擦范围比本页建议的 0.3~1.2 宽得多（RMA 到 4.5），说明「地面摩擦」是被刻意放宽的随机化维度；而 Hwangbo 在执行器上走的是**建模**而非**随机化**路线。
- 延迟随机化 0~40 ms 的取值**在这几篇论文中没有出现**；DreamWaQ++ 明确把本体延迟限制在 0–15 ms，并说再加大会让策略过保守。因此「覆盖真实延迟上限」应以**实测延迟**为准，而不是照搬 40 ms。
- 控制频率：RMA 100 Hz，Margolis 50 Hz，DreamWaQ++ 50 Hz（本体 200 Hz），Hwangbo 200 / 100 Hz，Digit 50 Hz（PD 1 kHz）。本页「50 Hz 或 100 Hz」的建议与之相符。
- 动作与增益：想在 DR 里扰动 PD 增益，可参考 RMA（Kp 50–60、Kd 0.4–0.8）与 DreamWaQ++（×0.9–1.1）；本页「±20%」比两者都宽（经验值）。
- DR 的代价：Margolis 与 RMA 都指出域随机化让策略偏保守；用特权编码器 + 历史回归（RMA / 在线系统辨识）可缓解——Margolis 表 IV 中带在线辨识的真机速度 3.81 m/s 对无辨识 2.49 m/s。
- 域随机化的起源（Tobin 2017）针对**视觉**迁移：随机纹理、干扰物、相机位姿、光照，在 480 张真机图像上达到平均 1.5 cm 内的定位精度，不需要真实图像预训练；不能直接当作动力学随机化的数值依据。

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
- [Sim2Real Gap 缩减指南](./sim2real-gap-reduction.md) — 迁移失败根因分类与修复决策树
- [Sim2Real 闭环误差分层工程](./sim2real-closed-loop-engineering.md) — 从 SysID 到在线适应与分层安全的叙事骨架
- [RL 策略真机调试 Playbook](./robot-policy-debug-playbook.md) — 已上机后的系统排查
- [Domain Randomization](../concepts/domain-randomization.md) — 随机化策略详解
- [System Identification](../concepts/system-identification.md) — 参数辨识方法
- [Privileged Training](../concepts/privileged-training.md) — Teacher-Student / RMA 框架
- [Balance Recovery](../tasks/balance-recovery.md) — 扰动测试与恢复策略
- [Locomotion](../tasks/locomotion.md) — locomotion 任务完整描述

## 参考来源

- Kumar et al., [*RMA: Rapid Motor Adaptation for Legged Robots*](https://arxiv.org/abs/2107.04034) (2021) — RMA 两阶段框架；[通读归档](../../sources/papers/rma_reading_arxiv_2107_04034_pt3.md)
- Tobin et al., [*Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*](https://arxiv.org/abs/1703.06907) (2017) — 域随机化基础；[通读归档](../../sources/papers/domain_randomization_tobin_reading_arxiv_1703_06907.md)
- Hwangbo et al. (2019)、Margolis et al. (2022)、DreamWaQ++ (2026)、Extreme Parkour (2023)：见各自通读归档 [Hwangbo](../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md)、[Margolis](../../sources/papers/rapid_locomotion_rl_arxiv_2205_02824.md)、[DreamWaQ++](../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md)、[Extreme Parkour](../../sources/papers/extreme_parkour_arxiv_2309_14341.md)
- Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain* (Science Robotics, 2020) — 足式 sim2real 完整 pipeline（本次未作为通读原文列入）
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md) — DR / RMA / InEKF ingest 摘要
- [sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md](../../sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md) — 闭环六段与分层安全要点
