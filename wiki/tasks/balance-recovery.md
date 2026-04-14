---
type: task
tags: [locomotion, balance, stability, humanoid, bipedal, control]
status: complete
---

# Balance Recovery（平衡恢复）

**平衡恢复**（Balance Recovery / Push Recovery）：机器人在受到外部扰动（推力、地形突变、碰撞等）后，从失衡状态恢复到稳定姿态并继续任务的能力。

## 一句话定义

> 机器人被推了一把，能不摔倒还能走回来。

---

## 为什么重要

Locomotion 是动态平衡问题，真实环境中扰动不可避免：

- 人类推碰、物体碰撞
- 地面不平、意外台阶
- 负载突变（拿起重物）
- 关节执行误差累积

一个无法应对扰动的机器人，在真实环境中几乎无法部署。平衡恢复是从"在实验室走路"到"在真实世界工作"的关键能力。

---

## 核心理论工具

### Capture Point / DCM（Divergent Component of Motion）

**Capture Point**：机器人在某姿态下，必须将脚踩到该点才能在不摔倒的情况下停下来。

$$x_{CP} = x_{CoM} + \frac{\dot{x}_{CoM}}{\omega_0}, \quad \omega_0 = \sqrt{g / z_c}$$

其中 $\omega_0$ 是倒立摆的自然频率，$z_c$ 是质心高度。

**恢复判断**：
- 若 Capture Point 在支撑多边形内 → 可以通过原地站立恢复
- 若超出支撑多边形 → 必须迈步才能恢复（stepping recovery）

### N-step Capture Point

单步恢复不够时，可以规划 N 步使 Capture Point 落回支撑域：

$$x_{CP}^{(n)} = p_{step}^{(n)} + (x_{CP}^{(n-1)} - p_{step}^{(n)}) e^{-\omega_0 T_{step}}$$

N-step 规划形成了阶梯式平衡恢复策略：先定恢复步序列，再执行。

### 稳定裕度（Stability Margin）

衡量当前状态离失稳有多远的指标：

| 指标 | 定义 | 适用场景 |
|------|------|---------|
| CoP（Center of Pressure）裕度 | CoP 到支撑多边形边界的距离 | 静态/拟静态平衡 |
| ZMP 裕度 | ZMP 到支撑多边形边界的距离 | 动态行走 |
| Capture Point 裕度 | CP 到支撑域边界的距离 | 推力恢复预判 |
| DCM 误差 | 当前 DCM 与期望 DCM 的偏差 | 实时控制反馈 |

---

## 恢复策略分类

### 1. Ankle Strategy（踝关节策略）

**适用**：小扰动，重心轻微偏移。

通过踝关节的主动力矩调节 CoP 位置，将 ZMP 推回支撑域中心。不需要迈步，速度快。

**局限**：踝关节力矩有限，超过范围即失效。

### 2. Hip Strategy（髋关节策略）

**适用**：中等扰动。

通过髋关节反向运动（类似人低头弯腰）调整全身质量分布。常与踝关节策略联合使用。

### 3. Stepping Strategy（迈步策略）

**适用**：大扰动，CP 已超出支撑域。

主动迈步到新位置使 Capture Point 回到新支撑域内。是最通用的恢复策略，但需要快速步位规划。

**关键约束**：
- 步位约束（腿的可达范围）
- 步时约束（最短迈步时间）
- 地形约束（落脚点需要可行）

### 4. Multi-Contact Strategy（多接触策略）

**适用**：极端扰动、接近倒地时。

主动利用环境接触（扶墙、蹲低、扶地）避免摔倒。常见于人类本能反应，机器人实现较难。

---

## 典型方法路线

### 经典控制方法

**基于 Capture Point 的在线步位规划**（Koolen et al., Englsberger et al.）：
1. 实时计算当前 Capture Point
2. 判断是否需要迈步及迈步位置
3. 用 LIPM + MPC 生成恢复轨迹
4. WBC 执行

**IHMC 推力恢复**：
- 多策略切换：Ankle → Hip → Stepping → Multi-Contact
- 根据稳定裕度实时选择策略
- 已在 Atlas、Valkyrie 等机器人上验证

### RL 方法

近年来 RL 在平衡恢复上取得了强结果，尤其是在不规则扰动和复杂地形上：

**端到端 RL**（如 ANYmal 系列）：
- 在仿真中以随机推力作为扰动进行域随机化训练
- 策略直接输出关节力矩
- 优点：无需手工设计策略切换逻辑
- 缺点：可解释性差，边界行为难以预测

**RL + 经典控制融合**：
- 用 RL 学习高层稳定策略（是否迈步、迈到哪里）
- 用 TSID/WBC 执行底层动作
- 结合两者优点

---

## Sim2Real 挑战

平衡恢复的 sim2real 难度较高：

- **推力时序**：仿真中的瞬时冲击和真实硬件响应不一致
- **接触模型**：地面刚度、鞋底摩擦在仿真中难以精确模拟
- **延迟**：控制延迟在高扰动恢复中影响显著
- **状态估计噪声**：剧烈扰动下 IMU 和状态估计会产生较大误差

常用缓解措施：
- 域随机化（推力大小/方向/时序 + 质量参数）
- 加入状态估计噪声
- 延迟随机化

---

## 评估指标

| 指标 | 说明 |
|------|------|
| 最大可恢复推力 | 在保证不摔倒的前提下可承受的最大推力 |
| 恢复时间 | 从扰动到重新稳定行走的耗时 |
| 迈步次数 | 恢复过程中需要的额外步数（越少越好） |
| 成功率 @ N kg·m/s 冲量 | 给定冲量大小下的恢复成功比例 |

---

## 参考来源

- Pratt et al., *Capture Point: A Step toward Humanoid Push Recovery* (2006) — Capture Point 理论基础
- Englsberger et al., *Three-Dimensional Bipedal Walking Control Based on Divergent Component of Motion* (2015) — DCM 三维扩展与步行控制
- Koolen et al., *Capturability-based Analysis and Control of Legged Locomotion* (2012) — N-step capturability 理论框架
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021) — RL 结合域随机化应对扰动
- Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain* (Science Robotics, 2020) — 足式机器人扰动恢复
- **ingest 档案：** [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md)
- **ingest 档案：** [sources/papers/state_estimation.md](../../sources/papers/state_estimation.md) — Nobili 2017 本体感知状态估计（平衡恢复的感知基础）

---

## 关联页面

- [Capture Point / DCM](../concepts/capture-point-dcm.md) — 平衡恢复的核心理论工具，步位规划的依据
- [Locomotion](./locomotion.md) — 平衡恢复是高鲁棒 locomotion 的子问题
- [Whole-Body Control](../concepts/whole-body-control.md) — WBC 执行恢复策略输出的关节指令
- [TSID](../concepts/tsid.md) — 经典恢复策略的底层执行器
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md) — 在线步位规划常用 MPC 求解
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 方法直接从仿真中学习鲁棒恢复策略

## 一句话记忆

> 平衡恢复是让机器人在被推之后不摔倒还能继续走的能力——从 Capture Point 的几何直觉，到踝-髋-迈步的策略层次，再到现代 RL 的端到端学习，都是在解决"扰动后稳回来"这一个问题。
