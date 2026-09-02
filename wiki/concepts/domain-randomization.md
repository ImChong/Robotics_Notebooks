---
type: concept
summary: "Domain Randomization 通过在仿真中随机化参数扩大训练分布，是 sim2real 的主流鲁棒化手段。"
updated: 2026-09-02
---

# Domain Randomization

**域随机化**：在仿真训练中主动随机化物理参数、视觉纹理、环境设置，让策略被迫学会适应各种变化的泛化能力，从而实现零样本从仿真迁移到现实。

## 一句话定义

把仿真环境“搞乱”，强制机器人在各种随机化的环境变体中训练，这样到了真实环境反而能应对。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DR | Domain Randomization | 训练时随机化仿真参数以提升鲁棒迁移 |
| Sim2Real | Simulation to Real | DR 是缩小 domain gap 的核心手段之一 |
| RL | Reinforcement Learning | DR 多在仿真 RL 训练阶段施加 |
| SysID | System Identification | 与 DR 互补：缩小参数而非覆盖分布 |
| URDF | Unified Robot Description Format | 被随机化的质量/摩擦等常来自模型参数 |

## 为什么重要

Sim2Real 的核心问题是 **domain gap**——仿真和现实的差异。

传统做法想把这个 gap 越小越好（精确建模）。

但 Domain Randomization 的思路相反：

> **不追求仿真逼真，而追求策略见过足够多样的仿真变体，从而对任何仿真-现实差异都不怕。**

## 核心思想

训练时，随机化以下维度：

- **物理参数**：质量、摩擦力、关节阻尼、电机延迟等
- **观测噪声**：传感器噪声、分辨率、光照
- **动作延迟**：执行延迟、控制频率
- **视觉纹理**：背景、物体颜色、光照方向
- **任务参数**：目标位置、障碍物形状

这样策略学到的不是“在某个固定环境里怎么做”，而是“在任何环境里怎么做”。

## 主要类型

### 1. 物理参数随机化
最经典的形式。

随机化：
- 机器人质量、重心位置
- 各关节摩擦力、阻尼
- 电机响应延迟
- 地面摩擦系数

代表工作：Tobio et al. 2018 "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"

### 2. 视觉/纹理随机化
用于让视觉策略泛化到不同光照、背景、遮挡。

方法：
- 随机化背景图片
- 随机化光照方向、强度
- 随机化物体的纹理和颜色

代表：PixelCNN 背景随机化，Domain Randomization for Vision-based RL

### 3. 观测随机化
对传感器观测加噪声。

- IMU 偏置噪声
- 编码器量化误差
- 触觉传感器噪声

### 4. 任务随机化
任务参数本身随机：

- 目标位置/朝向
- 障碍物配置
- 初始状态分布

### 5. 自动域随机化（AutoDR）
手动调随机化范围很难。

自动域随机化：用策略在真实环境中的表现来自动调整随机化范围——在真实环境表现差就增大对应维度的随机化。

代表：AutoDR (Peng et al.)

## 关键参数：随机化范围

随机化范围太大 → 任务太难，训练不出来。  
随机化范围太小 → 过拟合特定仿真，泛化不够。

这是 domain randomization 最核心的调参问题。

常见做法：

- 从小范围开始，逐步增大
- 关键维度重点随机化（通常是摩擦力、延迟）
- 用真实机器人数据做 curriculum

## 在人形机器人中的典型应用

人形机器人 sim2real 的 DR 特别关注：

- 地面摩擦系数（最重要）
- 电机控制延迟（响应速度）
- 足底接触力学（最不稳定）
- 观测中的噪声和偏置

典型 pipeline：

```
固定仿真参数训练 → 加入 DR → 逐步增大随机化范围 → 零样本迁移到真实机器人
```

## 常见误区

1. **随机化范围越大越好**：不一定，范围太大策略训练不出来；应先 [SysID](./system-identification.md) 定基准，再围绕公差扩 DR（见 [闭环误差分层](../queries/sim2real-closed-loop-engineering.md)）
2. **所有维度均匀随机化**：应该优先随机化对任务影响最大的维度
3. **只看仿真指标**：最终目标是真实机器人表现，仿真 reward 高不等于迁移成功
4. **忽略动作延迟**：人形机器人动作延迟对稳定性影响很大，是 DR 里的高频坑
5. **一失败就盲目扩大 DR**：先分解可建模参数 / 难建模动态 / 观测误差，再决定校准还是随机化

## 和其他方法的关系

- **vs System Identification**：SysID 追求精确建模，DR 追求鲁棒泛化。可以互补。
- **vs Domain Adaptation**：DA 是在特征空间对齐分布，DR 是在数据分布层面增加多样性
- **和 privileged information 结合**：用更丰富的状态信息训练，推理时只用可观测信息
- **轮足高动态案例：** [AWARE](../entities/paper-aware-wheeled-legged-reflexive-evasion.md) 用质量/惯量/摩擦/执行器增益/外扰等 TABLE II 式 DR 支撑 M20 真机反射规避；真机 ASR 仍显著低于仿真，说明极限机动下 DR 必要但常不够。
- **解析控制不是 DR 的免费乘客：** [RL vs GC](../entities/paper-rl-vs-gc.md) 把同一质量/惯量/推重比随机化同时施加给 PPO 与 \(SE(3)\) 几何控制——RL 几乎保住跟踪奖励，GC 因可调参数少、模型写死而掉得更明显。跨类对比时不要只给学习侧做 DR。

## 参考来源

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* (2017) — DR 概念提出
- OpenAI, *Learning Dexterous In-Hand Manipulation* (2019) — DR 大规模应用于操作任务
- Hwangbo et al., *Learning Agile and Dynamic Motor Skills for Legged Robots* (2019) — 执行器网络与 actuator DR
- **ingest 档案：** [sources/papers/simulation_tools.md](../../sources/papers/simulation_tools.md) — Genesis/Isaac Gym 仿真平台（DR 的执行环境）
- **ingest 档案：** [sources/papers/barkour_arxiv_2305_14654.md](../../sources/papers/barkour_arxiv_2305_14654.md) — Barkour 在 Rudin 默认 DR 之外，对 >1m/s 敏捷动作补充躯干惯量 / 电机建模 / 关节静摩擦随机化（Table II）
- **ingest 档案：** [sources/papers/aware_arxiv_2604_23761.md](../../sources/papers/aware_arxiv_2604_23761.md) — AWARE 轮足反射避障 DR 表与真机 ASR 落差
- **ingest 档案：** [sources/papers/leveling_playing_field_rl_vs_gc_arxiv_2506_17832.md](../../sources/papers/leveling_playing_field_rl_vs_gc_arxiv_2506_17832.md) — 四旋翼上 DR 对 RL vs 几何控制不对称
- **ingest 档案：** [sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md](../../sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md) — DR 应围绕 SysID 基准，忌盲目扩范围
- **ingest 档案：** [Getting Started With Isaac Lab](../../sources/courses/nvidia_getting_started_isaac_lab.md) — 模块 4：物理/视觉/深度/点云 DR 与「通才 vs 专才」权衡

## 关联页面

- [Sim2Real](./sim2real.md)
- [Sim2Real 闭环误差分层工程](../queries/sim2real-closed-loop-engineering.md) — 误差分流：可建模→校准，难建模→DR
- [Procedural Terrain Generation](./procedural-terrain-generation.md) — DR 的地形载体（坡、台阶、碎石）
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Whole-Body Control](./whole-body-control.md)
- [Locomotion](../tasks/locomotion.md)
- [RL 运动控制完整管线](../overview/robot-rl-motion-control-pipeline.md) — DR 作为腿式 Sim2Real 积木之一
- [AWARE（轮足高动态反射避障）](../entities/paper-aware-wheeled-legged-reflexive-evasion.md)
- [RL vs GC](../entities/paper-rl-vs-gc.md) — 同一 DR 预算下 PPO 比几何控制更稳
- [RL vs 几何控制](../comparisons/rl-vs-geometric-control.md)
- [NVIDIA Getting Started With Isaac Lab](../entities/nvidia-getting-started-isaac-lab.md) — 官方课把视觉/深度/点云 DR 放进「仿真增强」一类，并警告 DR 过宽会变成弱通才
- [tita_rl](../entities/tita-rl.md) / [wheel_legged_genesis](../entities/wheel-legged-genesis.md) — 双轮足训练里的摩擦/质量/滞后随机化实例

## 推荐继续阅读

- Tobio et al. 2018, "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
- [SAGE（执行器 Sim2Real 间隙估计）](../entities/sage-sim2real-actuator-gap-estimator.md)（[上游仓库](https://github.com/isaac-sim2real/sage)）
- [Deployment-Ready RL](https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/)
