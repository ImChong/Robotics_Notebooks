---
type: query
tags: [locomotion, rl, sim2real, humanoid, cookbook, training]
status: stable
---

# 人形机器人 RL 策略训练完整 Checklist

> **Query 产物**：本页由以下问题触发：「从零开始训练一个能在真机上走路的人形机器人 RL 策略，完整 checklist？」
> 综合来源：[sim2real-checklist.md](./sim2real-checklist.md)、[rl-algorithm-selection.md](./rl-algorithm-selection.md)、[locomotion-reward-design-guide.md](./locomotion-reward-design-guide.md)、[wiki/concepts/privileged-training.md](../concepts/privileged-training.md)

---

## TL;DR 决策路径

```
目标：真机行走策略

1. 硬件确定？
   → G1/H1（双足）      → 参考本页 Stage 1-6
   → Unitree A1/Go1    → 参考 legged_gym 默认配置，直接 Stage 3 起步

2. 仿真环境？
   → Isaac Lab（推荐）  → GPU 并行训练，数万环境
   → MuJoCo + legged_gym → 入门友好，10分钟出策略

3. 训练策略？
   → 从零       → PPO + 简单 reward → Stage 3 开始
   → 有参考动作  → 先 motion imitation（AMP/PhysHOI），再 RL 微调
```

---

## Stage 1：仿真资产准备

- [ ] URDF/MJCF 获取：来源于官方 SDK 或 URDF-to-MJCF 转换
- [ ] 关节参数验证：
  - PD 增益（kp/kd）对应真实执行器带宽
  - 关节限位对齐硬件手册（尤其脚踝/髋关节）
  - 质量/惯性参数：运行 RNEA + 静态平衡测试验证
- [ ] 接触几何：脚底碰撞形状用简单盒/球代替 mesh（提速 10x+）
- [ ] 执行器模型：是否需要 ActuatorNet？（关节误差历史 → 实际力矩）
  - 简化版：PD 控制 + 力矩限制
  - 精确版：Hwangbo 2019 ActuatorNet（数据驱动，推荐真机差距大时使用）

---

## Stage 2：Observation 设计

**最小可用 observation**（推荐入门）：

| 变量 | 维度 | 备注 |
|------|------|------|
| 基座角速度（IMU） | 3 | 滤波后的陀螺仪输出 |
| 重力向量（projected） | 3 | IMU 计算，关键稳定性信号 |
| 速度命令 | 3 | vx / vy / yaw_rate |
| 关节位置（偏移） | n_joints | 相对默认站立姿态的偏差 |
| 关节速度 | n_joints | 低通滤波 |
| 上一步动作 | n_joints | 平滑动作历史 |

**特权信息 observation**（teacher policy，仅仿真）：
- 地形高度图（3x3 或 11x11 点阵）
- 接触状态（binary per foot）
- 基座线速度（仿真可直接读，真机需估计）
- 地形摩擦系数

---

## Stage 3：Reward 设计

参考 [locomotion-reward-design-guide.md](./locomotion-reward-design-guide.md)，以下是双足人形的关键调整：

**人形特有奖励（vs 四足）：**

```python
# 1. 直立奖励（四足不需要，人形关键）
r_upright = exp(-5 * ||gravity_proj_xy||²)  # 重力在水平面的投影接近0

# 2. 手臂对称摆动（单纯步行可以暂时固定手臂）
r_arm_swing = -0.1 * ||arm_joints - default_arm_pose||²

# 3. 脚踝对齐（脚掌平行于地面）
r_foot_orient = exp(-10 * ||foot_orientation_error||²)

# 4. 避免过大髋部偏移（人形侧倾不稳定）
r_base_height = exp(-2 * (base_height - 0.82)²)  # 目标高度根据机型调整
```

**Reward 调试顺序**：
1. 先只保留速度追踪 + 存活奖励，确认策略能学会不摔倒
2. 加入脚掌接触奖励，诱导出步态节律
3. 加入直立/高度奖励，改善姿态
4. 最后加入能耗/平滑惩罚

---

## Stage 4：训练配置

| 超参数 | 推荐值 | 备注 |
|--------|--------|------|
| 算法 | PPO | 首选；SAC 可用但调参更难 |
| 并行环境数 | 4096（GPU）/ 512（CPU） | Isaac Lab 默认 4096 |
| 最大 episode 长度 | 1000 步（20s @ 50Hz） | 太短策略不稳，太长难优化 |
| 域随机化强度 | 从弱开始，逐步加强 | 过早强随机化导致收敛困难 |
| 观测噪声 | 加入（模拟真实传感器） | IMU 噪声、关节编码器分辨率 |
| clip_param | 0.2 | PPO 标准 |
| 学习率 | 1e-4 ~ 3e-4 | 自适应调度 |

**Curriculum 策略**（参考 [curriculum-learning.md](../concepts/curriculum-learning.md)）：
- 速度命令从 0.5 m/s 开始，成功率 > 80% 后提升到 1.5 m/s
- 地形从平地开始，逐步引入随机地形

---

## Stage 5：Teacher-Student 蒸馏

当基础策略收敛后（Isaac Lab 通常 2000 iteration），进入 sim2real 阶段：

```
Teacher Policy（特权 obs） → 训练完成
        ↓
Adaptation Module（学生）：
  - 输入：真实 obs 历史（50 步）
  - 目标：模仿 Teacher 的隐状态 / 直接模仿动作
  - 方法：行为克隆（MSE loss on actions or latent）
  - 数据：从 Teacher 策略采样的仿真数据（不需要真机数据）
```

**两种蒸馏方案选择**：
- **RMA 风格**：训练环境编码器 → Adaptation Module 模仿编码器输出（推荐）
- **直接行为克隆**：Adaptation Module 直接模仿 Teacher 动作（更简单，效果略差）

---

## Stage 6：真机部署 Checklist

- [ ] 策略推理频率对齐控制频率（50Hz / 100Hz）
- [ ] observation 归一化参数与训练时一致（mean/std）
- [ ] 动作 clip：防止极端关节命令（通常 ±0.5 rad 或力矩限制）
- [ ] 安全停止机制：基座倾斜 > 阈值时立即切换为安全控制
- [ ] 部署前在仿真中用"真实噪声模拟"验证鲁棒性
- [ ] 真机第一次运行：人工扶住 + 低速命令 + 随时急停

---

## 参考来源

- [sources/papers/privileged_training.md](../../sources/papers/privileged_training.md) — ingest 档案（Kumar RMA 2021 / Lee Science Robotics 2020）
- [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md) — ingest 档案（PPO / Rudin 2022）
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md) — ingest 档案（sim2real 核心论文）

---

## 关联页面

- [Sim2Real Checklist](./sim2real-checklist.md) — 详细的域随机化配置清单
- [RL Algorithm Selection](./rl-algorithm-selection.md) — PPO vs SAC vs TD3 选型指南
- [Locomotion Reward Design Guide](./locomotion-reward-design-guide.md) — 奖励函数详细设计
- [Privileged Training](../concepts/privileged-training.md) — Teacher-Student 蒸馏理论
- [Curriculum Learning](../concepts/curriculum-learning.md) — 课程式训练加速收敛

---

## 一句话记忆

> 人形 RL 训练的核心链路：仿真资产验证 → 最小 obs 设计 → 分阶段 reward → PPO 训练 + curriculum → teacher-student 蒸馏 → 真机验证，每个阶段有明确验收标准。
