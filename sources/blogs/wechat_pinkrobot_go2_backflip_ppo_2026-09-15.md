# 开源 | GO2 Backflip：基于相位条件 PPO 的四足机器人后空翻训练与 Sim2Real 部署

> 来源归档（blog / 微信公众号）

- **标题：** 开源 | GO2 Backflip：基于相位条件 PPO 的四足机器人后空翻训练与 Sim2Real 部署
- **类型：** blog
- **作者：** PinkRobot（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/_BV8hQ7-DwXk2bgx_HnsCA
- **发表日期：** 2026-09-15（入库日）
- **入库日期：** 2026-09-15
- **抓取方式：** WebFetch
- **一句话说明：** 对 [Robot-Nav/GO2_backflip](https://github.com/Robot-Nav/GO2_backflip/tree/PPO-backflip) 的 **原理→代码** 长文：Isaac Lab 相位条件 PPO、60D 可部署 Actor / 165D 特权 Critic、电机包络与时延随机化、ONNX→MuJoCo→SDK2 真机状态机部署。

## 步骤 2.5（开源核查）

- **已开源：** <https://github.com/Robot-Nav/GO2_backflip/tree/PPO-backflip>（训练 `isaaclab_backflip/`、MuJoCo `mujoco/`、真机 `deploy_real/`、预训练 `model_*.onnx`）
- **演示：** B 站 BV1HbtJ6AENK
- **栈：** Isaac Sim 5.1 + Isaac Lab + RSL-RL + PyTorch → ONNX → MuJoCo Sim2Sim → Unitree SDK2

## 核心摘录（归纳）

### 技术路线

后空翻 ≈ 2 s 窗口内：站立→下蹲蓄力→起跳→空中负俯仰角速度→对齐→四足落地→抑制冲击恢复站姿。策略 **不依赖预录关节轨迹**，靠相位特征 + 奖励塑形学习时序。

### Actor 60D 可部署观测

| 块 | 维数 | 含义 |
|----|------|------|
| `root_ang_vel_b` | 3 | 机体系角速度 |
| `projected_gravity_b` | 3 | 重力投影 |
| `q-q_default` | 12 | 关节位置偏差 |
| `dq` | 12 | 关节速度 |
| `current_action` / `last_action` | 12+12 | 动作历史 |
| `phase_features` | 6 | 多频率 sin/cos 相位编码 |

### 相位条件化

借鉴 DeepMimic 周期动作思想但 **不用模仿奖励**；6 维谐波相位让同一姿态在不同阶段输出不同控制。

### Sim2Real 关键

- 非对称 Actor–Critic（RSS 2018 思路）
- 电机力矩–转速包络 `Go2HV`
- 动作/观测时延、传感器偏置与噪声
- 摩擦、质量、惯量、质心、电机强度等 DR
- 安全课程：逐步收紧关节/速度/接触约束
- 真机：50 Hz ONNX 策略 + 500 Hz LowCmd；RL 阶段直接发 torque（避免主机 PD + 固件 PD 双环）

### 真机状态机

`STARTUP → ARM → WAIT(phase=0) → [按 A] FLIP(0→2s) → RECOVERY → 可再次触发`；按 A **只重置 phase**，不重置真实状态与动作历史。

## 对 wiki 的映射

- **新建：** [repo-go2-backflip](../../wiki/entities/repo-go2-backflip.md)、[sources/repos/go2-backflip.md](../repos/go2-backflip.md)
- **交叉：** [DeepMimic](../../wiki/methods/deepmimic.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[Domain Randomization](../../wiki/concepts/domain-randomization.md)、[legbot-MPC-WBC](../../wiki/entities/legbot-mpc-wbc.md)（同 Robot-Nav 维护）
