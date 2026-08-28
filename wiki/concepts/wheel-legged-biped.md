---
type: concept
tags: [wheel-legged, biped, locomotion, hybrid, two-wheel, tita, flamingo]
status: complete
updated: 2026-08-28
related:
  - ./wheel-legged-quadruped.md
  - ../tasks/hybrid-locomotion.md
  - ../tasks/locomotion.md
  - ../entities/tita-rl.md
  - ../entities/ddt-lab.md
  - ../entities/wheel-legged-genesis.md
  - ../entities/isaac-rl-two-wheel-legged-bot.md
  - ../entities/stackforce.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/tita_rl.md
  - ../../sources/repos/wheel_legged_genesis.md
  - ../../sources/repos/isaac_rl_two_wheel_legged_bot.md
  - ../../sources/repos/ddt_lab.md
summary: "轮腿双足（双轮足）是两条腿末端各带驱动轮的倒立摆式混合底盘：平地靠轮式滚动与俯仰平衡，越障靠腿长/髋摆调节；与四轮足（Go2W 类）不是同一形态。"
---

# 轮腿双足机器人（双轮足 / Two-Wheel-Legged Biped）

## 一句话定义

轮腿双足在 **两条腿末端各装一只驱动轮**，整机像可调腿长的两轮倒立摆：平地用轮速保持平衡与巡航，越障/下蹲用髋–大腿–小腿改变轮距与质心高度；典型量产如 Direct Drive Tech **TITA**，开源训练仓覆盖 Isaac Gym、Isaac Lab 与 Genesis。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TITA | Direct Drive Tech TITA | 直驱科技轮腿双足整机；官方 RL 分 Isaac Gym（`tita_rl`）与 Isaac Lab（`DDT_Lab`）两代 |
| NP3O | Constrained PPO + 代价项（仓内算法名） | TITA 官方栈用的约束 PPO，带关节/力矩等 cost |
| CaT | Constraints as Termination | 把约束违反映射成随机终止信号的 RL 技巧（Flamingo 仓实现） |
| RSL-RL | Robotic Systems Lab RL | Genesis / Flamingo 常用的 GPU PPO 训练核 |
| Sim2Real | Simulation to Real | 仿真策略迁真机；双轮足还要过轮–地接触与俯仰平衡 |

## 为什么重要

- **能效介于轮式与双足之间**：平坦路面不必高频踏步，比纯双足省；比差速小车多一截主动调高与越障。
- **控制形态独特**：本质是 **欠驱动平衡 + 轮速跟踪 + 可变腿长**，观测/奖励不能直接套四足或四轮足模板。
- **开源栈已经分代**：同一机体（TITA）同时存在 **Isaac Gym 官方仓** 和 **Isaac Lab 官方仓**；社区另有 Genesis 与 Flamingo Lab 扩展。选型先认仿真世代，再认算法。

## 核心原理

1. **运动学**：左右对称开链（髋 / 大腿 / 小腿）+ 轮关节连续旋转。轮提供切向力，腿关节改变轮心高度与左右轮距。
2. **平衡**：俯仰角接近倒立摆；奖励里几乎都会重罚大 pitch/roll 与基座触地（TITA 对 `base` 接触直接 terminate）。
3. **指令**：常见是 \(v_x, v_y, \omega_{\mathrm{yaw}}\) 再加腿长或机体高度；部分仓把 TrackZ / 跳跃 / 后空翻拆成独立 Gym 任务。
4. **与四轮足的边界**：四轮足（Go2W / D1 / M20）是 **四条腿 + 足端轮**，可全向滚动或抬腿踏步；本页是 **两轮 + 两条腿**。不要把 [轮足四足](./wheel-legged-quadruped.md) 的资产或 reward 直接搬过来。

```mermaid
flowchart LR
  cmd["速度 / 航向 / 腿长指令"] --> pol["RL 策略"]
  pol --> pd["关节 PD + 轮速"]
  pd --> contact["轮–地接触"]
  contact --> imu["IMU 俯仰 / 轮速"]
  imu --> pol
```

## 工程实践

| 栈 | 仿真 | 机体 | 训练入口 | 下游 |
|----|------|------|----------|------|
| [tita_rl](../entities/tita-rl.md) | Isaac Gym | TITA | `train.py --task=tita_constraint` | ONNX → TensorRT → [Webots / 真机仓](../../sources/repos/tita_rl_sim2sim2real.md) |
| [DDT_Lab](../entities/ddt-lab.md) | Isaac Lab | Tita + D1 | `DDT-Velocity-{Flat,Rough}-Tita-v0` | JIT / ONNX |
| [wheel_legged_genesis](../entities/wheel-legged-genesis.md) | Genesis | CJ-003 | `locomotion/wheel_legged_train.py` | MuJoCo `gs2mj` |
| [lab.flamingo](../entities/isaac-rl-two-wheel-legged-bot.md) | Isaac Lab 2.0 | Flamingo | `scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-v1-ppo` | 宣称零样本；sim2sim 分支迁移中 |

桌面级教育向轮足（非本页工业/研究机）见 [StackForce 小轮足](../entities/stackforce.md)。

## 局限与风险

- **形态名易混**：中文「轮足」常同时指四轮足与双轮足；本库用 **轮腿双足 / 双轮足** 专指两轮构型。
- **仿真世代不互通**：Gym 的 `tita_constraint` 观测与 Lab 的 `DDT-Velocity-*-Tita-v0` 不能互换 checkpoint。
- **接触模型敏感**：轮胎摩擦、轮半径、动作滞后（Genesis 默认 1 step latency）比纯足式更容易把 sim2real 拉开。
- **开源深度不一**：TITA 有官方真机 bringup；Genesis 仓停在 MuJoCo sim2sim；Flamingo README 的零样本需对照具体硬件修订。

## 关联页面

- [轮足四足机器人（四轮足）](./wheel-legged-quadruped.md) — 四条腿 + 足端轮，不要与本页混淆
- [Hybrid Locomotion](../tasks/hybrid-locomotion.md)
- [Locomotion](../tasks/locomotion.md)
- [tita_rl](../entities/tita-rl.md)
- [DDT_Lab](../entities/ddt-lab.md)
- [wheel_legged_genesis](../entities/wheel-legged-genesis.md)
- [Isaac-RL-Two-wheel-Legged-Bot](../entities/isaac-rl-two-wheel-legged-bot.md)
- [Sim2Real](./sim2real.md)

## 参考来源

- [tita_rl](../../sources/repos/tita_rl.md)
- [wheel_legged_genesis](../../sources/repos/wheel_legged_genesis.md)
- [Isaac-RL-Two-wheel-Legged-Bot](../../sources/repos/isaac_rl_two_wheel_legged_bot.md)
- [DDT_Lab](../../sources/repos/ddt_lab.md)

## 推荐继续阅读

- Direct Drive Tech TITA 官方 Gym 仓：<https://github.com/DDTRobot/tita_rl>
- NP3O 上游（Go2）：<https://github.com/zeonsunlightyu/LocomotionWithNP3O>
- Constraints as Termination（arXiv:2403.18765）：<https://arxiv.org/abs/2403.18765>
