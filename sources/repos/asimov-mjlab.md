# asimov-mjlab

> 来源归档

- **标题：** Asimov Locomotion（asimov-mjlab）
- **类型：** repo
- **来源：** Asimov Inc. / Menlo Research 相关组织（GitHub `asimovinc`）
- **链接：** https://github.com/asimovinc/asimov-mjlab
- **入库日期：** 2026-05-07
- **一句话说明：** 基于 mujocolab/mjlab 的 Asimov 双足行走训练 fork：12-DOF 腿模型、含 imitation 项的 PPO 奖励、面向真机的观测裁剪与 PD 参数化 Sim2Real 说明。
- **沉淀到 wiki：** 是 → 已并入 [`wiki/entities/asimov-v1.md`](../../wiki/entities/asimov-v1.md) 的「仿真与训练」叙述

---

## 为什么值得保留

- **与主仓分工清晰**：`asimov-v1` 主仓提供全栈 CAD/MuJoCo/板载软件；`asimov-mjlab` 提供 **GPU 并行**下的速度跟踪 + **参考步态模仿 shaping** 的可复现训练入口。
- **Sim2Real 取向明确**：README 写明去掉 `base_lin_vel`、给出 PD 增益与硬件上限的推导式叙述，便于与 [`wiki/entities/mjlab.md`](../../wiki/entities/mjlab.md) 生态对照。

## 与本仓库现有资料的关系

- 底层框架见 [`wiki/entities/mjlab.md`](../../wiki/entities/mjlab.md)（Isaac Lab 风格 API + MuJoCo Warp）。
- 人形硬件总览见 [`wiki/entities/asimov-v1.md`](../../wiki/entities/asimov-v1.md)。

## 官方延伸资源（外链）

- [asimovinc/asimov-mjlab README（main）](https://github.com/asimovinc/asimov-mjlab/blob/main/README.md)
