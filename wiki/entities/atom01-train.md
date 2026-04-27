---
type: entity
tags: [humanoid, isaac-lab, rl, sim2sim, sim2real, roboparty]
status: complete
updated: 2026-04-27
related:
  - ./roboto-origin.md
  - ./robot-lab.md
  - ./isaac-gym-isaac-lab.md
sources:
  - ../../sources/repos/atom01_train.md
summary: "atom01_train 是 Atom01 的训练仓库，围绕 IsaacLab 训练配置、策略学习与仿真迁移流程构建。"
---

# Atom01 Train

**atom01_train** 是 Roboparty Atom01 项目的训练主仓库，聚焦 IsaacLab 场景下的策略学习、实验配置与迁移链路。

## 为什么重要

- 是 Atom01 从模型到策略的训练入口。
- 将硬件平台约束映射到训练环境，帮助减少部署落差。
- 与 `atom01_deploy` 联合构成训练→部署闭环。

## 核心结构/机制

- **训练配置**：任务参数、奖励项与训练超参数。
- **仿真迁移**：支持 Sim2Sim/Sim2Real 工作流。
- **工程接口**：与模型描述与部署链路对齐。


## 训练任务的输入与输出（面向 IsaacLab / PPO）

> 这一节用于回答「detail 子网页里 atom01_train 的输入输出到底是什么」。

### 输入（Observation / Command）

在 Atom01 这类人形 locomotion 训练里，策略网络通常接收以下输入块（按功能分组）：

- **本体状态（proprioception）**：
  - IMU 重力投影 / 角速度
  - 关节位置（相对默认姿态）
  - 关节速度
  - 上一时刻动作（action history）
- **高层命令（command）**：
  - 目标前进速度 `vx`
  - 侧向速度 `vy`
  - 偏航角速度 `yaw_rate`
- **可选历史堆叠（history stack）**：
  - 过去 `N` 帧观测，用于改善延迟与状态估计鲁棒性
- **可选特权信息（仅 teacher / 仿真）**：
  - 接触状态、地形信息、扰动参数等

### 输出（Action）

- 主流输出是 **每个可控关节的目标增量或目标角**（再经 PD 转力矩），例如：
  - `target_joint_pos = default_pos + action_scale * policy_output`
- 在部署链路里，动作会进一步经过：
  - 限幅（joint/action clipping）
  - 滤波（anti-jitter）
  - 与控制频率对齐（如 50Hz/100Hz）

## Reward 设计（建议从“稳定+跟踪+代价”三层开始）

为了便于在 detail 页快速查阅，这里把 reward 设计收敛成可执行模板：

1. **任务主目标（必须项）**
   - 线速度跟踪：`r_lin_vel`
   - 角速度跟踪：`r_ang_vel`
2. **稳定性项（人形关键）**
   - 躯干直立：`r_upright`
   - 基座高度：`r_base_height`
   - 脚接触节律/防滑：`r_contact` / `r_no_slip`
3. **代价与约束项（防止“钻奖励漏洞”）**
   - 能耗/力矩惩罚：`r_torque_penalty`
   - 动作变化率惩罚：`r_action_rate`
   - 关节越界惩罚：`r_joint_limit`

建议流程：

- **先训出不摔（简化 reward）**，再逐步加入平滑、能耗与步态细节。
- 若出现“原地抖动但不走”，优先检查：
  - 跟踪奖励是否过小
  - `action_rate` / `torque` 惩罚是否过重

## 训练代码配置细节（你应该优先看的参数）

围绕 `atom01_train`，可把训练配置分成 5 组：

1. **环境与任务**：任务名、episode 长度、并行环境数 `num_envs`
2. **观测与动作**：obs 归一化、history 长度、动作缩放与 clipping
3. **奖励与终止**：reward 权重表、跌倒阈值、姿态越界终止条件
4. **PPO 超参数**：learning rate、clip range、entropy coef、horizon、mini-batch
5. **域随机化（Sim2Real 前置）**：摩擦、质量、延迟、传感器噪声、推扰动

实操建议：

- 每次只改一组参数，避免多变量耦合导致无法归因。
- 训练日志至少跟踪：总 reward、各 reward 分量、跌倒率、速度跟踪误差。

## 开始训练与 Sim2Sim 指令（可直接放到 detail 页）

> 说明：本仓库当前已收录 `atom01_train` 的来源与定位，但没有把上游 README 的完整命令原文入库。为保证页面可用性，先给出 IsaacLab 生态常用命令模板（按实际脚本名替换）。

### 1) 开始训练（Train）

```bash
# 进入训练仓库后
python scripts/train.py --task atom01_walk --headless
```

常见可选参数：

```bash
python scripts/train.py   --task atom01_walk   --num_envs 4096   --max_iterations 5000   --seed 42   --headless
```

### 2) Sim2Sim（例如 Isaac → MuJoCo）

```bash
# 使用训练好的 checkpoint 在第二仿真器回放/评测
python scripts/sim2sim.py --task atom01_walk --checkpoint <ckpt_path>
```

常见可选参数：

```bash
python scripts/sim2sim.py   --task atom01_walk   --checkpoint logs/atom01_walk/model_XXXX.pt   --sim mujoco
```

如果上游仓库采用 `isaaclab.sh` 启动方式，可等价写成：

```bash
./isaaclab.sh -p scripts/train.py --task atom01_walk --headless
./isaaclab.sh -p scripts/sim2sim.py --task atom01_walk --checkpoint <ckpt_path>
```

> 建议在后续 ingest 中补全 `atom01_train` README 的原始命令段落，再把本节模板替换成“仓库实参版”。

## 常见误区或局限

- 误区：只要训练收敛就能真机稳定。真机仍受通信、校准、硬件误差影响。
- 局限：训练仓库往往强调算法迭代，部署鲁棒性要靠额外工程补齐。

## 参考来源

- [sources/repos/atom01_train.md](../../sources/repos/atom01_train.md)
- [Roboparty/atom01_train](https://github.com/Roboparty/atom01_train)

## 关联页面

- [Roboto Origin（开源人形机器人基线）](./roboto-origin.md)
- [robot_lab (IsaacLab 扩展框架)](./robot-lab.md)
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)

## 推荐继续阅读

- [Atom01 Deploy](./atom01-deploy.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
