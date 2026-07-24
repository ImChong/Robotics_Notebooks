---
type: entity
tags: [repo, unitree, unitreerobotics, reinforcement-learning, isaac-gym, locomotion, sim2real]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-rl-lab.md
  - ./unitree-rl-mjlab.md
  - ./unitree-mujoco.md
  - ./legged-gym.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/repos/unitree_rl_gym.md
  - ../../sources/repos/unitree.md
summary: "unitree_rl_gym 是宇树官方基于 Isaac Gym + legged_gym 风格的 RL 训练仓（组织内星标最高），支持 Go2/H1/H1_2/G1；标准流程 Train→Play→Sim2Sim→Sim2Real，与 rl_lab / rl_mjlab 并行不可混拼观测。"
---

# unitree_rl_gym

**unitree_rl_gym** 是官方 **Isaac Gym** 路线的强化学习实现，任务与脚本组织接近社区 [legged_gym](./legged-gym.md)，覆盖 Unitree **Go2、H1、H1_2、G1**。

## 一句话定义

在 GPU 并行 Isaac Gym 环境里训练速度跟踪等 locomotion 策略，再经 Play / 外仿真 / 真机完成官方四段式落地。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真（本仓训练后端） |
| legged_gym | Legged Gym | ETH RSL 风格足式 RL 框架 |
| Sim2Sim | Simulation to Simulation | 换仿真器验证 |
| Sim2Real | Simulation to Real | 真机部署 |
| PPO | Proximal Policy Optimization | 常见策略优化算法（具体实现以上游为准） |

## 为什么重要

- **组织内星标最高**的官方 RL 入口，社区教程与复现材料最多。
- 明确写出 **Train → Play → Sim2Sim → Sim2Real**，便于对照其它两条官方线。
- 适合「先跑通一条完整链」；若团队已全面迁到 Isaac Lab 2.x / mjlab，应改选对应仓而非在本仓硬改。

## 核心原理

```text
Train (Isaac Gym) → Play (同环境可视化)
        → Sim2Sim (如 MuJoCo / 其它仿真)
        → Sim2Real (SDK2 / 真机)
```

| 阶段 | 含义 |
|------|------|
| Train | 多环境交互最大化回报；建议 headless |
| Play | 加载 checkpoint 目视/指标检查 |
| Sim2Sim | 导出网络到其它仿真，避免过拟合 Gym |
| Sim2Real | 物理机器人部署 |

训练产物默认：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`。

## 工程实践

```bash
python legged_gym/scripts/train.py --task=go2   # 或 g1 / h1 / h1_2
python legged_gym/scripts/play.py --task=go2
```

常用参数：`--headless`、`--resume`、`--num_envs`、`--max_iterations`、`--sim_device` / `--rl_device`。安装步骤见仓库 `doc/setup_en.md` / 中文文档。

**与另外两条官方线对照**：

| 仓 | 后端 | 何时选 |
|----|------|--------|
| **本仓** | Isaac Gym | 快速复现、社区资料多 |
| [unitree_rl_lab](./unitree-rl-lab.md) | Isaac Lab 2.x | 已在 Lab/Sim 5.x 生态 |
| [unitree_rl_mjlab](./unitree-rl-mjlab.md) | mjlab + MuJoCo Warp | 要官方 ONNX→C++ 闭环 |

## 局限与风险

- **Isaac Gym 已进入维护后期**，长期项目需评估迁 Lab / mjlab 的成本。
- **观测与动作定义与其它官方仓不通用**，禁止混拷配置。
- Play 默认加载「最近一次 run」——多人共享 logs 目录时易拿错模型。

## 关联页面

- [unitree_rl_lab](./unitree-rl-lab.md)
- [unitree_rl_mjlab](./unitree-rl-mjlab.md)
- [unitree_mujoco](./unitree-mujoco.md)
- [legged_gym](./legged-gym.md)
- [Locomotion](../tasks/locomotion.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/unitree_rl_gym.md](../../sources/repos/unitree_rl_gym.md)
- 上游：<https://github.com/unitreerobotics/unitree_rl_gym>

## 推荐继续阅读

- 仓库内 `doc/setup_en.md` 与中文 README

