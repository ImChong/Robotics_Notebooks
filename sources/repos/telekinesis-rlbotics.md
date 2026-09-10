# Telekinesis RLbotics

> 来源归档

- **标题：** Telekinesis RLbotics
- **类型：** repo
- **来源：** Telekinesis GmbH（`telekinesis-ai` GitHub 组织）
- **链接：** https://github.com/telekinesis-ai/telekinesis-rlbotics
- **PyPI：** https://pypi.org/project/telekinesis-rlbotics/（v0.1.3，2026-08）
- **文档：** https://docs.telekinesis.ai/skills/rlbotics/overview.html
- **官网：** https://telekinesis.ai/
- **Stars：** ~33（2026-09-10）
- **入库日期：** 2026-09-10
- **一句话说明：** Telekinesis Agentic OS 下的轻量 GPU 加速 RL 技能库：单 YAML 描述整条训练管线，统一对接 Gymnasium / mjlab / Isaac Lab，训练后导出 ONNX，部署仅需 NumPy + onnxruntime。
- **代码：** https://github.com/telekinesis-ai/telekinesis-rlbotics（**已开源**，Apache 2.0）
- **沉淀到 wiki：** [telekinesis-rlbotics](../../wiki/entities/telekinesis-rlbotics.md)
- **交叉归档：** [telekinesis-docs-rlbotics.md](../sites/telekinesis-docs-rlbotics.md)

---

## 核心定位

RLbotics 是 **Telekinesis Agentic Skill Library** 中的强化学习模块，定位为跨仿真后端的 **统一 PPO 训练与 ONNX 部署层**，而非替代各仿真器自身的任务注册与资产管线。

| 维度 | 说明 |
|------|------|
| 仿真后端 | **Gymnasium**（无 GPU）、**mjlab**（MuJoCo Warp）、**Isaac Lab**（Omniverse） |
| 配置驱动 | 单 YAML：`env`（framework / task / num_envs / device）+ `runner`（网络、PPO、日志、checkpoint） |
| 训练入口 | `python examples/training_example.py configs/<framework>/<task>.yaml` |
| 部署产物 | 最佳 checkpoint → 自包含 `policy.onnx`；`Policy(path).get_action(obs)` |
| 可选依赖 | `[gym]` / `[mjlab]` / `[isaaclab]` extras；Python 3.10–3.12 |

---

## 仓库结构（摘录）

```
telekinesis-rlbotics/
├── configs/
│   ├── gymnasium/          # Ant, Humanoid, Pendulum 等经典控制
│   ├── mjlab/              # Unitree G1/Go1 velocity、Yam 操作等
│   └── isaaclab/           # Anymal、Unitree、Franka 等 Isaac 任务
├── examples/
│   ├── training_example.py # 训练 + ONNX 导出 + 部署烟测
│   └── configuration_example.py
├── src/telekinesis/rlbotics/
│   ├── envs/               # gym_env / mjlab_env / isaaclab_env
│   ├── runner.py           # OnPolicyRunner
│   ├── algorithms.py       # PPO
│   └── policy.py           # ONNX 推理封装
└── tests/
```

---

## 预置任务规模（README / configs，2026-08）

| 后端 | 示例配置 | 硬件要求 |
|------|----------|----------|
| Gymnasium | `Humanoid-v5.yaml` | 任意 OS，无 GPU |
| mjlab | `Mjlab-Velocity-Flat-Unitree-G1.yaml` | Linux/Windows + NVIDIA GPU |
| Isaac Lab | `Isaac-Velocity-Flat-Anymal-C-v0.yaml` | Linux/Windows + NVIDIA GPU，Python 3.11 |

CLI 覆盖：`--num-envs`、`--device`、`--num-learning-iterations`、`--log-dir`、`--resume [last|best|path]`。

---

## 对 wiki 的映射

- [telekinesis-rlbotics](../../wiki/entities/telekinesis-rlbotics.md) — 实体页
- 交叉：[mjlab](../../wiki/entities/mjlab.md)、[Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)、[unitree_rl_mjlab](../../wiki/entities/unitree-rl-mjlab.md)、[robot_lab](../../wiki/entities/robot-lab.md)
- 概念：[Sim2Real](../../wiki/concepts/sim2real.md)
