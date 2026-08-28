# RSL-RL: A Learning Library for Robotics Research

> 来源归档（ingest）

- **标题：** RSL-RL: A Learning Library for Robotics Research
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2509.10771>
- **作者：** Clemens Schwarke、Mayank Mittal、Nikita Rudin、David Hoeller、Marco Hutter
- **机构：** ETH Zürich Robotic Systems Lab；NVIDIA
- **代码：** <https://github.com/leggedrobotics/rsl_rl>
- **入库日期：** 2026-08-28
- **一句话说明：** ETH RSL 把 GPU PPO 训练栈写成可引用论文；仓库同期提供 Student–Teacher Distillation 与可选 BF16 混合精度。

## 核心摘录（MVP）

### 1) 面向机器人研究的轻量 GPU RL 库

- **摘录要点：** 目标不是再做一个通用 RL 框架，而是让腿式 / 人形仿真研究者能在不改巨型库的前提下插算法；支持多 GPU，PyPI `rsl-rl-lib`。
- **对 wiki 的映射：**
  - [RSL-RL](../../wiki/entities/rsl-rl.md)
  - [Isaac Lab](../../wiki/entities/isaac-lab.md)

### 2) 算法面：PPO + 蒸馏

- **摘录要点：** 主算法 PPO（clip、GAE、KL 自适应学习率）；蒸馏把 teacher 特权策略压成 student。仓库另有 RND 与对称增广。
- **对 wiki 的映射：**
  - [PPO](../../wiki/methods/ppo.md)
  - [特权训练](../../wiki/concepts/privileged-training.md)

### 3) 工程：BF16 混合精度（仓库，2026）

- **摘录要点：** 论文本身不展开混合精度；截至 2026-08 的 main 在 PPO / Distillation 的 `update()` 中可选 bf16 autocast。PR #219 在 4090 上报告单次 `update()` **2.39×**、显存 **−33%**；端到端常被仿真采集绑死。
- **对 wiki 的映射：**
  - [RSL-RL](../../wiki/entities/rsl-rl.md) — BF16 实践表

## 当前提炼状态

- [x] 与官方仓 README / `ppo.py` / `distillation.py` 对齐
- [x] wiki 映射：`wiki/entities/rsl-rl.md`
