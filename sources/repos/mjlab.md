# mjlab

> 来源归档

- **标题：** mjlab
- **类型：** repo
- **来源：** mujocolab（GitHub 组织）
- **链接：** https://github.com/mujocolab/mjlab
- **Stars / Forks：** ~2.2k / 344（2026-04）
- **入库日期：** 2026-04-29
- **论文：** Zakka et al. (2026) arXiv:2601.22074
- **一句话说明：** 将 Isaac Lab 的 manager-based API 与 MuJoCo Warp（GPU 加速物理）融合的轻量 RL 框架，是 AMP_mjlab 和 unitree_rl_mjlab 的底层依赖。
- **沉淀到 wiki：** 是 → [`wiki/entities/mjlab.md`](../../wiki/entities/mjlab.md)

---

## 核心定位

mjlab 解决的问题：Isaac Lab API 设计良好但绑定 NVIDIA Isaac Sim；MuJoCo Warp 物理精确但缺乏上层封装。mjlab 把两者接合：

- **上层**：复用 Isaac Lab 的 manager-based 环境设计 API（任务管理器、奖励管理器、观测管理器）
- **下层**：用 MuJoCo Warp 做 GPU 加速并行物理仿真
- **结果**：不依赖 Isaac Sim 的轻量、可组合 RL 训练框架

许可证：Apache 2.0，部分工具函数从 Isaac Lab 移植（BSD-3-Clause 头保留）。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 速度跟踪训练 | humanoid/四足机器人速度指令跟踪，支持 flat/rough terrain |
| 动作模仿训练 | 基于参考运动数据的 motion imitation |
| 策略评估工具 | 内置零动作/随机动作 dummy agent，快速 sanity check |
| 多 GPU 分布式训练 | 支持 multi-GPU scaling |
| Weights & Biases 集成 | 实验追踪 |

---

## 仓库结构

```
mjlab/
├── src/mjlab/       # 主包（环境、任务、管理器）
├── notebooks/       # Demo / 教程 notebook（含 Colab 链接）
├── scripts/         # 训练与评估脚本
├── tests/           # 测试套件
└── docs/            # 文档源码（mujocolab.github.io/mjlab）
```

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [isaac_gym_isaac_lab.md](isaac_gym_isaac_lab.md) | mjlab 借用了 Isaac Lab 的 API 设计，但用 MuJoCo Warp 替换底层仿真 |
| [mujoco.md](mujoco.md) | mjlab 基于 MuJoCo Warp（MuJoCo 的 GPU 加速版本） |
| [legged_gym.md](legged_gym.md) | 同为足式/人形 RL 训练框架，legged_gym 绑定 IsaacGym，mjlab 绑定 MuJoCo |
| [amp_mjlab.md](amp_mjlab.md) | AMP_mjlab 以 mjlab 为底层构建 G1 统一 AMP 策略 |
| [unitree_rl_mjlab.md](unitree_rl_mjlab.md) | Unitree 官方以 mjlab 为底层构建的官方 RL 训练框架 |
