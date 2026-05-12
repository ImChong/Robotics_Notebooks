# mjlab_playground

> 来源归档

- **标题：** mjlab_playground
- **类型：** repo
- **来源：** mujocolab（GitHub 组织）
- **链接：** https://github.com/mujocolab/mjlab_playground
- **入库日期：** 2026-05-12
- **许可证：** Apache-2.0
- **一句话说明：** 基于 [mjlab](mjlab.md) 的任务集合，首批从 [MuJoCo Playground](https://playground.mujoco.org/) 移植，提供可脚本化训练与回放（`uv run train` / `uv run play`）的足式机器人技能示例。
- **沉淀到 wiki：** 是 → [`wiki/entities/mjlab-playground.md`](../../wiki/entities/mjlab-playground.md)

---

## 核心定位

官方 README 将其描述为「用 mjlab 搭建的任务集合」，并与 MuJoCo Playground 的端口工作关联：在 mjlab 的统一 RL API 上复现/扩展 playground 中的代表性任务。

---

## 当前公开任务（README 表格摘要）

| Task ID | 机器人 | 描述 |
|---------|--------|------|
| `Mjlab-Getup-Flat-Unitree-Go1` | Unitree Go1 | 平地摔倒恢复（fall recovery） |
| `Mjlab-Getup-Flat-Booster-T1` | Booster T1 | 平地摔倒恢复 |

README 还提到：在单卡 NVIDIA 5090 上 Go1 getup 约 2 分钟量级收敛、T1 约 8 分钟量级收敛，并采用逐步收紧 action rate、关节速度与功率惩罚的课程式训练以得到更平滑、更安全的策略（叙述来自仓库 README，实际耗时随硬件与超参变化）。

---

## 使用方式（来自 README）

```bash
git clone https://github.com/mujocolab/mjlab_playground.git && cd mjlab_playground
uv sync
uv run train <task-id> --num_envs 4096
uv run play <task-id>
```

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [mjlab.md](mjlab.md) | 底层 RL 与并行仿真框架 |
| [amp_mjlab.md](amp_mjlab.md) / [unitree_rl_mjlab.md](unitree_rl_mjlab.md) | 同生态的 Unitree 侧重仓库；playground 更偏「任务示例与 playground 端口」 |
| [mujoco.md](mujoco.md) | 物理与 Playground 上游站点所依赖的 MuJoCo 生态 |
