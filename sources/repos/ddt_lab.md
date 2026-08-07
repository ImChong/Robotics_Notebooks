# DDT_Lab

> 来源归档

- **标题：** DDT_Lab（ddt_lab）
- **类型：** repo
- **来源：** DDTRobot（直驱科技 Direct Drive Tech 官方 GitHub 组织）
- **链接：** https://github.com/DDTRobot/DDT_Lab
- **星标（截至 2026-08-07）：** ~10
- **最近推送：** 2026-08-03
- **主要语言：** Python
- **许可证：** 仓库 API 未返回 SPDX（以源码头 / README 为准）
- **分类：** 强化学习训练 / Isaac Lab 厂商扩展 / 轮足
- **入库日期：** 2026-08-07
- **一句话说明：** 直驱科技基于 Isaac Lab 的 NP3O 轮足 locomotion 训练仓，覆盖 D1（四轮足）与 Tita（轮腿双足），导出 JIT/ONNX 策略。
- **沉淀到 wiki：** 是 → [`wiki/entities/ddt-lab.md`](../../wiki/entities/ddt-lab.md)
- **机构：** 直驱科技（Direct Drive Tech）→ `direct-drive-tech` / `ddt`

---

## README 要点（编译自上游）

- 标题叙事：**NP3O Locomotion for Wheel-Legged Robots**；由 Isaac Lab 项目模板派生。
- 依赖：Isaac Sim **5.1**、Isaac Lab **v2.3.0**、Python 3.11、CUDA 12.x。
- 机型与任务（`python scripts/list_envs.py` 应列出 8 个 `DDT-*`）：

| 机器人 | 形态 | Flat | Rough |
|--------|------|------|-------|
| **D1** | 四足 + 足端轮 | `DDT-Velocity-Flat-D1-v0` | `DDT-Velocity-Rough-D1-v0` |
| **Tita** | 轮腿双足 | `DDT-Velocity-Flat-Tita-v0` | `DDT-Velocity-Rough-Tita-v0` |

  另有 `*-Play-v0` 可视化变体（少环境、零指令、无域随机）。

- 资产：`DDT_MODEL_DIR` 默认指向仓内 `ddt_ros2_control/urdfs/`；需另 clone [`DDTRobot/ddt_ros2_control`](https://github.com/DDTRobot/ddt_ros2_control)。
- 安装：`python -m pip install -e source/ddt_lab`。
- 训练 / 评估：`scripts/np3o/train.py`、`scripts/np3o/play.py`；play 支持 `--export_policy` 导出 JIT + ONNX（双输入：当前本体观测 + 历史缓冲）。
- **NP3O** 相对 PPO 的扩展：BarlowTwins SSL 历史编码器（隐式速度估计）、Lagrangian 约束代价项、特权 Critic。
- README 中 `git clone https://github.com/DDTRobot/DDT_Lab/tree/np3o` 写法易误导——应为仓库 URL（默认分支 `main`；另有 `dev` / `feature/platform`）；入库以 `main` README 为准。

## 开源状态

- **已开源**：公开 GitHub 仓库（DDTRobot/DDT_Lab）。
- **资产依赖**：URDF 在配套仓 `ddt_ros2_control`（同组织，需单独 clone）。

## 对 wiki 的映射

- 实体页：[`wiki/entities/ddt-lab.md`](../../wiki/entities/ddt-lab.md)
- 轮足概念：[`wiki/concepts/wheel-legged-quadruped.md`](../../wiki/concepts/wheel-legged-quadruped.md)
- 对照多机型扩展：[`wiki/entities/robot-lab.md`](../../wiki/entities/robot-lab.md)（robot_lab 机型表亦含 DDTRobot Tita）
- 对照厂商 Lab：[`wiki/entities/unitree-rl-lab.md`](../../wiki/entities/unitree-rl-lab.md)、[`wiki/entities/deeprobotics-rl-training.md`](../../wiki/entities/deeprobotics-rl-training.md)
