# rl-vs-gc（Isaac Lab 四旋翼 / 空中机械臂 RL vs 几何控制）

> 来源归档

- **标题：** rl-vs-gc
- **类型：** repo
- **来源：** 宾夕法尼亚大学 GRASP Lab（Pratik Kunapuli / Jake Welde / Dinesh Jayaraman / Vijay Kumar）
- **链接：** <https://github.com/PratikKunapuli/rl-vs-gc>
- **项目页：** <https://pratikkunapuli.github.io/rl-vs-gc/>
- **论文：** <https://arxiv.org/abs/2506.17832>（RSS 2025）
- **许可：** 截至 2026-08-26 **未声明 SPDX**（无 `LICENSE` 文件）
- **依赖钉扎：** IsaacSim 4.2.0.2、Isaac Lab 1.4.1、Python 3.10
- **入库日期：** 2026-08-26
- **一句话说明：** RSS 2025 论文配套仓：Isaac Lab DirectRLEnv 上的轨迹跟踪 / 接球，以及几何控制器 Optuna 调参与 RSL-RL PPO 训练/评测。
- **沉淀到 wiki：** [`wiki/entities/paper-rl-vs-gc.md`](../../wiki/entities/paper-rl-vs-gc.md)

---

## 开源边界（2026-08-26 核查）

| 项 | 状态 |
|----|------|
| 环境 | `envs/trajectory_tracking/`、`envs/ball_catching/`（DirectRLEnv） |
| RL | `rl/train_rslrl.py`、`rl/eval_rslrl.py`；Hydra 配奖励/初值/Lissajous/horizon |
| GC | `controllers/geometric_controller.py`、`gc_tuning.py`（Optuna + SQLite）、`gc_params.py` 预置增益 |
| Checkpoint | `rl/logs/rsl_rl/PaperModels/`（Hover/Lissajous × FF/NoFF） |
| 真机部署 | **无**；论文评测为仿真 |
| 许可证 | **未声明** |

结论：**已开源**可运行训练/调参/评测入口；复用需自行核对授权。

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 安装 | conda `isaaclab` → IsaacSim pip → clone IsaacLab `./isaaclab.sh --install` → `pip install -e .` |
| Demo | `demo_env.py` 从 Gymnasium API 拉起 IsaacSim 窗口 |
| 训练 | `python train_rslrl.py --task Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0 --num_envs 4096 ...` |
| 评测 | `python eval_rslrl.py --task ... --num_envs 1000 --experiment_name PaperModels --load_run ...`；`--baseline true` 切 GC |
| GC 调参 | `python gc_tuning.py --task ... --num_envs 1024`（可选 MySQL；默认可 SQLite） |
| 任务 ID | `Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0`、`...-QuadOnly-...`、`Isaac-BrushlessCrazyflie-TrajectoryTracking-v0` |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-rl-vs-gc](../../wiki/entities/paper-rl-vs-gc.md) | 论文结论与三条不对称协议 |
| [rl-vs-geometric-control](../../wiki/comparisons/rl-vs-geometric-control.md) | 选型对比 |
| [isaac-lab](../../wiki/entities/isaac-lab.md) | DirectRLEnv + RSL-RL 训练栈 |
| [gym-pybullet-drones](../../wiki/entities/gym-pybullet-drones.md) | 更轻的四旋翼 RL 基准对照 |
