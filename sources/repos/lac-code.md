# lac-humanoid/lac-code

> 来源归档

- **标题：** LAC MuJoCo sim2sim + G1 checkpoint
- **类型：** repo
- **链接：** https://github.com/lac-humanoid/lac-code
- **主页：** https://lac-humanoid.github.io/
- **论文：** https://arxiv.org/abs/2608.25405
- **许可：** MIT（`sim/` 为 Unitree `unitree_mujoco` 子集，BSD-3-Clause）
- **入库日期：** 2026-08-28
- **一句话说明：** 23-DoF G1 的推理栈：52 MB 策略、100 个 OMOMO 重定向上身姿态、ROS 2 部署节点与无 ROS smoke test。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-lac.md`](../../wiki/entities/paper-lac.md)

## 开源核查（2026-08-28）

**部分开源** — 可运行推理 / sim2sim，无可运行训练。

| 路径 | 角色 |
|------|------|
| `checkpoints/lac_g1_23dof.pt` | 推理权重（52 MB） |
| `config/policy_cfg.yaml` | 网络结构 |
| `motions/lac_motion_library.npz` | 100 上身姿态 |
| `ros2/lac_deploy/` | `inference` / `relay` / `stiffness_control` |
| `sim/` | MuJoCo（vendored unitree_mujoco） |
| `tests/smoke.py` | 观测布局 + ckpt 前向 |

最短路径：`python tests/smoke.py`；四终端 + Xbox：`unitree_mujoco.py` → `ros2 run lac_deploy inference` → `relay` → `stiffness_control`。

## 对 wiki 的映射

- [LAC 论文摘录](../papers/lac_arxiv_2608_25405.md)
- [LAC 项目页](../sites/lac-humanoid.md)
- [LAC 实体](../../wiki/entities/paper-lac.md)
