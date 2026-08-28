# tita_rl

> 来源归档

- **标题：** tita_rl（TITA 官方 Isaac Gym RL）
- **类型：** repo
- **来源：** DDTRobot（直驱科技 Direct Drive Tech 官方 GitHub 组织）
- **链接：** https://github.com/DDTRobot/tita_rl
- **星标（截至 2026-08-28）：** 137
- **最近推送：** 2025-11-04
- **主要语言：** Python
- **许可证：** MIT
- **分类：** 强化学习训练 / Isaac Gym / 轮腿双足
- **入库日期：** 2026-08-28
- **一句话说明：** 直驱科技 TITA 轮腿双足的官方 Isaac Gym 训练仓；任务 `tita_constraint`，算法 NP3O，导出 ONNX 后走配套仓做 Webots sim2sim 与真机 TensorRT 部署。
- **沉淀到 wiki：** 是 → [`wiki/entities/tita-rl.md`](../../wiki/entities/tita-rl.md)
- **机构：** 直驱科技（Direct Drive Tech）→ `direct-drive-tech` / `ddt`
- **项目页：** 无独立 `*.github.io`；以 GitHub README 为准

---

## README 要点（编译自上游）

- 叙事：**Official support TITA Reinforcement Learning repo**；强化学习部分基于 [`zeonsunlightyu/LocomotionWithNP3O`](https://github.com/zeonsunlightyu/LocomotionWithNP3O)。
- 参考环境：RTX 3060、CUDA 12.5、**Isaac Gym**、Webots 2023 sim2sim、ROS 2 Humble、推理在 RTX 3060 或 TITA 板载 Jetson Orin NX + TensorRT；conda + Python 3.8。
- 开源范围拆成三截：
  1. 本仓：Isaac Gym 仿真训练（`python train.py --task=tita_constraint [--headless]`）
  2. 评估：`python simple_play.py --task=tita_constraint`（仓内附 `tita_example_10000.pt`）
  3. sim2sim / 真机：[`DDTRobot/tita_rl_sim2sim2real`](https://github.com/DDTRobot/tita_rl_sim2sim2real)（ONNX → `trtexec` → `model_gn.engine`）
- 姊妹形态仓（非本仓范围）：[`titatit_rl`](https://github.com/DDTRobot/titatit_rl)（TITATIT 四足模式）、[`quadruped-wheel-titatit-rl`](https://github.com/DDTRobot/quadruped-wheel-titatit-rl)（TITATIT 四轮足）。
- 资产：仓内 `resources/tita/urdf/tita_description.urdf` + meshes；8 关节（左右 `leg_1`–`leg_4`），足端名为 `leg_4`，基座接触终止。
- 观测：本体 `n_proprio=33` + 高程扫描 187 + 历史 10 步 + 特权 latent；默认 `num_envs=4096`，`measure_heights=True`。
- 域随机：摩擦 / 恢复系数 / 基座质量与 CoM / 推扰 / 电机强度 / KpKd / 动作滞后。
- 代价项（NP3O `costs`）：位置/力矩/关节速度限、加速度平滑、足端接触力、绊倒；`num_costs=6`。
- 默认速度指令：\(v_x,v_y \in [-1,1]\) m/s，偏航 \(\pm 1\) rad/s；`heading_command=True`。

## 开源状态

- **已开源**：公开 GitHub 仓库（MIT）；含可运行 `train.py` / `simple_play.py` / `export_policy_as_onnx.py` 与示例 checkpoint。
- **部署分仓**：Webots / 真机不在本仓，见 [`tita_rl_sim2sim2real`](./tita_rl_sim2sim2real.md)。
- **与 Isaac Lab 官方栈的关系**：同组织后续入口是 [`DDT_Lab`](./ddt_lab.md)（Isaac Lab + NP3O，覆盖 D1 / Tita）；本仓是 **Isaac Gym 世代** 的 TITA 训练实现，不要当成 Lab 扩展。

## 对 wiki 的映射

- 实体页：[`wiki/entities/tita-rl.md`](../../wiki/entities/tita-rl.md)
- 形态概念：[`wiki/concepts/wheel-legged-biped.md`](../../wiki/concepts/wheel-legged-biped.md)
- 对照 Isaac Lab 官方仓：[`wiki/entities/ddt-lab.md`](../../wiki/entities/ddt-lab.md)
- 对照四轮足：[`wiki/concepts/wheel-legged-quadruped.md`](../../wiki/concepts/wheel-legged-quadruped.md)
