# tita_rl_sim2sim2real

> 来源归档

- **标题：** tita_rl_sim2sim2real（TITA RL 的 Webots sim2sim 与真机部署）
- **类型：** repo
- **来源：** DDTRobot（直驱科技 Direct Drive Tech）
- **链接：** https://github.com/DDTRobot/tita_rl_sim2sim2real
- **星标（截至 2026-08-28）：** 20
- **最近推送：** 2025-08-15
- **主要语言：** 未在 API 标明（ROS 2 / C++ 工作空间）
- **许可证：** 仓库 API 未返回 SPDX（以源码头 / README 为准）
- **分类：** sim2sim / sim2real 部署 / Webots / ROS 2
- **入库日期：** 2026-08-28
- **一句话说明：** TITA 官方 RL 的下游部署仓：把 `tita_rl` 导出的 TensorRT engine 接到 Webots 2023 + `ros2_control`，同一套 bringup 再上真机。
- **沉淀到 wiki：** 否（并入 [`wiki/entities/tita-rl.md`](../../wiki/entities/tita-rl.md) 的部署节，不单独建实体页）
- **机构：** 直驱科技（Direct Drive Tech）→ `direct-drive-tech` / `ddt`
- **项目页：** 无独立项目页

---

## README 要点（编译自上游）

- 依赖：ROS 2 Humble、[`TITA_ROS2_Control_Sim`](https://github.com/DDTRobot/TITA_ROS2_Control_Sim)；可选 Docker [`webots2023b_ros2_docker`](https://github.com/DDTRobot/webots2023b_ros2_docker)（已配 TensorRT / Webots / ROS 2）。
- 工作空间用 `vcs import < sim2sim2real.repos` 拉齐依赖。
- 必须把推理出的 `model_gn.engine` 路径写进 `FSMState_RL.cpp`（Docker 内常见 `/mnt/dev/*.engine`，不要写宿主机路径）。
- sim2sim：`ros2 launch locomotion_bringup sim_bringup.launch.py` + `keyboard_controller_node`（命名空间 `/tita`）。
- 真机：scp 到 `robot@192.168.42.1`（默认密码见 README）、停 `tita-bringup.service`、`hw_bringup.launch.py ctrl_mode:=wbc`；板载镜像缺 TensorRT dev 时需自装后再 `trtexec`。
- TensorRT 10.x 兼容见上游 [issue #1](https://github.com/DDTRobot/tita_rl_sim2sim2real/issues/1)。

## 开源状态

- **已开源**：公开 GitHub 仓库；训练不在本仓，上游是 [`tita_rl`](./tita_rl.md)。

## 对 wiki 的映射

- 部署细节写入：[`wiki/entities/tita-rl.md`](../../wiki/entities/tita-rl.md)
- Webots 对照：[`wiki/entities/webots.md`](../../wiki/entities/webots.md)
