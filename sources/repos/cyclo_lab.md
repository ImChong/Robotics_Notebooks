# cyclo_lab（ROBOTIS Isaac Lab RL/IL）

> 来源归档

- **标题：** cyclo_lab
- **类型：** repo
- **链接：** https://github.com/ROBOTIS-GIT/cyclo_lab
- **机构：** 乐百机器人（ROBOTIS）
- **Stars：** ~136（2026-08）
- **许可：** Apache-2.0
- **Topics：** imitation-learning, isaaclab, isaacsim, reinforcement-learning, sim2real, robotis
- **入库日期：** 2026-08-07
- **一句话说明：** 基于 Isaac Lab 的 ROBOTIS 官方 RL/IL 教程仓：OMY / FFW-BG2 等任务、Isaac Lab Mimic、Sim2Real DDS bringup；对齐 Isaac Sim 5.1 / Lab 2.3。
- **沉淀到 wiki：** [cyclo-lab](../../wiki/entities/cyclo-lab.md)

---

## 核心定位

厂商官方 Isaac Lab 扩展（`source/cyclo_lab`），Docker 内预装 Isaac Sim 5.1.0、Isaac Lab 2.3、CycloneDDS、`robotis_dds_python`、独立 `lerobot_env`。

### 示例任务（README）

| 范式 | Gym / 任务名 | 说明 |
|------|--------------|------|
| RL | `Cyclo-Reach-OMY-v0` / `Cyclo-Lift-Cube-OMY-v0` / `Cyclo-Open-Drawer-OMY-v0` | OMY 臂 |
| RL | `Cyclo-Reach-FFW-BG2-v0` | AI Worker FFW-BG2 |
| IL | `Cyclo-Stack-Cube-OMY-IK-Rel-v0` (+ Mimic) | 录制 → annotate → generate → robomimic BC |
| IL | `Cyclo-PickPlace-FFW-BG2-IK-Rel-v0` | FFW pick-place + cameras |

训练入口：`scripts/reinforcement_learning/rsl_rl|rl_games|sb3|skrl/`；IL：`scripts/imitation_learning/`；Sim2Real：`scripts/sim2real/`（含 `sh5_dds_bringup.py`、OMY reach 推理、IsaacLab→LeRobot 转换）。

真机 bringup 依赖 [open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator) / [ai_worker](https://github.com/ROBOTIS-GIT/ai_worker)；采集训练侧指 [physical_ai_tools](https://github.com/ROBOTIS-GIT/physical_ai_tools)。

---

## 开源状态

**已开源** — Apache-2.0；可运行训练/播放/Sim2Real 脚本齐全。

---

## 对 wiki 的映射

- **wiki/entities/cyclo-lab.md**（新建）— 与 unitree_rl_lab / deeprobotics-rl-training / DDT_Lab 厂商 Lab 对照
- **wiki/entities/robot-lab.md** — 社区多厂商扩展对照
