# ai_worker（ROBOTIS AI Worker / FFW ROS 2）

> 来源归档

- **标题：** AI Worker: FFW (Freedom From Work)
- **类型：** repo
- **链接：** https://github.com/ROBOTIS-GIT/ai_worker
- **机构：** 乐百机器人（ROBOTIS）
- **Stars：** ~159（2026-08）
- **许可：** Apache-2.0
- **主页：** https://ai.robotis.com/
- **入库日期：** 2026-08-07
- **一句话说明：** ROBOTIS AI Worker（FFW 系列）官方 ROS 2 包：描述、bringup、导航、遥操作与 Docker；对接 Physical AI Tools / LeRobot。
- **沉淀到 wiki：** [robotis-ai-worker](../../wiki/entities/robotis-ai-worker.md)

---

## 核心定位

官方 ROS 2 工作区元包族，前缀多为 `ffw_*`（Freedom From Work）。提供：

- `ffw_description` / `ffw_bringup` — URDF 与启动
- `ffw_navigation` — Nav2 + BT 导航模式
- `ffw_teleop`、摇杆 / 轨迹广播 / swerve 驱动控制器
- `ffw_moveit_config`、`ffw_robot_manager`、弹簧执行器控制器等
- `docker/` — AMD64/ARM64 Dockerfile + s6 服务（bringup / navigation / avatar）

配套：`physical_ai_tools`、`robotis_mujoco_menagerie`、HF `ROBOTIS`、Docker `robotis/ros`。

---

## 开源状态

**已开源** — Apache-2.0；组织页与 README 链齐全。

---

## 对 wiki 的映射

- **wiki/entities/robotis-ai-worker.md**（新建）
- **wiki/entities/robotis.md** — hub
- **wiki/entities/cyclo-lab.md** / **robotis-physical-ai-tools.md** — Sim2Real / 采集训练链
