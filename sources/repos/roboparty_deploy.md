# roboparty_deploy

> 来源归档（国内具身开源全景 · RP1 官方仓库体系复核）

- **标题：** roboparty_deploy — RPO / RP1 的 ROS 2 部署框架
- **类型：** repo
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/roboparty_deploy （旧名 [`atom01_deploy`](atom01_deploy.md)，两者 HEAD 同为 `c4d8b87`，2026-10-27 核对）
- **分类：** 部署运行时
- **入库日期：** 2026-09-06（占位）；2026-10-27 复核并扩写（快照 `c4d8b87`，2026-10-26，提交说明 `update motors submodule for LRO and RP protocol fixes`）
- **一句话说明：** C++17 / ROS 2（Humble、Jazzy）部署框架，经 5 个 Git 子模块提供电机（DM / EVO / LRO / RP / XYN）、IMU（HIPNUC / MCT7123）、ONNX 推理与 RealSense 深度感知，并含 Python SDK、归零脚本与实时内核资源。
- **沉淀到 wiki：** 是（[`wiki/entities/roboparty-deploy.md`](../../wiki/entities/roboparty-deploy.md)，`project_id: roboparty-deploy`）
- **旧来源：** [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 开源状态

- **已开源**：GPL-3.0-only（`LICENSE`、`LICENSES.md`）。
- **子模块各自授权**：`roboparty_inference` / `roboparty_motors` / `roboparty_imu` GPL-3.0，`roboparty_camera` GPL-3.0-only，`tools/create_ap` BSD-2-Clause（`LICENSES.md`）；`assets/` 中的 Linux 内核 `.deb` 等被声明可能来自第三方，不因仓库级 GPL 重新授权。
- **RP1 文档口径不一致**：`README.md` 文档表仍写「RP1 Deployment — Coming soon」，但 `docs/operation.md` 已给出 `rp1_24dof` 启动命令，子模块 `roboparty_inference` 中存在 `robots/rp1_24dof/` 的配置与 ONNX 模型（见 wiki 页）。
- 子模块目录在浅克隆中为空，需 `git submodule update --init --recursive`；本次归档另行浅克隆了 inference / motors / imu / camera 四个子仓核对。

## 关键文件清单

| 路径 | 内容 |
|------|------|
| `docs/rpo_deploy.md` | 仅 RPO：Orange Pi 5 Plus（Ubuntu 22.04，内核 5.10）与 RDK X5（内核 6.1.83）、实时优先级 `rtprio 98`、udev、CAN 映射 `can0–can3`、归零 |
| `docs/operation.md` | RPO / RP1 通用：构建、`start_robot.sh`、手柄、10 个 `std_srvs/Trigger` 服务 |
| `docs/configuration.md` | `robot.yaml` 与策略 YAML 字段、电机型号枚举、观测 `name:size@tap` 语法 |
| `docs/python_sdk.md` | `imu_py` / `motors_py` / `robot_py` |
| `scripts/config/set_zero_rp1.yaml`、`set_zero_rp1_24dof.yaml`、`set_zero_rpo.yaml` | 归零配置（RP1：LRO，`canfd`，`motor_num` 分别 `[7,7,7,7]` / `[7,7,5,5]`） |
| `assets/` | `rt_fastdds_profile.xml`、两份 udev 规则、`linux-*-legacy-rockchip-rk3588_1.2.0_arm64.deb` ×3 |
| `tools/` | `start_robot.sh`、`start_camera.sh`、`ros_environment.sh`、`create_ap` 子模块 |

## 对 wiki 的映射

- [wiki/entities/roboparty-deploy.md](../../wiki/entities/roboparty-deploy.md)
- 上游：[roboparty_train](roboparty_train.md)（策略来源）、[rp1_description](rp1_description.md)（关节顺序）
- 总览：[roboparty-rp1](../../wiki/entities/roboparty-rp1.md)、[roboparty](../../wiki/entities/roboparty.md)
