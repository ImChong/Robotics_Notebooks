# unitreerobotics（Unitree Robotics 官方 GitHub 组织）

> 来源归档

- **标题：** Unitree Robotics（unitreerobotics）
- **类型：** repo（GitHub **组织**总览，非单一仓库）
- **机构：** 宇树科技（Unitree）
- **链接：** <https://github.com/unitreerobotics>
- **官网：** <https://www.unitree.com>
- **开发者文档：** <https://support.unitree.com/home/zh/developer>（SDK2 等入口；站点可用性以当时访问为准）
- **Hugging Face：** <https://huggingface.co/unitreerobotics>（模型 / 数据集 / USD 等）
- **公开仓库数：** 约 **52**（截至 2026-07-24 组织 API）
- **独立 wiki 节点：** **45** 个活跃仓（跳过 7 个过时/元仓）
- **一句话说明：** 宇树官方开源组织：覆盖 **SDK2 / ROS / MuJoCo / Isaac Lab RL / XR 遥操作 / LeRobot IL / UnifoLM 基础模型** 等从底层通信到仿真训练、数据采集与具身大模型的完整研发栈；硬件主线见 [wiki/entities/unitree.md](../../wiki/entities/unitree.md)。
- **入库日期：** 2026-04-11（初建）；**深度补全：** 2026-07-20；**全仓独立节点：** 2026-07-24
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)
- **应用商店门户：** [unitree-unistore](../sites/unitree-unistore.md)

---

## 开源状态（组织级，2026-07-24）

- **已开源**：组织下绝大多数研发相关仓库为公开 GitHub 仓；UnifoLM 系列额外在 Hugging Face 发布权重与数据集。
- **资产迁移注意**：`unitree_model` README 标明 GitHub 仓 **deprecated**，后续 USD/模型更新以 [Hugging Face `unitreerobotics/unitree_model`](https://huggingface.co/datasets/unitreerobotics/unitree_model) 为准；URDF 描述仍大量引用 [unitree_ros](https://github.com/unitreerobotics/unitree_ros)。
- **成品技能分发**：GitHub 研发栈与 **[UniStore](https://unistore.unitree.com/)**（手机 App 一键下发技能）互补，后者见 [sources/sites/unitree-unistore.md](../sites/unitree-unistore.md)。

---

## 已单独归档并升格的子仓（全量）

### 底层 SDK / 通信

| 仓库 | sources | wiki |
|------|---------|------|
| [unitree_sdk2](unitree_sdk2.md) | [`sources/repos/unitree_sdk2.md`](unitree_sdk2.md) | [`wiki/entities/unitree-sdk2.md`](../../wiki/entities/unitree-sdk2.md) |
| [unitree_sdk2_python](unitree_sdk2_python.md) | [`sources/repos/unitree_sdk2_python.md`](unitree_sdk2_python.md) | [`wiki/entities/unitree-sdk2-python.md`](../../wiki/entities/unitree-sdk2-python.md) |
| [unitree_legged_sdk](unitree_legged_sdk.md) | [`sources/repos/unitree_legged_sdk.md`](unitree_legged_sdk.md) | [`wiki/entities/unitree-legged-sdk.md`](../../wiki/entities/unitree-legged-sdk.md) |
| [unitree_actuator_sdk](unitree_actuator_sdk.md) | [`sources/repos/unitree_actuator_sdk.md`](unitree_actuator_sdk.md) | [`wiki/entities/unitree-actuator-sdk.md`](../../wiki/entities/unitree-actuator-sdk.md) |
| [unitree_dds_wrapper](unitree_dds_wrapper.md) | [`sources/repos/unitree_dds_wrapper.md`](unitree_dds_wrapper.md) | [`wiki/entities/unitree-dds-wrapper.md`](../../wiki/entities/unitree-dds-wrapper.md) |

### ROS 集成

| 仓库 | sources | wiki |
|------|---------|------|
| [unitree_ros](unitree_ros.md) | [`sources/repos/unitree_ros.md`](unitree_ros.md) | [`wiki/entities/unitree-ros.md`](../../wiki/entities/unitree-ros.md) |
| [unitree_ros2](unitree_ros2.md) | [`sources/repos/unitree_ros2.md`](unitree_ros2.md) | [`wiki/entities/unitree-ros2.md`](../../wiki/entities/unitree-ros2.md) |
| [unitree_ros_to_real](unitree_ros_to_real.md) | [`sources/repos/unitree_ros_to_real.md`](unitree_ros_to_real.md) | [`wiki/entities/unitree-ros-to-real.md`](../../wiki/entities/unitree-ros-to-real.md) |
| [unitree_ros2_to_real](unitree_ros2_to_real.md) | [`sources/repos/unitree_ros2_to_real.md`](unitree_ros2_to_real.md) | [`wiki/entities/unitree-ros2-to-real.md`](../../wiki/entities/unitree-ros2-to-real.md) |

### 仿真与模型

| 仓库 | sources | wiki |
|------|---------|------|
| [unitree_mujoco](unitree_mujoco.md) | [`sources/repos/unitree_mujoco.md`](unitree_mujoco.md) | [`wiki/entities/unitree-mujoco.md`](../../wiki/entities/unitree-mujoco.md) |
| [unitree_guide](unitree_guide.md) | [`sources/repos/unitree_guide.md`](unitree_guide.md) | [`wiki/entities/unitree-guide.md`](../../wiki/entities/unitree-guide.md) |
| [unitree_model](unitree_model.md) | [`sources/repos/unitree_model.md`](unitree_model.md) | [`wiki/entities/unitree-model.md`](../../wiki/entities/unitree-model.md) |

### 强化学习训练

| 仓库 | sources | wiki |
|------|---------|------|
| [unitree_rl_gym](unitree_rl_gym.md) | [`sources/repos/unitree_rl_gym.md`](unitree_rl_gym.md) | [`wiki/entities/unitree-rl-gym.md`](../../wiki/entities/unitree-rl-gym.md) |
| [unitree_rl_lab](unitree_rl_lab.md) | [`sources/repos/unitree_rl_lab.md`](unitree_rl_lab.md) | [`wiki/entities/unitree-rl-lab.md`](../../wiki/entities/unitree-rl-lab.md) |
| [unitree_rl_mjlab](unitree_rl_mjlab.md) | [`sources/repos/unitree_rl_mjlab.md`](unitree_rl_mjlab.md) | [`wiki/entities/unitree-rl-mjlab.md`](../../wiki/entities/unitree-rl-mjlab.md) |

### 遥操作与采数

| 仓库 | sources | wiki |
|------|---------|------|
| [xr_teleoperate](xr_teleoperate.md) | [`sources/repos/xr_teleoperate.md`](xr_teleoperate.md) | [`wiki/entities/xr-teleoperate.md`](../../wiki/entities/xr-teleoperate.md) |
| [unitree_sim_isaaclab](unitree_sim_isaaclab.md) | [`sources/repos/unitree_sim_isaaclab.md`](unitree_sim_isaaclab.md) | [`wiki/entities/unitree-sim-isaaclab.md`](../../wiki/entities/unitree-sim-isaaclab.md) |
| [kinect_teleoperate](kinect_teleoperate.md) | [`sources/repos/kinect_teleoperate.md`](kinect_teleoperate.md) | [`wiki/entities/kinect-teleoperate.md`](../../wiki/entities/kinect-teleoperate.md) |
| [teleimager](teleimager.md) | [`sources/repos/teleimager.md`](teleimager.md) | [`wiki/entities/teleimager.md`](../../wiki/entities/teleimager.md) |
| [televuer](televuer.md) | [`sources/repos/televuer.md`](televuer.md) | [`wiki/entities/televuer.md`](../../wiki/entities/televuer.md) |

### 模仿学习

| 仓库 | sources | wiki |
|------|---------|------|
| [unitree_lerobot](unitree_lerobot.md) | [`sources/repos/unitree_lerobot.md`](unitree_lerobot.md) | [`wiki/entities/unitree-lerobot.md`](../../wiki/entities/unitree-lerobot.md) |
| [UniArmL1](UniArmL1.md) | [`sources/repos/UniArmL1.md`](UniArmL1.md) | [`wiki/entities/uniarm-l1.md`](../../wiki/entities/uniarm-l1.md) |

### 基础模型（UnifoLM）

| 仓库 | sources | wiki |
|------|---------|------|
| [unifolm-world-model-action](unifolm-world-model-action.md) | [`sources/repos/unifolm-world-model-action.md`](unifolm-world-model-action.md) | [`wiki/entities/unifolm-world-model-action.md`](../../wiki/entities/unifolm-world-model-action.md) |
| [unifolm-vla](unifolm-vla.md) | [`sources/repos/unifolm-vla.md`](unifolm-vla.md) | [`wiki/entities/unifolm-vla.md`](../../wiki/entities/unifolm-vla.md) |

### LiDAR 感知

| 仓库 | sources | wiki |
|------|---------|------|
| [point_lio_unilidar](point_lio_unilidar.md) | [`sources/repos/point_lio_unilidar.md`](point_lio_unilidar.md) | [`wiki/entities/point-lio-unilidar.md`](../../wiki/entities/point-lio-unilidar.md) |
| [unilidar_sdk](unilidar_sdk.md) | [`sources/repos/unilidar_sdk.md`](unilidar_sdk.md) | [`wiki/entities/unilidar-sdk.md`](../../wiki/entities/unilidar-sdk.md) |
| [unilidar_sdk2](unilidar_sdk2.md) | [`sources/repos/unilidar_sdk2.md`](unilidar_sdk2.md) | [`wiki/entities/unilidar-sdk2.md`](../../wiki/entities/unilidar-sdk2.md) |

### 相机外设

| 仓库 | sources | wiki |
|------|---------|------|
| [UnitreecameraSDK](UnitreecameraSDK.md) | [`sources/repos/UnitreecameraSDK.md`](UnitreecameraSDK.md) | [`wiki/entities/unitree-camera-sdk.md`](../../wiki/entities/unitree-camera-sdk.md) |

### Z1 机械臂

| 仓库 | sources | wiki |
|------|---------|------|
| [z1_sdk](z1_sdk.md) | [`sources/repos/z1_sdk.md`](z1_sdk.md) | [`wiki/entities/z1-sdk.md`](../../wiki/entities/z1-sdk.md) |
| [z1_ros](z1_ros.md) | [`sources/repos/z1_ros.md`](z1_ros.md) | [`wiki/entities/z1-ros.md`](../../wiki/entities/z1-ros.md) |
| [z1_controller](z1_controller.md) | [`sources/repos/z1_controller.md`](z1_controller.md) | [`wiki/entities/z1-controller.md`](../../wiki/entities/z1-controller.md) |
| [z1_joystick](z1_joystick.md) | [`sources/repos/z1_joystick.md`](z1_joystick.md) | [`wiki/entities/z1-joystick.md`](../../wiki/entities/z1-joystick.md) |

### 灵巧手服务

| 仓库 | sources | wiki |
|------|---------|------|
| [dfx_inspire_service](dfx_inspire_service.md) | [`sources/repos/dfx_inspire_service.md`](dfx_inspire_service.md) | [`wiki/entities/dfx-inspire-service.md`](../../wiki/entities/dfx-inspire-service.md) |
| [dex1_1_service](dex1_1_service.md) | [`sources/repos/dex1_1_service.md`](dex1_1_service.md) | [`wiki/entities/dex1-1-service.md`](../../wiki/entities/dex1-1-service.md) |
| [brainco_hand_service](brainco_hand_service.md) | [`sources/repos/brainco_hand_service.md`](brainco_hand_service.md) | [`wiki/entities/brainco-hand-service.md`](../../wiki/entities/brainco-hand-service.md) |
| [linker_hand_service](linker_hand_service.md) | [`sources/repos/linker_hand_service.md`](linker_hand_service.md) | [`wiki/entities/linker-hand-service.md`](../../wiki/entities/linker-hand-service.md) |

### 开源平台

| 仓库 | sources | wiki |
|------|---------|------|
| [Qmini](Qmini.md) | [`sources/repos/Qmini.md`](Qmini.md) | [`wiki/entities/qmini.md`](../../wiki/entities/qmini.md) |

### UniStore / App

| 仓库 | sources | wiki |
|------|---------|------|
| [unitree-app-templates](unitree-app-templates.md) | [`sources/repos/unitree-app-templates.md`](unitree-app-templates.md) | [`wiki/entities/unitree-app-templates.md`](../../wiki/entities/unitree-app-templates.md) |

### SLAM / 行业示例

| 仓库 | sources | wiki |
|------|---------|------|
| [Python_unitree_demos](Python_unitree_demos.md) | [`sources/repos/Python_unitree_demos.md`](Python_unitree_demos.md) | [`wiki/entities/python-unitree-demos.md`](../../wiki/entities/python-unitree-demos.md) |
| [unitree_slam](unitree_slam.md) | [`sources/repos/unitree_slam.md`](unitree_slam.md) | [`wiki/entities/unitree-slam.md`](../../wiki/entities/unitree-slam.md) |

### 工具与调试

| 仓库 | sources | wiki |
|------|---------|------|
| [logging-mp](logging-mp.md) | [`sources/repos/logging-mp.md`](logging-mp.md) | [`wiki/entities/logging-mp.md`](../../wiki/entities/logging-mp.md) |
| [digital_servo](digital_servo.md) | [`sources/repos/digital_servo.md`](digital_servo.md) | [`wiki/entities/digital-servo.md`](../../wiki/entities/digital-servo.md) |
| [unitree-motor-debugging-assistant](unitree-motor-debugging-assistant.md) | [`sources/repos/unitree-motor-debugging-assistant.md`](unitree-motor-debugging-assistant.md) | [`wiki/entities/unitree-motor-debugging-assistant.md`](../../wiki/entities/unitree-motor-debugging-assistant.md) |

### 赛事 / Challenge

| 仓库 | sources | wiki |
|------|---------|------|
| [unibot_submission](unibot_submission.md) | [`sources/repos/unibot_submission.md`](unibot_submission.md) | [`wiki/entities/unibot-submission.md`](../../wiki/entities/unibot-submission.md) |

### 组织元资料

| 仓库 | sources | wiki |
|------|---------|------|
| [Publications](Publications.md) | [`sources/repos/Publications.md`](Publications.md) | [`wiki/entities/publications.md`](../../wiki/entities/publications.md) |

## 跳过的过时 / 元仓

| 仓库 | 原因 |
|------|------|
| `.github` | 组织 workflow / 元配置 |
| `unitreerobotics.github.io` | 组织 GitHub Pages |
| `Acknowledgement` | 致谢清单 |
| `laikago_ros` | Laikago 停产；最近推送 2020 |
| `unitree_pybullet` | PyBullet 遗产；最近推送 2020 |
| `aliengo_sdk` | Aliengo 旧 SDK；最近推送 2020 |
| `unitree_cad` | CAD 资产；非软件栈入口 |

## 对 wiki 的映射

- **组织枢纽：** [`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)
- **交叉主题：** [`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md)、[`wiki/entities/lerobot.md`](../../wiki/entities/lerobot.md)、[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)、[`wiki/entities/legged-gym.md`](../../wiki/entities/legged-gym.md)
