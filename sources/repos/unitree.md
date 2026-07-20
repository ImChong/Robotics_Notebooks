# unitreerobotics（Unitree Robotics 官方 GitHub 组织）

> 来源归档

- **标题：** Unitree Robotics（unitreerobotics）
- **类型：** repo（GitHub **组织**总览，非单一仓库）
- **机构：** 宇树科技（Unitree）
- **链接：** <https://github.com/unitreerobotics>
- **官网：** <https://www.unitree.com>
- **开发者文档：** <https://support.unitree.com/home/zh/developer>（SDK2 等入口；站点可用性以当时访问为准）
- **Hugging Face：** <https://huggingface.co/unitreerobotics>（模型 / 数据集 / USD 等）
- **公开仓库数：** 约 **49**（截至 2026-07-20 组织 API）
- **一句话说明：** 宇树官方开源组织：覆盖 **SDK2 / ROS / MuJoCo / Isaac Lab RL / XR 遥操作 / LeRobot IL / UnifoLM 基础模型** 等从底层通信到仿真训练、数据采集与具身大模型的完整研发栈；硬件主线见 [wiki/entities/unitree.md](../../wiki/entities/unitree.md)。
- **入库日期：** 2026-04-11（初建）；**深度补全：** 2026-07-20
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)
- **已单独归档子仓：** [unitree_ros](unitree_ros.md)、[unitree_ros_to_real](unitree_ros_to_real.md)、[unitree_rl_mjlab](unitree_rl_mjlab.md)；应用商店门户见 [unitree-unistore](../sites/unitree-unistore.md)

---

## 开源状态（组织级，2026-07-20）

- **已开源**：组织下绝大多数研发相关仓库为公开 GitHub 仓；UnifoLM 系列额外在 Hugging Face 发布权重与数据集。
- **资产迁移注意**：`unitree_model` README 标明 GitHub 仓 **deprecated**，后续 USD/模型更新以 [Hugging Face `unitreerobotics/unitree_model`](https://huggingface.co/datasets/unitreerobotics/unitree_model) 为准；URDF 描述仍大量引用 [unitree_ros](https://github.com/unitreerobotics/unitree_ros)。
- **成品技能分发**：GitHub 研发栈与 **[UniStore](https://unistore.unitree.com/)**（手机 App 一键下发技能）互补，后者见 [sources/sites/unitree-unistore.md](../sites/unitree-unistore.md)。

---

## 组织定位（README / About 快照）

> High performance civilian robot manufacturer. Please everyone be sure to use the robot in a Friendly and Safe manner.

对研究者而言，该组织的价值不在「又一个硬件厂商主页」，而在于：

1. **真机通信入口统一化**：`unitree_sdk2`（C++）+ `unitree_sdk2_python` + CycloneDDS，成为当前 Go2 / B2 / H1 / G1 等新机型的默认底层。
2. **三条并行 RL 训练路线**：Isaac Gym（`unitree_rl_gym`）→ Isaac Lab（`unitree_rl_lab`）→ mjlab/MuJoCo（`unitree_rl_mjlab`），各自配套 Sim2Sim / Sim2Real 叙述。
3. **人形数据闭环**：XR 遥操作（`xr_teleoperate`）→ Isaac Lab 仿真采集（`unitree_sim_isaaclab`）→ LeRobot 训练（`unitree_lerobot`）→ UnifoLM VLA/WMA。

---

## 核心仓库导航（按研究用途，星标截至 2026-07-20）

### A. 底层 SDK 与通信

| 仓库 | ★ | 说明 |
|------|---|------|
| [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) | ~1228 | **SDK v2**（C++，CycloneDDS）；官方开发者文档主入口 |
| [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | ~741 | SDK2 的 Python 绑定（依赖 cyclonedds 0.10.2） |
| [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) | ~762 | ROS 2 直接吃 Unitree DDS msg（Go2 / B2 / H1）；推荐 Humble |
| [unitree_legged_sdk](https://github.com/unitreerobotics/unitree_legged_sdk) | ~426 | **旧代**腿式 SDK（与 ROS1 真机桥配套） |
| [unitree_dds_wrapper](https://github.com/unitreerobotics/unitree_dds_wrapper) | ~33 | 简化 Unitree DDS 通信的薄封装 |

### B. 仿真、模型与经典控制

| 仓库 | ★ | 说明 |
|------|---|------|
| [unitree_ros](https://github.com/unitreerobotics/unitree_ros) | ~1479 | ROS1 + Gazebo URDF/关节仿真；本库已升格 [wiki/entities/unitree-ros.md](../../wiki/entities/unitree-ros.md) |
| [unitree_ros_to_real](https://github.com/unitreerobotics/unitree_ros_to_real) | ~169 | ROS1 ↔ 真机桥 + `unitree_legged_msgs` |
| [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | ~1091 | 官方 MuJoCo 仿真 / Sim2Sim 验证常用仓 |
| [unitree_guide](https://github.com/unitreerobotics/unitree_guide) | ~615 | 配套《四足机器人控制算法》的 Gazebo FSM/Trotting 示例 |
| [unitree_model](https://github.com/unitreerobotics/unitree_model) | ~146 | **已弃用**；USD 等改 HF `unitree_model` dataset |
| [unitree_pybullet](https://github.com/unitreerobotics/unitree_pybullet) | ~71 | PyBullet 仿真（遗产） |

### C. 强化学习训练与部署

| 仓库 | ★ | 说明 |
|------|---|------|
| [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | ~3434 | **星标最高**：Isaac Gym + legged_gym 风格；Go2 / H1 / H1_2 / G1；流程 Train→Play→Sim2Sim→Sim2Real |
| [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | ~1210 | **Isaac Lab 2.x** 官方环境；Go2 / H1 / G1-29dof；资产可从 HF USD 或 `unitree_ros` URDF 引入 |
| [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) | ~533 | **mjlab + MuJoCo Warp**；ONNX→C++→真机；本库已升格 [wiki/entities/unitree-rl-mjlab.md](../../wiki/entities/unitree-rl-mjlab.md) |

### D. 遥操作、模仿学习与仿真采集

| 仓库 | ★ | 说明 |
|------|---|------|
| [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) | ~1563 | XR（AVP / PICO / Quest）全身遥操作 G1/H1；含仿真支持与 Wiki |
| [unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot) | ~723 | 基于 Hugging Face LeRobot 的 G1 双臂灵巧手 IL 训练/测试改版 |
| [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) | ~521 | Isaac Lab 上 G1/H1-2 多执行器任务仿真；与 `xr_teleoperate` 同 DDS，便于仿真采数 |
| [kinect_teleoperate](https://github.com/unitreerobotics/kinect_teleoperate) | ~116 | Azure Kinect 驱动的 H1/G1 遥操作 |
| [televuer](https://github.com/unitreerobotics/televuer) / [teleimager](https://github.com/unitreerobotics/teleimager) | ~46 / ~48 | XR 视觉与多相机（ZMQ/WebRTC）图像服务周边 |
| [UniArmL1](https://github.com/unitreerobotics/UniArmL1) | ~10 | 轻量机械臂遥操作 + 标准化采数，对接 `unitree_lerobot` |

### E. UnifoLM 基础模型族

| 仓库 | ★ | 说明 |
|------|---|------|
| [unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action) | ~1075 | **UnifoLM-WMA-0**：世界模型作仿真引擎 + 动作头策略增强；训练/推理/权重/部署已开源；HF Collections + 项目页 |
| [unifolm-vla](https://github.com/unitreerobotics/unifolm-vla) | ~526 | **UnifoLM-VLA-0**：人形操作 VLA；训练/推理/checkpoint 已开源；配套 G1 Dex1 等 HF 数据集 |

项目页（核查用）：

- WMA：<https://unigen-x.github.io/unifolm-world-model-action.github.io>
- VLA：<https://unigen-x.github.io/unifolm-vla.github.io>

### F. 感知外设与其它

| 仓库 | ★ | 说明 |
|------|---|------|
| [point_lio_unilidar](https://github.com/unitreerobotics/point_lio_unilidar) | ~501 | Unitree LiDAR 的 Point-LIO |
| [unilidar_sdk](https://github.com/unitreerobotics/unilidar_sdk) / [unilidar_sdk2](https://github.com/unitreerobotics/unilidar_sdk2) | ~139 / ~92 | L1 / L2 LiDAR SDK |
| [Qmini](https://github.com/unitreerobotics/Qmini) | ~715 | 小型四足相关开源（社区热度高） |
| [z1_sdk](https://github.com/unitreerobotics/z1_sdk) / [z1_ros](https://github.com/unitreerobotics/z1_ros) | ~45 / ~43 | Z1 机械臂 SDK / ROS |
| 灵巧手服务仓 | — | `dex1_1_service`、`dfx_inspire_service`、`inspire_hand_service`、`brainco_hand_service`、`linker_hand_service` 等 Serial↔DDS |
| [unitree-app-templates](https://github.com/unitreerobotics/unitree-app-templates) | ~16 | UniStore / App Store 应用模板 |
| [Publications](https://github.com/unitreerobotics/Publications) | ~96 | 宇树发表论文索引仓 |

---

## 研发栈总览（选型用）

```text
真机 DDS / SDK2  ←── unitree_sdk2 (+ python) / unitree_ros2
        ↑
   Sim2Real 部署
        ↑
 ┌──────┼──────────────────────────────┐
 │ Isaac Gym     Isaac Lab      mjlab  │
 │ unitree_rl_gym unitree_rl_lab unitree_rl_mjlab
 └──────┬──────────────────────────────┘
        │ 策略 / ONNX
        ↓
 unitree_mujoco（Sim2Sim）→ 真机

人形 IL / VLA 支线：
 xr_teleoperate / Kinect → unitree_sim_isaaclab（仿真采数）
        → unitree_lerobot → UnifoLM-VLA / UnifoLM-WMA
```

---

## 对 wiki 的映射

- **组织枢纽（本资料主升格）：** [`wiki/entities/unitree.md`](../../wiki/entities/unitree.md) — 补充「官方开源软件生态」
- **已有子实体：** [`wiki/entities/unitree-ros.md`](../../wiki/entities/unitree-ros.md)、[`wiki/entities/unitree-rl-mjlab.md`](../../wiki/entities/unitree-rl-mjlab.md)、[`wiki/entities/unitree-g1.md`](../../wiki/entities/unitree-g1.md)、[`wiki/entities/unitree-unistore.md`](../../wiki/entities/unitree-unistore.md)
- **交叉主题：** [`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md)、[`wiki/entities/lerobot.md`](../../wiki/entities/lerobot.md)、[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)、[`wiki/entities/legged-gym.md`](../../wiki/entities/legged-gym.md)
- **后续可单独升格（未在本次新建实体页）：** `unitree_rl_gym`、`unitree_rl_lab`、`unitree_sdk2`、`xr_teleoperate`、`unifolm-vla`、`unifolm-world-model-action` — 优先服务学习主线时再深度 ingest
