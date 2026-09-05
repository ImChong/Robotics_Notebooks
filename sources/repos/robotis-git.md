# ROBOTIS-GIT（乐百机器人官方 GitHub 组织）

> 来源归档

- **标题：** ROBOTIS（ROBOTIS-GIT）
- **类型：** repo（GitHub **组织**总览，非单一仓库）
- **机构：** 乐百机器人（ROBOTIS）
- **链接：** <https://github.com/ROBOTIS-GIT>
- **官网：** <https://www.robotis.com/>
- **Physical AI 文档：** <https://ai.robotis.com/>
- **产品手册：** <https://docs.robotis.com/> / eManual <https://emanual.robotis.com/>
- **Hugging Face：** <https://huggingface.co/ROBOTIS>
- **Docker Hub：** <https://hub.docker.com/r/robotis>
- **公开仓库数：** **154**（截至 2026-09-05；Followers ~1182）
- **wiki 策略：** 组织总览升格为 hub；Physical AI / Cyclo 主线仓与 Dynamixel SDK、MuJoCo 资产等建**有详情的独立节点**；教育平台（TurtleBot3 / OP3 / THORMANG3 / OpenMANIPULATOR）复用已有实体并回链；周边与过时仓**仅本目录或组织页归档**，不建 stub。
- **入库日期：** 2026-08-07
- **沉淀到 wiki：** 是 → [`wiki/entities/robotis.md`](../../wiki/entities/robotis.md)

---

## 开源状态（2026-09-05 复核）

- 组织页公开；主线 Physical AI / ROS / Dynamixel 仓多为 **Apache-2.0**。
- Physical AI 教程与数据集入口：[ai.robotis.com](https://ai.robotis.com/)、[Hugging Face/ROBOTIS](https://huggingface.co/ROBOTIS)。
- Cyclo 框架 README 标明 **public modules + optional private stack**（Supervisor / Hub 等不在公开组织内）。

---

## 产品与软件分层（导航）

| 分层 | 代表仓 / 入口 | 说明 |
|------|---------------|------|
| 执行器基座 | [DynamixelSDK](https://github.com/ROBOTIS-GIT/DynamixelSDK) | Protocol 1.0/2.0 多语言 SDK |
| 教育 / 经典平台 | [turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3)、[OpenCR](https://github.com/ROBOTIS-GIT/OpenCR)、[ROBOTIS-OP3](https://github.com/ROBOTIS-GIT/ROBOTIS-OP3) | ROS 教学与小型人形 |
| 机械臂 / 手 | [open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator)、[robotis_hand](https://github.com/ROBOTIS-GIT/robotis_hand) | OMY/OMX 与灵巧手 ROS 2 |
| Physical AI 硬件 | [ai_worker](https://github.com/ROBOTIS-GIT/ai_worker)、[ai_sapiens](https://github.com/ROBOTIS-GIT/ai_sapiens) | 半人形 AI Worker（FFW）与人形 AI Sapiens K1 |
| Cyclo 框架索引 | [cyclo](https://github.com/ROBOTIS-GIT/cyclo) | 模块表：Manager / Intelligence / Control / Lab |
| 学习与仿真 | [cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab)、[cyclo_mjlab](https://github.com/ROBOTIS-GIT/cyclo_mjlab)、[robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie) | Isaac Lab RL/IL；**K1 专用 mjlab 路径**；MuJoCo 资产 |
| LeRobot × Zenoh | [zenoh_ros2_sdk](https://github.com/ROBOTIS-GIT/zenoh_ros2_sdk)、[lerobot_robot_ros2_zenoh](https://github.com/ROBOTIS-GIT/lerobot_robot_ros2_zenoh) | 无 ROS 2 安装下的 LeRobot 机器人接口（α） |
| 运维 / 文档 | [cyclo_manager](https://github.com/ROBOTIS-GIT/cyclo_manager)、[docs](https://github.com/ROBOTIS-GIT/docs) | FastAPI+Next.js 容器编排；[docs.robotis.com](https://docs.robotis.com/) 源仓 |
| 真机 AI 工具 | [physical_ai_tools](https://github.com/ROBOTIS-GIT/physical_ai_tools)、[cyclo_intelligence](https://github.com/ROBOTIS-GIT/cyclo_intelligence) | LeRobot+ROS 2 界面；BT+VLA 全栈 |
| 运动控制 | [cyclo_control](https://github.com/ROBOTIS-GIT/cyclo_control) | WBC / 运动控制器核心与 ROS 包装 |
| 重定向 | [soma-retargeter](https://github.com/ROBOTIS-GIT/soma-retargeter) | SOMA BVH → 人形（G1 / AI Sapiens 资产） |

---

## 已升格 / 复用 wiki 节点

| 主题 | wiki |
|------|------|
| 组织 hub | [robotis.md](../../wiki/entities/robotis.md) |
| AI Worker | [robotis-ai-worker.md](../../wiki/entities/robotis-ai-worker.md) |
| AI Sapiens | [robotis-ai-sapiens.md](../../wiki/entities/robotis-ai-sapiens.md) |
| Cyclo Lab | [cyclo-lab.md](../../wiki/entities/cyclo-lab.md) |
| cyclo_mjlab（K1 / mjlab） | [robotis-cyclo-mjlab.md](../../wiki/entities/robotis-cyclo-mjlab.md) |
| Physical AI Tools | [robotis-physical-ai-tools.md](../../wiki/entities/robotis-physical-ai-tools.md) |
| Cyclo Intelligence | [cyclo-intelligence.md](../../wiki/entities/cyclo-intelligence.md) |
| Dynamixel SDK | [dynamixel-sdk.md](../../wiki/entities/dynamixel-sdk.md) |
| MuJoCo Menagerie | [robotis-mujoco-menagerie.md](../../wiki/entities/robotis-mujoco-menagerie.md) |
| TurtleBot3 | [turtlebot3.md](../../wiki/entities/turtlebot3.md) |
| OpenMANIPULATOR 线 | [robotis-open-manipulator-line.md](../../wiki/entities/robotis-open-manipulator-line.md) |
| OP3 / THORMANG3 | [robotis-op3.md](../../wiki/entities/robotis-op3.md) · [robotis-thormang3.md](../../wiki/entities/robotis-thormang3.md) |

单仓 `sources/repos/<name>.md` 的「沉淀到 wiki」字段指向合并页或标注仅归档。

---

## 为什么值得保留

- **Dynamixel → 教育平台 → Physical AI** 完整开源链：从舵机协议到半人形/人形 ROS 2 与 LeRobot/Isaac Lab。
- **Cyclo** 把厂商 RL/IL（`cyclo_lab`）、真机控制（`cyclo_control`）、BT+VLA（`cyclo_intelligence`）收成可导航模块表，便于与宇树 / 云深处厂商 Lab 对照。
- 站内已有零散 ROBOTIS 实体，缺少 **组织级导航 hub**；本页补齐选型入口。

---

## 对 wiki 的映射

- **wiki/entities/robotis.md**（新建）— 组织 hub 与 Cyclo / Physical AI 地图
- **wiki/entities/robotis-ai-worker.md** 等主线仓 — 独立详情节点
- **已有** turtlebot3 / open-manipulator / op3 / thormang3 / cyclo-intelligence — 回链 hub
- **wiki/entities/robot-lab.md** / 厂商 Lab 页 — 交叉对照 cyclo_lab
