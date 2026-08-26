# ROS2SmolVLA（工业级轻量臂上的本地小型 VLA）

> 来源归档（ingest）

- **标题：** ROS2SmolVLA: Enabling Small Vision-Language-Action Models for Integration into Industrial-Grade Lightweight Robots
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.23320>
- **机构：** 奥格斯堡大学机电一体化教席（University of Augsburg, Chair of Mechatronics）
- **会议：** International Conference on Industry of the Future and Smart Manufacturing 2026
- **项目页：** <https://una-auxme.github.io/en/projects/ros2smolvla/> — 归档见 [`sources/sites/ros2smolvla-una-auxme.md`](../sites/ros2smolvla-una-auxme.md)
- **代码入口：** <https://github.com/una-auxme/ros2smolvla_docker> — 归档见 [`sources/repos/ros2smolvla_docker.md`](../repos/ros2smolvla_docker.md)
- **权重 / 数据：** <https://huggingface.co/una-auxme/ROS2SmolVLA_ur10e_no_joints_crop_pick_place>（同名 dataset 共存）
- **入库日期：** 2026-08-26
- **一句话说明：** 把 Hugging Face SmolVLA（450M）接到 ROS 2 + Universal Robots UR10e：Docker 隔离推理、三相机 LeRobot 采集、笛卡尔速度动作；349 条遥操作 episode 微调后，九场景 pick-and-place 总体 [pick, place, place|pick] = [77.72%, 63.59%, 81.69%]；强调本地/边缘合规，而非通才泛化。

## 核心摘录（MVP）

### 1) 工业约束：云端大 VLA 与桌面臂评测都不够

- **摘录要点：** 柔性/可重构制造需要自适应机器人，但大 VLA 依赖外部算力，带来合规与安全问题；现有小 VLA 多在 SO-101 / OpenMANIPULATOR 等实验室小臂上评测，掩盖了工业轻量臂的控制环、安全、负载与工作空间差异。ROS2SmolVLA 把 SmolVLA 部署到 UR10e，推理走工作站级消费级 GPU（RTX 4080），相机汇聚走 Jetson AGX Orin。
- **对 wiki 的映射：**
  - [ROS2SmolVLA](../../wiki/entities/paper-ros2smolvla.md) — 问题设定与部署边界。
  - [VLA](../../wiki/methods/vla.md) — 轻量本地 VLA 相对云端通才的工程轴。

### 2) ROS 2 接口：笛卡尔速度 + 多相机 + 容器

- **摘录要点：** 自研栈包括 `ros2smolvla_docker`、`ros2smolvla_ur10e_{sim,real}`、`ros2smolvla_interface_camera`、`ros2smolvla_interface_lerobot`。`lerobot-ros` 作 LeRobot↔ROS 2 经纪；动作用笛卡尔 delta / `TwistStamped`，经 Cartesian Motion Controller 跟踪，便于跨本体移植。相机包把顶视、侧视与腕部相机转成 LeRobot 相机接口，可裁剪/旋转。Gazebo 数字孪生与真机共用 topic schema，但验证只用了真机数据。
- **对 wiki 的映射：**
  - [ROS2SmolVLA](../../wiki/entities/paper-ros2smolvla.md) — 架构与运行时序。
  - [LeRobot](../../wiki/entities/lerobot.md) — 训练/推理后端。
  - [ROS 2 基础](../../wiki/concepts/ros2-basics.md) — 中间件胶水层。

### 3) UR10e 拾放：ID 可做、OOD place 与颜色指令脆弱

- **摘录要点：** 349 episode、手柄遥操作；顶视裁到工作区下半以适配每帧 64 visual tokens。微调 `train_expert_only=False`、`compile_model=True`，2e5 step 约 25 h（L40S）。九场景拆 pick / place：ID [78.33%, 72.50%, 92.47%]，OOD [76.56%, 46.88%, 61.22%]，全量 [77.72%, 63.59%, 81.69%]。颜色指令常被绿/蓝纹理偏置压过；黑盒检测失败、蓝盒触发 drop。作者明确：可行性有了，产线鲁棒性不够。
- **对 wiki 的映射：**
  - [ROS2SmolVLA](../../wiki/entities/paper-ros2smolvla.md) — 指标读法与 lessons learned。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 工业拾放语境。

### 4) 开源状态（截至 2026-08-26）

- **摘录要点：** 项目页与论文均指向 GitHub 组织 + Hugging Face。步骤 2.5 核查：**已开源** — Docker 入口仓 Apache-2.0；姊妹仓 `interface_lerobot` / `interface_camera` / `ur10e_real` / `ur10e_sim` 均公开；HF 权重基于 `lerobot/smolvla_base`，数据集同名、约 349 episode。
- **对 wiki 的映射：**
  - [ROS2SmolVLA](../../wiki/entities/paper-ros2smolvla.md) — 工程实践与源码时序图。
  - [ros2smolvla_docker](../repos/ros2smolvla_docker.md) — 复现入口。

## 当前提炼状态

- [x] arXiv 摘要、方法与验证节已对齐摘录
- [x] 项目页与 GitHub / HF 开源状态已交叉核查
- [x] wiki 映射：`wiki/entities/paper-ros2smolvla.md` 新建
