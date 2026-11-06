# LeRobot 官方 README 通读

> 来源归档（原文通读 · 支撑 `wiki/entities/lerobot.md`）

- **标题：** LeRobot README（huggingface/lerobot，main 分支）
- **类型：** repo / readme
- **原文：** <https://github.com/huggingface/lerobot/blob/main/README.md>（2026-11 通过代理取得 raw 全文，201 行）
- **一句话说明：** Hugging Face 的 PyTorch 机器人学习库：统一 `Robot` 接口、LeRobotDataset 格式、多类策略实现、`lerobot-eval` / `lerobot-rollout` 评测与部署，以及按包名自动发现的第三方硬件插件。
- **核对范围：** 通读 README 全文（Quick Start、Robots & Control、LeRobot Dataset、SoTA Models 表、Inference & Evaluation、Third-Party Hardware、Resources、Citation、Contribute）。README 链接的官方文档各章节（安装、硬件、各策略 README、inference 文档）和源码未逐篇阅读，本归档不核对它们的细节；ICLR 2026 论文（arXiv:2602.22818）仅读到 README 中的引用信息，未读全文。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **定位：** 在 PyTorch 中提供真机机器人所需的模型、数据集与工具，降低门槛、共享数据集与预训练模型。四点卖点：硬件无关的 Python 接口（从 SO-100 低成本臂到人形）、标准化可流式的 LeRobotDataset（Parquet + MP4 或图片，托管于 HF Hub）、可迁移到真机的 SOTA 策略、开源生态支持。
- **安装：** `pip install lerobot` + `lerobot-info`；基础安装轻量，按需加 extras，如 `lerobot[dataset]`、`lerobot[training]`、`lerobot[core_scripts,feetech]`（SO-101 采集）。
- **Robot 接口：** 统一 `Robot` 类解耦控制逻辑与硬件：`connect()` → `get_observation()` → `model.select_action(obs)` → `send_action(action)`。原生硬件：SO101、OpenARM、LeKiwi、Unitree G1、reBot B601、Koch、HopeJR、OMX、EarthRover、Reachy2，以及手柄、键盘、手机等遥操作设备；可自行实现 Robot 接口接入自有机器人。
- **LeRobotDataset：** 同步 MP4 视频（或图片）存视觉，Parquet 存状态 / 动作；工具支持删 episode、按索引或比例切分、增删特征、合并数据集；`LeRobotDataset("lerobot/aloha_mobile_cabinet")` 自动处理视频解码。
- **策略（SoTA Models 表）：** 纯 PyTorch 实现，覆盖模仿学习（ACT、Diffusion、VQ-BeT、Multitask DiT）、强化学习（HIL-SERL、TDMPC，QC-FQL 标 coming soon）、15 个 VLA、7 个世界模型、3 个奖励模型（SARM、TOPReward、Robometer）。训练入口 `lerobot-train --policy.type=act --dataset.repo_id=...`；可自建策略并推到 Hub；GPU / 内存需求见官方 Compute Hardware Guide。
- **评测与部署：** `lerobot-eval` 在 LIBERO、MetaWorld 等基准上仿真评测；真机运行策略用 `lerobot-rollout`（见 Policy Deployment 文档）；自定义仿真环境经 EnvHub 从 Hub 分发。
- **第三方硬件：** 社区插件（xArm、UR5e、Franka、Piper、WidowX、ARX5、YAM、GELLO、SpaceMouse、Meta Quest、ROS 2 桥、触觉 / 深度相机等）按包名前缀 `lerobot_robot_` / `lerobot_teleoperator_` / `lerobot_camera_` 自动发现，装上即可在 `lerobot-record --robot.type=... --teleop.type=...` 中使用。
- **资源与引用：** 官方文档、Discord、X、Robot Learning Tutorial（Space）、T-Shirt Folding 实验、LeLab（浏览器端遥操作 / 校准 / 采集 / 训练）；引用主仓库，学术引用为 ICLR 2026 论文 "LeRobot: An Open-Source Library for End-to-End Robot Learning"（arXiv:2602.22818）。

## 对 wiki 的映射

- [`wiki/entities/lerobot.md`](../../wiki/entities/lerobot.md)

## 参考来源

- README：<https://github.com/huggingface/lerobot/blob/main/README.md>
