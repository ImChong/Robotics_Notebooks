# NVIDIA Isaac ROS 5.0 — Agentic, Open Source Robotics（博客）

> 来源归档（ingest）

- **标题：** NVIDIA Isaac ROS 5.0 Advances Agentic, Open Source Robotics Development
- **类型：** blog
- **URL：** <https://blogs.nvidia.com/blog/isaac-ros-5-0-agentic-open-source-robotics/>
- **发布日期：** 2026-09-22（ROSCon Toronto）
- **入库日期：** 2026-09-24
- **一句话说明：** Isaac ROS **5.0** 发布：ROS Lyrical / Ubuntu 24.04、**agent-ready** 文档与 manipulation/setup **skills**、FoundationStereo 微调 skill、FoundationPose 推理加速；生态伙伴案例；**ROBOTIS AI Worker** 集成 Isaac ROS **GPU 感知 + cuMotion 视觉引导操作**。

## 核心摘录（MVP）

### 1) 版本与 agentic 开发

- **要点：** 5.0 面向「人类 + AI agent 共建机器人」；支持 **ROS Lyrical**、与 **Open Source Robotics Alliance** 贡献的跨硬件数据接口；CUDA 作为 GPU 加速示例。
- **Skills：** setup / manipulation 可复用工作流；FoundationStereo **微调 skill**；FoundationPose **agent-ready 推理库**（宣称最高 **5.5×** 更快）；pick-and-place 独立 skill。
- **对 wiki 的映射：** [nvidia-physical-ai-toolchain-technology-map](../../wiki/overview/nvidia-physical-ai-toolchain-technology-map.md)

### 2) 生态与部署

- **要点：** Jetson **Orin Nano → Thor** 可扩展；Magna、Universal Robots AI Accelerator、MenteeBot、FieldAI 等客户叙事；**Ekumen** 案例：**isaac_ros_cumotion** 在 GPU 上 **~2–5 ms** 为仓储臂规划无碰路径。
- **对 wiki 的映射：** [cuRobo](../../wiki/entities/curobo.md)、[MoveIt 2](../../wiki/entities/moveit2.md)

### 3) ROBOTIS AI Worker × cuMotion（Customer Story）

- **原文要点：** 「ROBOTIS, which builds the developer-friendly ROS-based TurtleBot3, is integrating Isaac ROS into its **AI Worker** robot, using GPU-accelerated object perception to enable vision-guided manipulation tasks including picking, placing and alignment.» 配图说明：**ROBOTIS performs object manipulation tasks using NVIDIA Isaac ROS CuMotion.**
- **对 wiki 的映射：**
  - [robotis-ai-worker-isaac-cumotion](../../wiki/entities/robotis-ai-worker-isaac-cumotion.md)
  - [robotis-ai-worker](../../wiki/entities/robotis-ai-worker.md)

## 开源核查

- 博客声明：**Isaac ROS 5.0 is free and open source**；入口 [GitHub](https://github.com/NVIDIA-ISAAC-ROS)（以官方为准）。
