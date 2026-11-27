# HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory

- **作者：** Xiaolin Zhou, Liu Liu, Tingyang Xiao, Wei Feng, Fa Fu, Xinrui Meng, Xinjie Wang, Jialiang Han, Boyang Yu, Yun Du, Wei Sui, Zhizhong Su
- **机构：** Horizon Robotics, D-Robotics Robotics
- **版本 / 载体：** arXiv:2606.23565v1 [cs.RO]（2026-06-22 提交，arXiv HTML 全文）；官方项目页（https://horizonrobotics.github.io/robot_lab/holoagent）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2606.23565)（摘要页 [arXiv:2606.23565](https://arxiv.org/abs/2606.23565)）；[官方项目页](https://horizonrobotics.github.io/robot_lab/holoagent)
- **项目页 / 代码：** 已开源（部分核心组件持续迭代中） — [HorizonRobotics/HoloAgent](https://github.com/HorizonRobotics/HoloAgent)（`main` 分支，README 勾选完成 HoloAgent-0 与 FSR-VLN 代码发布，已包含 `agentic_robot/agentOS/` 技能框架与长程沙盒执行、`agentic_robot/chatbot/g1/` 语音交互及 `agentic_robot/core/src/fast_livo/` 建图定位源码；`HoloAgent-1` 规划中的 HoloBrain、HoloMotion 集成、主动探索 Agent-Loop 与纯视觉自主导航仍列为待更新项）；数据集 [HorizonRobotics/fsrvln_datasets](https://huggingface.co/datasets/HorizonRobotics/fsrvln_datasets)
- **核对范围：** 通读 arXiv HTML 全文 Abstract、§1 Introduction、§2 Embodied AgentOS、§3 Skill Layer、§4 Memory Layer、§5 Experiments（含 Table 1、Table 2、Table 3 及 Fig. 1–8 图注）、§6 Related Work、§7 Conclusion、§8 Future Work 与 References；同时核对官方项目页全文及 GitHub 仓库根目录 README 与采样文件列表。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory](../../wiki/entities/paper-agentic-2606-23565.md)

### 1. 问题定义与系统动机（§1–§2）
- 论文指出数字世界 LLM 智能体与物理机器人之间存在显著的“具身鸿沟（embodiment gap）”：物理执行连续、依赖具体本体、充满不确定性且受安全约束，物理技能不像软件 API 那样具有干净的输入输出与可逆副作用。
- HoloAgent-0 的核心思路是用系统级闭环运行时 **Embodied AgentOS** 组织异构底层模型与控制器，形成三个耦合层：**Embodied AgentOS**（任务规划、资源调度、执行监控、澄清与重规划）、**Memory Layer**（持久 3D 空间记忆与时序执行记忆）、**Skill Layer**（类型化、可监控的具身技能调用接口），通过 **ROS2 指令/状态总线（command/status bus）** 串联。
- 四个设计原则：Closed-loop first（观察-检索-动作-监控循环）、Memory-centric、Typed skill interface、Observable by default。运行时内部划分为 AgentOS 层、Skill 层、Memory 层与 Monitoring & Verification 层。

### 2. 类型化技能层与核心后端（§3）
- 每个技能调用声明命令名、类型化参数、前置条件、目标引用与预期效果；后端向上发布结构化执行轨迹（progress、success、failure mode、confidence、latency、recoverability），使失败成为可重规划证据而非隐式布尔值。
- 技能家族包括：
  - **交互与感知技能**：语音接口封装 ASR/对话状态/TTS（`listen`、`speak`）；开放词汇感知暴露 `detect`、`localize`、`verify`。
  - **HoloNavi（空间导航与探索）**：遵循 FSR-VLN 的快慢推理思想，先由 LLM 解析指令并在 HMSG 上通过层级 CLIP 特征做快速匹配，再送入 VLM 进行在线视觉验证与局部视角调整；若记忆陈旧或验证失败，则按预期信息增益、语义相关性、可通行性与安全约束评分触发主动前沿探索并刷新局部记忆。
  - **HoloBrain（操作 VLA 后端）**：执行 `pick`、`place`、`open`、`handover`、`push`、`fold` 等技能，并上报目标缺失、目标移动、抓取失败、不可达位姿、碰撞风险与低置信度等状态事件。
  - **HoloMotion（人形全身运动控制）**：支持运动跟踪（motion-tracking，如挥手、鞠躬、握手、跳舞）与速度跟踪（velocity-tracking，行走、转向、停止、恢复）两种模式，并与 HoloNavi/HoloBrain 组合。
  - **跨本体协同**：不同机器人共享统一 3D 记忆与类型化状态总线，由 AgentOS 按能力、位置、可用性与安全状态绑定任务并转移阻塞职责。

### 3. 时空记忆与局部增量更新（§4）
- **几何记忆（§4.1）**：统一几何接口支持两种定位建图后端：
  - LiDAR 后端：融合 LiDAR、IMU 与相机观测（如 FAST-LIVO）。
  - 纯视觉后端：**GeoFlow-SLAM++**（面向 $N \ge 3$ 校准多相机），用 3D 基础模型 $\Phi_{\theta}$ 预测稠密深度并按外参反投影至机体坐标系，结合多视角两阶段光流跟踪、视觉重投影+点面+IMU 联合优化，以及多相机 BoW 聚合与 BA 重定位。
- **语义记忆（§4.2）**：在几何记忆上叠加开放词汇 3D 语义建图。对 SAM2 生成的每个 2D 掩码提取三组 SigLIP 描述符（全关键帧 $\mathbf{d}_0$、掩码区域 $\mathbf{d}_1$、最小外接框 $\mathbf{d}_2$），按逐维加权平均融合 $\mathbf{d}=\sum_{i=0}^{2}\mathbf{w}_i\odot\mathbf{d}_i$；通过将既有 3D 实例 $\mathcal{V}_{t-1}$ 投影到当前视角并与当前掩码计算 IoU 完成持久实例关联与新建。
- **HMSG（§4.3）**：复用 FSR-VLN 的四层结构（floor / room / view / object），其中 view 层以位姿与视觉证据连接几何记忆与 VLM 推理，实现由粗到细检索与剪枝。
- **时序记忆与动态更新（§4.4–§4.5）**：记录目标与计划状态、执行与恢复轨迹以及结果摘要。由冲突观测、移动/移除物体的技能结果或用户纠偏触发，仅局部重定位并刷新受影响的几何体素、实例与 HMSG 子图，无需重建全图。

### 4. 定量实验与定性演示（§5）
- **硬件平台**：Unitree G1 人形机器人、R1 人形机器人、轮式双臂移动操作机器人。
- **长程导航（Table 1 & Table 2）**：
  - 仿真 HM3D-ObjNav（沿用 MSGNav 协议）：HoloAgent-Nav (w AgentOS loop) 达到 **82.6% SR** 与 **42.8% SPL**，高于 FSR-VLN slow-reasoning（80.8% / 41.0%）、FSR-VLN fast-matching（72.1% / 36.9%）、MSGNav（74.1% / 33.4%）、WMNav（72.2% / 33.3%）、DORAEMON（66.5% / 20.6%）、VLFM（62.6% / 31.0%）与 SG-Nav（49.6% / 25.5%）。
  - 真实公寓及数字孪生导航（沿用 FSR-VLN 协议）：HoloAgent-Nav 在 Top-1@1.0m/2.0m/3.0m 均为 **97.70%**，在 Top-5@1.0m/2.0m/3.0m 均为 **98.90%**，高于 FSR-VLN（Top-1: 91.95% / 91.95% / 94.25%；Top-5: 94.25% / 96.55% / 96.55%）、HOV-SG、MobilityVLA 与 OK-Robot。
- **3D 语义建图（Table 3）**：
  - ScanNet 零样本分割：HoloAgent-Memory（Online）取得 mIoU **31.58**、mAcc **45.54**、f-mIoU **47.43**、f-Acc **61.58**。
  - Replica 零样本分割：HoloAgent-Memory（Online）取得 mIoU **29.93**、mAcc **43.60**、f-mIoU **57.00**、f-Acc **65.39**。
- **定性系统演示（§5.4）**：展示了短程全身运动控制、主动找咖啡机、跨机器人协同（导航+跳舞）及长程折叠衣物移动操作；作者明确说明端到端多技能尚无统一可重复定量基准。

### 5. 局限与未来方向（§5.4, §8）
- 端到端操作、全身控制与跨本体协同尚未建立统一标准化评测基准。
- 未来三方向：指令对齐的通用机器人基础模型、全栈人形技能与更广泛本体支持、引入 EmbodiedGen 数字孪生沙盒的智能体代码生成与安全预验证。
