# From Language to Navigation Goals: A Vision-Language Approach for Semantic Navigation of Mobile Robots Using RGB-D Perception（arXiv:2607.13624）

> 来源归档（ingest）

- **标题：** From Language to Navigation Goals: A Vision-Language Approach for Semantic Navigation of Mobile Robots Using RGB-D Perception
- **短名：** Language-to-Navigation-Goals（RGB-D VLN 框架）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2607.13624>
- **PDF：** <https://arxiv.org/pdf/2607.13624>
- **机构：** 巴勃罗·德·奥拉维德大学（Universidad Pablo de Olavide, UPO）
- **作者：** Jose Martinez Fajardo, Pablo Pueyo, Fernando Caballero, Luis Merino
- **资助：** MCIN/AEI AI-FUSE-ROBOT（SAIA202500X163851SV0）；CIN/AEI COBUILD（PID2024-161069OB-I00）
- **入库日期：** 2026-08-23
- **一句话说明：** ROS 2 模块化框架：远程 VLM 做语义 grounding + RGB-D 投影生成 Nav2 目标，TurtleBot3/Go2 仿真与真机验证。

## 开源状态（步骤 2.5，2026-08-23）

- **宣称将开源 / 待发布**：论文摘要与正文写明 *Code will be released open source upon acceptance*；arXiv v1 无项目页、GitHub 或 Hugging Face 链接。截至入库日 **尚无可运行官方仓库**。

## 核心摘录（面向 wiki 编译）

### 摘录 1：系统架构

- 四模块 ROS 2 管线：User（自然语言）→ Communication（协调）→ Semantic Perception（远程 VLM bbox + RGB-D 几何定位）→ Movement（Nav2 执行）。
- VLM 返回目标 bbox 与自然语言确认（如 “Okay, I will go to the chair”）；深度取 bbox 中心邻域 **最小有效深度**，可加偏移 \(\delta\) 使机器人停在目标前方而非几何中心。

**对 wiki 的映射：** vln、semantic-navigation、ros2、nav2、rgb-d

### 摘录 2：仿真实验

- **实验 1**（仅感知）：Gazebo + TurtleBot3 Waffle，公交站目标，4 种语言表述 × 不同初始位姿；平均定位误差 **0.68 m**（含 \(\delta=0.6\) m 有意偏移）。
- **实验 2**（端到端）：人/邮箱两类目标，6 组指令（含上下文请求如 “I need to send a letter…”）；平均导航误差 **0.70 m**、行程 **6.22 m**、耗时 **26 s**。

**对 wiki 的映射：** turtlebot、semantic-grounding

### 摘录 3：真机验证

- **实验 3**：Unitree Go2 + Intel RealSense RGB-D；微波炉/椅子目标，直接指令与上下文请求；机器人局部坐标系平均定位误差 **0.51 m**（\(\delta=0.5\) m）。

**对 wiki 的映射：** unitree-go2、sim2real

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-language-to-navigation-goals-rgbd.md`](../../wiki/entities/paper-language-to-navigation-goals-rgbd.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体回链
