# NVlabs / ScheduleStream

> 来源归档（ingest）

- **标题：** ScheduleStream — 带连续采样算子的规划与调度框架（多臂 TAMPAS）
- **类型：** repo + 项目页 + 论文
- **组织：** NVIDIA（NVlabs）
- **代码：** https://github.com/NVlabs/schedulestream
- **项目页：** https://schedulestream.github.io/
- **论文：** https://arxiv.org/abs/2511.04758 — *ScheduleStream: Temporal Planning with Samplers for GPU-Accelerated Multi-Arm Task and Motion Planning & Scheduling*（ICRA 2026）
- **演示视频：** https://www.youtube.com/watch?v=0LyTPmAXaQY
- **NVIDIA 研究发布页：** https://research.nvidia.com/publication/2026-06_schedulestream-temporal-planning-samplers-gpu-accelerated-multi-arm-task-and
- **入库日期：** 2026-05-30
- **一句话说明：** 开源 **领域无关** 的 ScheduleStream 规划语言与求解器；当前内置应用含 **2D trimesh TAMP**、**motion planning** 等示例；依赖 **custream** 等 CUDA 工具做批处理与可视化；Apache-2.0。
- **沉淀到 wiki：** [ScheduleStream](../../wiki/entities/schedulestream.md)

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Manipulation](../../wiki/tasks/manipulation.md) | 双臂/多臂 **任务分配 + 并行运动调度** 是操作任务从「单臂顺序」升级到 **双手协同节拍** 的规划层入口 |
| [cuRobo](../../wiki/entities/curobo.md) | **GPU 运动生成** 对照；ScheduleStream 编排 **何时哪条臂执行哪段 stream 采样**，连续轨迹可由各 backend 提供 |
| [Trajectory Optimization](../../wiki/methods/trajectory-optimization.md) | stream 中的 **motion** 段常对接轨迹/几何规划，与 TO 方法论文语境相连 |

---

## 仓库归纳（README 级）

1. **定位：** Planning & scheduling with **continuous sampling operations**；动机应用为 **multi-arm TAMP**。
2. **领域无关：** 支持不同机器人后端，亦可用于非机器人应用（以论文/教程为准）。
3. **内置应用（摘录）：**
   - `applications/trimesh2d/tamp.py` — 平面 **trimesh** 碰撞原语上的 TAMP 示例
   - `applications/trimesh2d/motion.py` — 运动规划示例
4. **相关仓库：** **custream**（CUDA stream 工具、TAMP/碰撞/可视化等，README 指向子模块路径）。
5. **安装：** 见仓库 `INSTALL.md`；需 CUDA 生态（具体版本以仓库为准）。

---

## 项目页与媒体（2025-10 更新）

- 主页强调：**无先验指定哪条臂抓哪个物体** 时，系统自动 **分配物体到臂**、**规划安全运动**、**调度无碰撞并行/错峰执行**。
- 三类双臂示意：仅单侧可达 → 必须顺序；双侧可达 → 并行或 **等待 clearance 后启动** 第二臂。

---

## 外部报道（可选交叉）

- NVIDIA Blog（ICRA 2026 机器人研究合集）提及 ScheduleStream 在 **Jetson** 等边缘平台上的 **多臂规划加速** 叙事（以官方博客为准，不复述具体倍数）。
