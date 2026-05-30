# ScheduleStream：GPU 加速多臂 TAMP 与调度（arXiv:2511.04758 / ICRA 2026）

> 论文来源归档（ingest）

- **标题：** ScheduleStream: Temporal Planning with Samplers for GPU-Accelerated Multi-Arm Task and Motion Planning & Scheduling
- **类型：** paper / task-and-motion-planning / scheduling / multi-arm / gpu
- **arXiv：** <https://arxiv.org/abs/2511.04758> · HTML：<https://ar5iv.labs.arxiv.org/html/2511.04758>
- **项目页：** <https://schedulestream.github.io/>
- **机构：** NVIDIA Research（Caelan Garrett）、University of Sydney（Fabio Ramos）
- **会议：** IEEE ICRA 2026
- **入库日期：** 2026-05-30
- **一句话说明：** 提出 **ScheduleStream** 规划–调度语言与 **领域无关** 求解器，用 **混合 durative action** 表达可异步启动、持续时间依赖参数的连续采样动作；在 **TAMPAS**（Task and Motion Planning & Scheduling）应用中把 **PDDLStream 式 stream 采样器** 批量化到 **GPU**，使双臂/多臂在 **并行无碰撞运动** 下获得比「单臂串行 TAMP 计划」更短的 **时间表（schedule）**。

## 核心摘录（面向 wiki 编译）

### 1) 问题：TAMP 计划 vs 调度

- **要点：** 经典 **TAMP** 在混合离散–连续空间中高效，但常见输出是 **顺序计划（plan）**——同一时刻通常只有一条臂在动；双臂/人形需要 **schedule**：多臂 **时间重叠** 的碰撞自由执行。
- **对 wiki 的映射：** [`wiki/entities/schedulestream.md`](../../wiki/entities/schedulestream.md)

### 2) ScheduleStream 语言：durative action + stream

- **要点：** 在 **PDDLStream**（有限动作 + **stream** 连续采样）上扩展 **durative action**：可 **异步 start**，**duration** 为动作参数的函数；全栈用 **Python** 声明式与过程式直接对接（相对 PDDL 文件管线）。
- **Stream 例（双臂抓取域）：** `grasps` 生成抓取位姿；`ik_qs` 由 grasp 得关节配置；`motions` 规划两配置间轨迹（时长进入调度）。
- **对 wiki 的映射：** 同上实体页「语言与 stream」节

### 3) 调度归约：durative → start/end 瞬时动作

- **要点：** `eager-stream` 管线中的 **`schedule`** 子程序：给定初始状态 \(I\)、目标 \(\mathcal{G}\)、durative 动作集 \(\mathcal{A}\)，求 **动作实例** \(a(x)\) 及其 **时间表** \(\tau\)。将每个 durative action 编译为 **`a.start` / `a.end`** 两个瞬时动作，在 **顺序规划** 中求 start/end 次序（借鉴 temporal planning，如 FF 类思想无法直接覆盖的调度层）。
- **对 wiki 的映射：** 同上实体页 mermaid「TAMPAS 主干」

### 4) TAMPAS 与 GPU 采样

- **要点：** **TAMPAS** = 把 ScheduleStream 用于多臂 **TAMP + 调度**；在 **sampler**（IK、运动、抓取等）内用 **GPU 批处理** 加速；仿真消融相对「仅代表先验工作的消融」报告 **更高效 schedule**；真机双臂任务演示（物体–臂自动分配、近距并行、错峰通过）。
- **对 wiki 的映射：** [`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md) 双手协调 / 规划路线交叉引用

### 5) 与 cuRobo / 运动规划栈的关系（对照，非论文主张）

- **要点：** ScheduleStream 侧重 **任务层离散选择 + 时间调度 + stream 采样接口**；连续 **无碰撞轨迹** 可由后端 motion planner（仓库示例含 **trimesh2d** 等）提供；与 **cuRobo** 类 GPU 运动生成是 **互补层**（任务调度 vs 单次运动优化）。
- **对 wiki 的映射：** 对照阅读 `wiki/entities/curobo.md`（GPU 运动生成，互补层）

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 项目页、arXiv、GitHub 链接对齐
