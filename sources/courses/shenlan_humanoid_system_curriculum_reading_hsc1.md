# 深蓝学院《人形机器人系统》课程大纲通读

> 来源归档（原文通读 · 支撑 `wiki/entities/humanoid-system-curriculum.md`）

- **标题：** 人形机器人系统 — 理论与实践（深蓝学院）课程大纲
- **类型：** course / curriculum
- **原文：** [shenlan_humanoid_system_theory_practice.md](./shenlan_humanoid_system_theory_practice.md)（具身智能研究室公众号大纲整理；课程页 <https://www.shenlanxueyuan.com/course/802/task/33927/show>）
- **一句话说明：** 以 Unitree G1 为平台的 8 章 + 实践课程大纲：行走 RL、LiDAR 建图定位、A*/DWA、TARE/FAR 探索、RealSense+YOLO 足球感知、EKF 融合定位、大模型/VLN/NaVid。
- **核对范围：** 逐行通读大纲归档全文（8 章 37 个小节与实践作业表、「对 wiki 的映射」表、外部入口），并与 wiki 页的章节映射表逐节比对一致。课程页 2026-11 抓取返回登录提示页，未能获取课程视频 / 讲义 / 作业细节；本页只依据大纲层面信息，不含讲授内容的技术细节。
- **通读：** 已完成

## 核心摘录（均出自大纲）

- **主线：** G1 软硬件 → RL 双足行走（PPO、Sim2Real）→ LiDAR 建图与定位（含里程计融合）→ A* 全局规划 + DWA 局部避障 → TarePlanner / FarPlanner 自主探索 → RealSense + YOLO11 感知 → RoboCup 仿真足球（坐标变换、线匹配、EKF）→ 大模型（语音交互、VLN、NaVid）。
- **每章配有实践作业**，如 G1 仿真与运动控制、RL 行走训练、激光建图定位、A*+DWA 避障、仿真探索、YOLO11 检测、仿真足球赛、语音交互导航。
- **开发环境：** 第 5 章探索实践对应 CMU Exploration 环境。

## 对 wiki 的映射

- [`wiki/entities/humanoid-system-curriculum.md`](../../wiki/entities/humanoid-system-curriculum.md)

## 参考来源

- 大纲归档：[shenlan_humanoid_system_theory_practice.md](./shenlan_humanoid_system_theory_practice.md)
