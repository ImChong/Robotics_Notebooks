---
type: paper
tags: [robotic-servo, dynamic-imitation, visual-servoing, trajectory-generation]
status: complete
updated: 2026-10-13
arxiv: "2609.14589"
project_id: robotic-servo-dynamic-imitation
summary: "arXiv v1 来源归档：收录 DCT 动态轨迹约束、控制器链路、假设和 Kinova 真机数据，并映射到 wiki 实体。"
---

# Robotic Servo Tracking of Moving Targets with Dynamic Imitation Constraints — 来源摘录

- **类型：** 论文（arXiv 预印本；作者公告称接收 IROS 2026）
- **作者：** Yazhe Luo, Sipu Ruan, Yifei Li, Diansheng Chen
- **机构：** Beihang University；Taiyuan University of Technology；Hunan Intelligent Rehabilitation Robot and Auxiliary Equipment Engineering Technology Research Center（作者具体署名依人而异）
- **来源：** [arXiv:2609.14589](https://arxiv.org/abs/2609.14589)，[HTML v1](https://arxiv.org/html/2609.14589v1)
- **版本日期：** v1 发布于 2026-09-13
- **值得保留：** 将视觉伺服从即时误差反馈拓展为动态变形示教轨迹约束，并提供移动目标跟踪分析与真机比较数据。
- **代码核查：** 截至 2026-10-13，未找到论文链接的官方项目页或可运行代码仓库；不推断未来发布状态。

## 核心摘录

### 1. 动态约束轨迹表示

论文提出 Dynamic Constrained Trajectory (DCT)，结合时间缩放以及解耦的形状/振幅调制。示教运动用高斯基函数和 Locally Weighted Regression 表达；目标姿态更新时在线更新广义轨迹，并在保持进度单调条件下适配变化速度与中间路径。

**对 wiki 的映射：** [论文实体](../../wiki/entities/paper-robotic-servo-dynamic-imitation.md)「问题与方法」「流程总览」；[Visual Servoing](../../wiki/methods/visual-servoing.md) 增补轨迹级约束对比。

### 2. 自适应跟踪点选择

每轮通过到终点距离的单调准则和相邻控制点一致性选取 DCT 上的下一动态跟踪点，减少来回摆动；路径中间形状可按动态约束改变，而非死跟原始示教每一点。

**对 wiki 的映射：** [论文实体](../../wiki/entities/paper-robotic-servo-dynamic-imitation.md)「问题与方法」「工程实践与适用边界」。

### 3. 视觉伺服和机械臂控制链

控制器可用连续目标位姿差分，或多个关键点对应关系，得到末端 6D 速度；随后做坐标变换并经机器人雅可比求关节速度。收敛分析假定目标运动有界；实验系统为 Kinova + RealSense，100 Hz 控制。

**对 wiki 的映射：** [论文实体](../../wiki/entities/paper-robotic-servo-dynamic-imitation.md)「流程总览」「评测与结果」；界定为依赖感知输入的轨迹适配方法，而非端到端感知策略。

### 4. 论文报告的真机结果

在 20 个场景、表格标记 Att.=150 的设置中，本文报告 7.0 ± 0.8 秒平均耗时、4.0 ± 1.2 mm 位姿误差、87% 成功率和 0 次超调/振荡。基线 RTMPose4VSG 为 6.4 ± 1.0 秒、8.0 ± 2.1 mm、83%、2 次；TSCTO 为 8.3 ± 1.1 秒、6.0 ± 1.8 mm、79%、1 次。该结果反映论文设置，不等价于独立复现。

**对 wiki 的映射：** [论文实体](../../wiki/entities/paper-robotic-servo-dynamic-imitation.md)「评测与结果」「结论」；保留适用边界与实验平台。

## 解释边界

- 方法需要目标位姿差分或可跟踪的关键点对应关系，不是从原始像素端到端学习。
- 收敛论证包含目标运动有界等条件；硬件结果来自 Kinova 与 RealSense 设置。
- 截至入库日未找到官方可运行代码链接；无代码运行图。
- Att.=150 与 Scen.=20 按表格原字段记录，不自行解读为独立试验总数。
