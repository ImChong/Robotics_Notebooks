# NVlabs / cuRobo

> 来源归档（ingest）

- **标题：** cuRobo — CUDA 加速机器人算法库；cuRoboV2 — 面向高自由度系统的动力学感知运动生成
- **类型：** repo + 官方文档站 + 论文（arXiv）
- **组织：** NVIDIA（NVlabs）
- **代码：** https://github.com/NVlabs/curobo
- **文档站：** https://curobo.org/
- **论文（初版技术报告 / ICRA 路线）：** https://arxiv.org/abs/2310.17274 — *cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation*
- **论文（cuRoboV2）：** https://arxiv.org/abs/2603.05493 — *cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots*
- **商业 / 产品集成：** 无碰撞运动规划以 **MoveIt 插件**形式提供：**[Isaac ROS cuMotion](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion)**（官网 2024+ 叙述）；Python 库商业许可需走 NVIDIA Research Licensing。
- **入库日期：** 2026-05-16
- **一句话说明：** **cuRobo** 把 **FK/IK、连续碰撞检测、多种子并行轨迹优化、几何规划、MPPI** 等栈在 **GPU** 上批量化，以「全局运动优化 ≈ 多样本局部优化 + 并行几何种子」在桌面 / Jetson 上追求 **数十毫秒级** 操作臂无碰撞运动生成；**cuRoboV2** 在同一代码基底上强调 **B 样条决策空间 + 逆动力学力矩约束、块稀疏 TSDF/按需稠密 ESDF 感知管线、面向人形/双臂的可扩展运动学–动力学与自碰撞**，把适用范围从典型单臂扩展到 **高自由度整机** 与 **重载可行性**。
- **沉淀到 wiki：** [cuRobo](../../wiki/entities/curobo.md)

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Trajectory Optimization](../../wiki/methods/trajectory-optimization.md) | cuRobo 将 **多轨迹并行 TO + 平滑（jerk/加速度）代价** 作为全局运动生成的核心；V2 用 **B 样条控制点** 作变量并显式 **力矩约束** |
| [Manipulation](../../wiki/tasks/manipulation.md) | 抓取–放置、避障到达等 **笛卡尔目标 + 关节空间可行轨迹** 的工程入口 |
| [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) | V2 论文将 **高自由度无碰撞 IK / 重定向约束** 与下游 **locomotion 策略跟踪误差** 关联叙述 |
| [Crocoddyl](../../wiki/entities/crocoddyl.md) | 经典 **shooting / DDP** 工具链对照；cuRobo 侧重 **GPU 批处理碰撞与并行重启**，问题剖分不同 |
| [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) | 文档与示例大量围绕 **Isaac Sim**、**nvblox** 深度避障；ROS 侧与 **Isaac ROS** 生态衔接 |

---

## 官网与仓库归纳（curobo.org / README 级信息）

1. **能力模块（初版栈）：** 正/逆运动学；机器人–环境 **碰撞**（cuboid、mesh、depth）；**梯度下降 / L-BFGS / MPPI**；几何规划；轨迹优化；**MotionGen** 组合 **IK + 几何规划 + TO**，报告 **~30ms** 量级运动生成叙事；**nvblox** 用于深度流避障。
2. **工程事实：** 公开代码库 **Apache-2.0**；文档站标注 **Preview Release** 并区分 **研究代码** 与 **Isaac ROS cuMotion** 商业插件路径。
3. **版本叙事（摘录）：** 站点 changelog 提及 **ESDF 体素碰撞**、**Isaac Sim 4.x**、**约束规划**、**Grasp API**、**re-timing**、**mimic 关节**、高精度 IK/MotionGen 选项等迭代。

---

## 论文 2310.17274 要点（映射用，非全文转录）

- **问题：** 从 **起始关节状态** 到 **末端笛卡尔目标** 的 **无碰撞运动生成**，表述为 **离散时间轨迹优化** + 关节 **位置/速度/加速度/jerk** 盒约束与 **终端静止**；世界碰撞用 **有符号距离** 的连续碰撞形式化。
- **表示：** 连杆体积用 **球冠包络** 降低碰撞查询成本；世界侧支持 **cuboid / mesh / depth**；碰撞代价在障碍表面附近用 **缓冲带内二次型** 改善条件数。
- **优化：** **多随机种子 + 粒子式粗搜索** 将种子推入好的吸引域后接 **GPU 批处理 L-BFGS**；配套 **近似并行线搜索**；与 **并行几何规划器**（启发式直连 / retract / 类 BIT\* 知情采样 + **并行 steering**）组合为 **全局运动生成管线**。
- **IK：** 在相同内核上报告 **高吞吐 IK / 无碰撞 IK**（论文摘要级数字；以论文 PDF 与复现实测为准）。
- **评测：** **motionbenchmaker**、**mpinets** 等数据集上相对 **Tesseract（OMPL + TrajOpt）** 的成功率与路径长度、时间指标叙事；部分失败归因于 **目标/起点不可行** 或 **cuboid 近似几何** 与数据集圆柱障碍不一致等。

---

## 论文 2603.05493（cuRoboV2）要点（映射用）

- **动机：** 快规划器常 **忽略动力学可行性**；反应式方法在 **高保真深度感知** 与 **严格安全** 间取舍；许多 GPU 方案在 **高 DoF** 上 **IK/TO 不收敛或极慢**。
- **贡献 1 — B 样条 TO：** 以 **均匀三次 B 样条控制点** 为决策变量，隐式 **C² 平滑**；优化目标含 **平滑/长度/能量（与力矩相关）**；硬约束含 **场景/自碰撞、关节状态界、逆动力学力矩界、多连杆 SE(3) 目标**；与 **非静态初值** 用 **虚拟控制点** 处理边界。
- **贡献 2 — 感知：** **块稀疏 TSDF**（depth + 几何双通道 `min` 查询）+ **按需稠密 ESDF**（PBA+ 类传播 + 符号恢复），强调相对 **nvblox 类「仅在已分配块内有距离」** 的 **全工作空间 O(1) 查询** 与 **显存/速度** 叙事；论文给出 **碰撞召回**、更新率等对比叙述。
- **贡献 3 — 可扩展刚体算法：** **拓扑感知运动学 + 稀疏雅可比**、**可微逆动力学**、**map-reduce 自碰撞**，面向 **双臂 / 人形** 的吞吐叙事。
- **评测叙事：** 负载下成功率、**48-DoF 人形无碰撞 IK**、与 **PyRoki** 等对比的 **重定向约束满足率**、对 **locomotion 策略跟踪误差 / 跨种子方差** 的间接收益叙述；另含 **LLM 辅助开发** 的方法学讨论（代码结构可发现性）。

---

## 对 wiki 的映射

- 新建 **`wiki/entities/curobo.md`**：初版与 V2 的一体化实体页 + **MotionGen 管线 Mermaid** + 与 TO/操作/重定向的对照阅读。
- 轻量互链：`wiki/methods/trajectory-optimization.md`、`wiki/tasks/manipulation.md`、`wiki/entities/crocoddyl.md`、`references/repos/utilities.md`。

---

## 外部参考（便于复核）

- Sundaralingam et al., *cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation*, [arXiv:2310.17274](https://arxiv.org/abs/2310.17274)
- Sundaralingam et al., *cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots*, [arXiv:2603.05493](https://arxiv.org/abs/2603.05493)
- [NVlabs/curobo（GitHub）](https://github.com/NVlabs/curobo)
- [cuRobo 文档站](https://curobo.org/)
- [Isaac ROS cuMotion](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion)
