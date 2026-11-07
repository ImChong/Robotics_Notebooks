# cuRoboV2-Fast（arXiv:2611.06913）论文原文归档

> 来源归档（paper）；以 NVIDIA 原论文为主，替换旧版仅依据公众号的摘录。

- **标题：** cuRoboV2-Fast: Fast Torque-Limited Collision-Free Motion Planning
- **作者：** Balakumar Sundaralingam、Adithyavairavan Murali、Stan Birchfield
- **机构：** NVIDIA（英伟达）
- **类型：** paper
- **版本：** arXiv:2611.06913v1，2026-11-04（13 页）
- **arXiv：** <https://arxiv.org/abs/2611.06913>
- **PDF：** <https://research.nvidia.com/labs/lpr/curobov2-fast/paper.pdf>
- **项目页：** <https://research.nvidia.com/labs/lpr/curobov2-fast/>
- **代码：** <https://github.com/NVlabs/curobo>
- **核对范围：** 原始 PDF 第 1–13 页；引言与相关工作、III 方法（GPU RRT、B-spline、TimeOpt、计算内核、动力学、抓取）、IV 实验（Fig. 10–19）、V 讨论及参考文献；另核对 NVIDIA 项目页与官方仓库源码入口。
- **通读：** 已完成
- **入库日期：** 2026-11-06；原文复核：2026-11-08
- **一句话说明：** 将快速无碰撞路径生成与负载感知、力矩受限的执行时间优化结合，兼顾规划延迟与执行周期。

## 原文方法与证据

1. **初值：** 由 IK 得到目标关节解；先尝试直线 seed，困难情况调用双向并行 GPU RRT。RRT 按轮并行扩展、做边约束/碰撞检测并截取有效前缀，找到连接路径后短化并拟合 B 样条（论文 III-A、Fig. 2）。
   - **对 wiki 的映射：** [主节点·流程总览](../../wiki/entities/paper-curobov2-fast.md#流程总览)。
2. **几何：** 三次 B 样条以少量控制点表达关节路径；每个采样点由四个相邻控制点决定，梯度通过局部 gather 和 warp reduction 回传。与 cuRoboV1 相比，收敛实验在 25 次初始迭代达到 93% 验证成功率，前代约 64%，需要 125 次才能达到相当成功率（III-B、IV-B、Fig. 3、11）。
   - **对 wiki 的映射：** [主节点·核心原理](../../wiki/entities/paper-curobov2-fast.md#核心原理)。
3. **时间：** TimeOpt 保持几何轨迹不变，以三次样条表示路径进度的平方速度；在采样速度、加速度、jerk、力矩限值下优化时长。局部四控制点关联使牛顿系统呈带状；使用 primal-dual predictor–corrector、Cholesky 因子复用、回溯步长（III-C、Fig. 4–5）。
   - **对 wiki 的映射：** [主节点·核心原理](../../wiki/entities/paper-curobov2-fast.md#核心原理)。
4. **机器人计算：** 并行 FK、每连杆 wrench 汇聚、逆动力学解析反向传播，以及自碰撞/环境 swept bounds 剪枝；运行时更新负载质量、质心、惯量，不需要重新编译内核（III-D/E、Fig. 6–8）。
   - **对 wiki 的映射：** [主节点·工程实践](../../wiki/entities/paper-curobov2-fast.md#工程实践)。
5. **抓取：** IK 排序候选 grasp；单次抓取规划包含 pregrasp、沿抓取轴 approach 和 lift，放置转移单独规划。真机使用双目、SceneFields、TSDF/ESDF、GraspGenX 和 cuRoboV2-Fast（III-F、IV-G、Fig. 9、17）。
   - **对 wiki 的映射：** [主节点·流程总览](../../wiki/entities/paper-curobov2-fast.md#流程总览)。

## 原文评测口径（不可跨任务混比）

| 实验 | cuRoboV2 | cuRoboV2-Fast | 条件及说明 |
|---|---:|---:|---|
| 3 kg 全链路规划平均时延 | 65.13 ms | **4.43 ms** | RTX 5090，14.7×；Fig. 14 |
| 3 kg 全链路规划平均时延 | 602.27 ms | **25.99 ms** | Jetson AGX Orin，23.2×；Fig. 14 |
| GPU RRT 路径初始化 | cuRoboV2 PRM 8.49 ms | **0.305 ms** | 2575 查询、粗碰撞检查；RRT 98.68% 初始化成功，Fig. 10 |
| 真实 YAM 抓放循环 | 9.07 s | **5.93 s** | RTX 6000 Ada，缩短 34.6%；Fig. 18–19 |
| 真实 YAM 手臂运动 | 6.78 s | **3.75 s** | 缩短 44.8%；Fig. 18 |

- 固定 25 次初始迭代时，在 93 条共同有效的轨迹上加 TimeOpt，比仅用固定时步再精修的运动时长**额外降低 15.7%**（Fig. 11）；该数据不是全系统吞吐提升。
- 3 kg RTX 5090 比较中，VAMP + TOPPRA 的规划时延可比本方法更短，但成功率更低，且其计时**未包含 IK**（IV-C）。
- 实际完整抓放周期里规划约 **1.0%**、手臂运动 63.1%、夹爪 25.8%、其他 10.1%（53 条已完成序列，Fig. 19）。因此 14.7× 规划加速不等于 14.7× 真机吞吐加速。
- 实验集中在特定 Panda/YAM 任务、场景和载荷，不能推导对任意自由度、负载或硬件的普适 4.4 ms 保证；时间与力矩约束主要按路径采样验证，部署仍需硬件安全审查。

## 开源与源码核验

- **状态：已开源**，官方 [NVlabs/curobo](https://github.com/NVlabs/curobo)，Apache-2.0；README 与 v0.9.0 更新日志对应 2026-11-04 的 cuRoboV2-Fast。
- 已阅读 Python 公开入口 `curobo/motion_planner.py`、实现 `curobo/_src/motion/motion_planner.py`，以及 `curobo/_src/solver/solver_timeopt.py`、`curobo/_src/motion/motion_trajectory_timeopt.py`；运行时序只标注这些实有调用，不声称在本地 GPU 跑通过。
- [源码归档](../repos/curobov2_fast.md) · [官方项目页归档](../sites/curobov2-fast.md)。

## 对 wiki 的映射

- [cuRoboV2-Fast 独立论文主节点](../../wiki/entities/paper-curobov2-fast.md)。
- [具身智能小站 11-06 论文阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)（仅作为二手发现线索，性能结论以上述论文为准）。
