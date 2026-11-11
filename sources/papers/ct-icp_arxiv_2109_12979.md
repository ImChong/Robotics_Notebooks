# CT-ICP: Real-time Elastic LiDAR Odometry with Loop Closure（arXiv:2109.12979）

> 来源归档（paper）

- **标题：** CT-ICP: Real-time Elastic LiDAR Odometry with Loop Closure
- **作者 / 机构：** Pierre Dellenbach、Jean-Emmanuel Deschaud、Bastien Jacquet、François Goulette（MINES ParisTech, PSL University 机器人中心；Kitware 计算机视觉组）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2109.12979>（读的是 v1，2021-09-27 提交）
- **PDF：** <https://arxiv.org/pdf/2109.12979v1>
- **代码：** 里程计 <https://github.com/jedeschaud/ct_icp>；回环与位姿图在 <https://github.com/Kitware/pyLiDAR-SLAM>（均为论文脚注给出）
- **入库日期：** 2026-11-11
- **核对范围：** 读了 arXiv v1 PDF 全文（摘要、I 引言、II 相关工作、III 里程计公式与局部地图、IV 回环与后端、V 实验含 Table I / II、VI 结论、参考文献列表），公式 (1)–(8) 与两张结果表逐项对过；该 v1 无附录 / 补充材料。图 1–4 仅读图注，未逐图细看像素内容。会议名（任务说明称 ICRA 2022）在 v1 PDF 内没有出现，未能在原文核实。GitHub 仓库 README 与源码因本会话无该仓库访问权限，未核实。
- **通读：** 已完成
- **一句话说明：** 每帧用「起点 + 终点」两个位姿做帧内插值的弹性连续时间 scan-to-map ICP，帧间位姿允许不连续（靠接近约束拉近）；配稠密点云稀疏体素局部地图，另附基于俯视高程图的回环检测与 g2o 位姿图，组成纯激光雷达 SLAM。

## 开源状态

- 论文脚注给出 C++ 里程计仓库与 Python 回环框架 pyLiDAR-SLAM 两个链接（论文自述，仓库内容本次未核实）。

## 核心摘录（均为论文自述，带章节）

1. **帧内连续、帧间不连续（I、III-A）：** 每帧参数为 Ω_b、Ω_e 两个 SE(3) 位姿，点位姿按时间比例 α_i 插值（旋转 slerp，平移线性，式 4–6）；本帧起点位姿不强制等于上一帧终点位姿，只加接近约束 C_loc（式 7）与常速约束 C_vel（式 8），权重 β_l = β_v = 0.001（全部实验相同）。
2. **目标函数（III-A，式 1–3）：** F_ICP + β_l·C_loc + β_v·C_vel；残差为点到平面距离乘平面度权重 a_i（沿用 IMLS-SLAM 的 a2D），法向与平面度用局部稠密地图而非当前帧计算。
3. **求解器（III-A）：** 提供 Ceres 与手写雅可比的单线程 Gauss-Newton 两种；论文称全部实验使用后者。
4. **局部地图（III-B）：** 稀疏体素哈希存稠密点云，驾驶场景体素 1.0 m，高频运动场景 0.80 m；每体素最多 20 点，点间距不小于 10 cm；邻域取 27 个邻接体素内 k=20 近邻。另有「鲁棒配置」：检测快速转向或配准失败后用更保守参数重配，朝向变化 ≥5° 时该帧不入图，代价是运行时间增加。
5. **回环（IV）：** 把 N_map 帧聚合成局部地图，投影为 2D 高程图，提取旋转不变 2D 特征，RANSAC 求 2D 刚体变换，Open3D ICP 精化出 6 自由度约束，g2o 位姿图仅在检测到回环时全局优化。论文明确写：要求传感器运动基本为地面平面运动，且外参使 z 轴对齐地面法向。
6. **结果（V，Table I）：** KITTI-corrected 平均 RTE 0.53%，60 ms/帧；KITTI-raw 0.55%（IMLS-SLAM 0.71%、MULLS 1.41%、pyLiDAR F2M 1.61%）；KITTI-360 0.45%；KITTI-CARLA 0.09%；ParisLuco 1.11%；NCD 0.55%（F2M 2.20%，430 ms）；NCLT 2012-01-08 1.17%（F2M 19.80%）。摘要称 KITTI 官方榜上在公开代码方法中第一，平均 RTE 0.59%，单线程 CPU 每帧 60 ms。
7. **消融（V-B）：** 只用常速模型去畸变（去掉弹性）：KITTI-raw 由 0.55% 升到 0.79%，KITTI-360 由 0.45% 升到 0.60%。
8. **回环表（V-C，Table II）：** 部分序列回环后 ATE 明显下降（KITTI-raw 00：6.22→0.66 m；KITTI-360 00：29.87→1.07 m；ParisLuco：29.16→9.65 m），KITTI-CARLA Town01 与 NCD 01 回环后 ATE 反而略升（0.21→0.26、0.22→0.36）。每次高程图构建与匹配平均 1.1 s，位姿图优化平均 1.2 s，回环跑在主线程，论文称改多线程可达实时。

## 对 wiki 的映射

- [paper-ct-icp](../../wiki/entities/paper-ct-icp.md)
