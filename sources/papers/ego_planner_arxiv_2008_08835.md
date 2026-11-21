# EGO-Planner: An ESDF-Free Gradient-Based Local Planner for Quadrotors（IEEE RA-L / arXiv:2008.08835）

> 论文来源归档（ingest）

- **标题：** EGO-Planner: An ESDF-Free Gradient-Based Local Planner for Quadrotors
- **类型：** paper / motion-planning / trajectory-optimization / esdf-free / b-spline / uav
- **出处：** IEEE Robotics and Automation Letters (RA-L), Vol. 6, No. 2, pp. 478–485, 2021 · arXiv:[2008.08835](https://arxiv.org/abs/2008.08835)
- **PDF：** <https://arxiv.org/pdf/2008.08835>
- **作者：** Xin Zhou, Zhepei Wang, Hongkai Ye, Chao Xu, Fei Gao（ZJU FAST Lab）
- **官方仓库：** <https://github.com/ZJU-FAST-Lab/ego-planner>
- **入库日期：** 2026-11-21
- **通读：** 已完成
- **一句话说明：** 提出无需全局预建欧氏符号距离场（ESDF-Free）的梯度局部轨迹规划器，仅按需对比初始无碰撞引导路径与障碍物表面生成排斥向量 $\{p_{ij}, v_{ij}\}$，经均匀 B 样条 L-BFGS 优化与各向异性曲线拟合时间重分配（Anisotropic Curve Fitting），将单次重规划总计算耗时从 Fast-Planner 的 7.3 ms 降至 **0.81 ms** 且保持实时避障成功率。

---

## 核心摘录（面向 wiki 编译）

### 1) 为什么放弃显式全局 ESDF 更新
- 在机载三维局部规划中，每次感知点云更新都在滚动物体素栅格内重建全区域 ESDF 会占用约 **70%+** 规划计算时间，而大多数被更新体素并不会进入优化轨迹邻域。
- **EGO-Planner 的核心洞察：** 从起点到目标生成一条仅考虑局部碰撞段的简化引导无碰撞路径（Collision-free path），仅针对穿越障碍物的控制点寻找逃逸方向上的最近障碍物点 $p_{ij}$ 及其朝向轨迹外侧的法向单位向量 $v_{ij}$，直接用有符号投影距离 $d_{ij} = (Q_i - p_{ij}) \cdot v_{ij}$ 构造平滑排斥势能，**按需局部计算梯度，零全局距离场维护开销**。

### 2) B 样条优化目标与 L-BFGS 求解
- 均匀 B 样条阶数 $p_b = 3$，凸包性质天然约束速度、加速度和加加速度（Jerk）极值。
- 联合最小化无约束目标函数：$J = \lambda_s J_s + \lambda_c J_c + \lambda_d J_d$，其中 $J_s$ 为最小 Snap/Jerk 三阶弹性带光滑项，$J_c$ 为分段三次平滑碰撞惩罚项，$J_d$ 为速度/加速度/加加速度动力学可行性软罚项；通过无约束拟牛顿法 **L-BFGS** 在亚毫秒内求解。

### 3) 各向异性曲线拟合时间重分配（Anisotropic Curve Fitting）
- 若优化后局部控制点触发动力学速度/加速度越界，EGO-Planner 单独延展局部时间节点间隔并保持轨迹与环境拓扑紧密贴合——用更高的加权惩罚抑制**法向偏离**（防止因拉长时间而在空间上偏出安全廊道重新撞墙），允许适度调整切向推进速度。

### 4) 基准性能实测数据（Table I–III）
- **单次重规划总耗时（Intel i7 / 同等对比）：** Fast-Planner（ESDF 维护 `3.29 ms` + 优化 `4.01 ms`）共 **7.30 ms**；EGO-Planner 仅花费拓扑初始化 `0.15 ms` + B 样条优化 `0.56 ms` + 时间重分配 `0.10 ms` = **0.81 ms**（快约 9 倍），规划总轨迹长度与能耗保持相当水平并已在微型四旋翼丛林实机穿飞验证（飞行速度 > 3 m/s）。

---

## 对 wiki 的映射

- 方法页：[`wiki/methods/smooth-navigation-path-generation.md`](../../wiki/methods/smooth-navigation-path-generation.md)
- 仓储归档：[`sources/repos/ego_planner_swarm.md`](../repos/ego_planner_swarm.md)
