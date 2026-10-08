# HOP 官方项目页：Horizon-Optimal Trajectory Optimization

> RAP Lab 项目页来源归档，核验日期：2026-10-08。

- **项目页：** <https://rap-lab.github.io/research/hop>
- **代码：** <https://github.com/rap-lab-org/public_HOP_horizon_optimal_tutorial>
- **论文：** <https://roboticsproceedings.org/rss22/p186.html>
- **作者 PDF：** <https://rap-lab.github.io/documents/publications/2026_RSS_HOP_MiaomiaoDai.pdf>
- **作者：** Miaomiao Dai、Zhongqiang Ren
- **机构：** 上海交通大学全球学院（Global College），Robotics Autonomy and Planning Lab（RAP Lab）
- **论文节点：** [HOP](../../wiki/entities/paper-hop-horizon-optimal-trajectory-planning.md)
- **代码归档：** [HOP Python tutorial 仓库](../repos/hop_horizon_optimal_tutorial.md)
- **论文归档：** [RSS 2026 论文](../papers/hop_2026_rss_186.md)

## 项目简介

HOP 面向 horizon-optimal control：不仅优化控制输入，还在预设的最大离散时域内选择更合适的规划长度。它将 LQR 的 Riccati 递归重写成线性分式变换（LFT），在反向计算时复用价值函数结构，以线性复杂度求解时变 LQR 的最优时域；再以增广状态空间把方法推广到非线性动力学和非二次代价，形成 HOP-DDP。

## 项目页结果

- 对 brute-force horizon sweep：作者报告 HOP 在所测线性和非线性系统中得到相同解，最快约 40×。
- 对 shift-horizon baseline：作者称运行时间相近，但非线性实例中 HOP 通常找到更好的局部解，成本最多降低约 7%。
- 对时不变 LQR 近似：非线性系统上项目页报告 HOP 得到更优解。

以上数值来自作者所列实验，不应外推为所有系统、时域上限和硬件上的固定速度或全局最优保证。

## 开放材料与范围

- **代码：** 项目页提供 GitHub 链接，公开仓库含 HOP-LQR/HOP-DDP 的 Python tutorial 与可运行脚本。
- **示例：** double integrator、quadrotor toy systems；运行输出轨迹、成本/候选时域和耗时，支持可选绘图。
- **数据：** 页面未列独立数据集。
- **许可：** 核验时仓库根目录未发现 LICENSE 文件；可访问代码不等于已明确授予再分发或商用许可。
- **真机/仿真接口：** tutorial 中没有机器人 IO 或真实部署入口，不应视为完整机器人规划器。

## 原始资料

- 项目页：<https://rap-lab.github.io/research/hop>
- 论文录：<https://roboticsproceedings.org/rss22/p186.html>
- RSS 2026 官方接收列表：<https://roboticsconference.org/2026/program/papers/>
- GitHub 代码：<https://github.com/rap-lab-org/public_HOP_horizon_optimal_tutorial>
