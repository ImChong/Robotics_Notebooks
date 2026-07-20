# Swarm of Micro Flying Robots in the Wild（Science Robotics, 2022）

> 来源归档（ingest）

- **标题：** Swarm of micro flying robots in the wild
- **类型：** paper / micro UAV swarm / trajectory planning / spatiotemporal optimization / onboard sensing
- **期刊：** Science Robotics, April 2022 — **封面文章**（Volume 7, Issue 66, eabm5954）
- **DOI：** <https://doi.org/10.1126/scirobotics.abm5954>
- **项目代码：** EGO-Planner-v2 — <https://github.com/ZJU-FAST-Lab/ego-planner-v2>（详见下文开源核查）
- **数据集：** <https://doi.org/10.5281/zenodo.5804079>
- **作者：** Xin Zhou†、Xiangyong Wen†、Zhepei Wang、Yuman Gao、Haojia Li、Qianhao Wang、Tiankai Yang、Haojian Lu、Yanjun Cao、Chao Xu*、Fei Gao*（† 同等一作；* 通讯作者）
- **机构：** 浙江大学 FAST-Lab（徐超、高飞实验室）
- **入库日期：** 2026-07-20
- **一句话说明：** 提出掌心级微型 UAV 群 + **时空联合轨迹优化**（MINCO），在无 GPS、无外部设施的竹林中实现 **10 机自主蜂群飞行**，成为无人机群在极度复杂户外环境中的首个 Science Robotics 级实证。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目代码 | [EGO-Planner-v2 (GitHub)](https://github.com/ZJU-FAST-Lab/ego-planner-v2) | 本文完整代码库，GPL v3，700+ stars |
| 数据集 | [Zenodo 5804079](https://doi.org/10.5281/zenodo.5804079) | 论文配套数据集 |
| MINCO 基础 | [ZJU-FAST-Lab/GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER) | MINCO 轨迹表示的原始仓库 |
| 前序工作 | [EGO-Planner (v1/Swarm)](../../wiki/entities/ego-planner-swarm.md) | 单机/多机局部规划器，ESDF + B-spline |
| 多旋翼规划栈 | [多旋翼仿真规划控制栈概览](../../wiki/overview/multirotor-simulation-planning-control-stack.md) | 宏观生态定位 |

## 摘要级要点

- **问题：** 密集森林等杂乱户外环境对单机已难，对机群几乎不可能——狭窄走廊 + 未知环境 + 群体协调碰撞规避，三个约束同时压在一个毫秒级在线规划器上。
- **平台：** 掌心级微型四旋翼（约 30–40 g），搭载 **深度相机 + VIO（视觉-惯性里程计）**，完全机载感知与计算；**无 GPS、无动捕、无外部通讯基础设施**。
- **规划核心——时空联合优化：**
  - **MINCO 轨迹表示**（Minimum-Jerk Continuity Optimization）：多项式分段轨迹，支持同时对轨迹 **形状** 和 **时间分配** 求导并联合优化——相比 MADER/EGO-Swarm 等只优化形状的方案，消除了后行无人机绕路等待的问题。
  - **统一代价函数**：飞行效率 + 静态障碍避碰 + 机间碰撞规避 + 动力学可行性 + 群体协调等，全部纳入同一可微 NLP。
  - **毫秒级求解：** 即使在最拥挤环境下，几毫秒内得到高质量轨迹。
- **去中心化异步规划：** 各无人机独立规划，通过广播共享本机当前计划轨迹用于预测性互避碰；不要求严格同步或集中指挥。
- **可扩展性（Extensibility）：** 在基础规划框架上演示三种任务：swarm formation、interlaced flights、multi-goal tracking——验证框架设计的通用性。
- **仿真与实验：**
  - 仿真：Intel Core i7-10700K，各机独立线程并行，对比 MADER 与 EGO-Swarm（轨迹质量 + 算力四项指标）
  - 实机：竹林实地，10 机同时飞行，无外部设施

## 核心摘录（面向 wiki 编译）

### 1) MINCO 时空联合优化

| 要素 | 传统方案 | 本文方案 |
|------|----------|----------|
| 时间分配 | 固定或独立后处理 | 与轨迹形状 **联合优化**（梯度同步回传） |
| 轨迹表示 | B-spline 或 waypoint 序列 | 分段多项式 + 映射矩阵保证 C^n 连续 |
| 与避碰的关系 | 解耦 → 后行机绕路等待 | 时间弹性 → 通过 **时间膨胀** 完成空间避碰 |

### 2) 去中心化机间避碰

- 每机通过局域无线广播 **本机计划轨迹**（未来一段时间内）；邻机收到后在代价函数中添加 **预测碰撞项**。
- **异步**：各机以自有感知频率触发规划，不需全局同步时钟。
- **对 wiki 的映射：** [`wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md`](../../wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md)

### 3) 开源与复现

- **完整代码**已以 **EGO-Planner-v2**（GPL v3）发布于 GitHub：包含 swarm、formation、tracking 三个 workspace，附 PDF 教程与演示视频。
- 数据集 deposit 于 Zenodo（含飞行轨迹数据）。
- **对 wiki 的映射：** 开源状态写入局限/工程实践节。

## 源码开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| 完整规划代码 | **已开源** | EGO-Planner-v2，GPL v3，GitHub，700+ stars（截至 2026-07-20） |
| 硬件平台代码 | **部分** | 代码库含仿真 playground；硬件配置（自研微型机体）不含 |
| 数据集 | **已开源** | Zenodo 10.5281/zenodo.5804079 |
| 模型权重 | N/A | 纯规划算法，无神经网络权重 |

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md`](../../wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md)**
- 交叉：**[`wiki/entities/ego-planner-swarm.md`](../../wiki/entities/ego-planner-swarm.md)**（EGO-Planner Swarm 单机/多机规划器）
- 谱系：**[`wiki/entities/crazyswarm2.md`](../../wiki/entities/crazyswarm2.md)**（Crazyflie 群体平台）、**[`wiki/entities/quad-swarm-rl.md`](../../wiki/entities/quad-swarm-rl.md)**（强化学习群控）
- 体系：**[`wiki/overview/multirotor-simulation-planning-control-stack.md`](../../wiki/overview/multirotor-simulation-planning-control-stack.md)**
