# SwarmNxt: Open-source Software-Hardware Platform for Fast and Agile Aerial Swarms

> 来源归档（ingest）

- **标题：** SwarmNxt: Open-source Software-Hardware Platform for Fast and Agile Aerial Swarms
- **类型：** paper / aerial swarm / platform / ROS 2 / deployment automation
- **出处：** arXiv preprint，2026-09-10
- **论文链接：** <https://arxiv.org/abs/2609.11382>
- **PDF：** <https://arxiv.org/pdf/2609.11382>
- **作者：** Charbel Toumieh、Niel Mistry、Benjamin Jarvis、Simon Jeger、Peize Liu、Shaojie Shen、Dario Floreano
- **机构：** EPFL Laboratory of Intelligent Systems（LIS）；香港科技大学（HKUST）电子与计算机工程系
- **项目页：** <https://lis-epfl.github.io/swarm-nxt/>
- **代码：** <https://github.com/lis-epfl/swarm-nxt>
- **硬件基座：** [OmniNxt](https://github.com/HKUST-Aerial-Robotics/OmniNxt)（IROS 2024 Oral，HKUST Aerial Robotics）
- **入库日期：** 2026-09-16
- **一句话说明：** EPFL LIS + HKUST 提出 **SwarmNxt**——基于开源 **OmniNxt** 机体与 **ROS 2** 的端到端空中蜂群研究平台：Ansible 并行部署/更新、HDSM 去中心化规划 + 自适应 MPC + S2M2 机载深度估计，在 8×8×4 m 动捕室内验证 6 机高速互避碰与 4 机障碍环境集体飞行。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 论文 | [arXiv:2609.11382](https://arxiv.org/abs/2609.11382) | 平台论文 |
| 代码 | [lis-epfl/swarm-nxt](https://github.com/lis-epfl/swarm-nxt) | Ansible + ROS 2 栈 + 文档站 |
| 项目文档 | [lis-epfl.github.io/swarm-nxt](https://lis-epfl.github.io/swarm-nxt/) | 硬件组装、IT 基础设施、飞行流程 |
| 演示视频 | [YouTube](https://youtu.be/9aOr5EDLQEo) | 平台总览 |
| 硬件 | [HKUST-Aerial-Robotics/OmniNxt](https://github.com/HKUST-Aerial-Robotics/OmniNxt) | 360° 鱼眼 + Jetson Orin NX 开源机体 |
| 飞控 | [PX4](https://px4.io/) | 底层姿态/电机控制 |
| 相邻平台 | [Crazyswarm2](../../wiki/entities/crazyswarm2.md)、[EGO-Planner Swarm](../../wiki/entities/ego-planner-swarm.md) | 室内 swarm / 规划栈对照 |

## 摘要级要点

- **问题：** 商用无人机多闭源或算力不足；研究平台虽强但 **ROS 1 单主节点** 限制多机扩展，且 **fleet 级部署、同步、维护** 工程成本高，阻碍敏捷蜂群实验迭代。
- **定位：** 不是新规划算法论文，而是 **开源软硬件一体化蜂群基础设施**——把 OmniNxt 机体、Ansible 编排、ROS 2 多域桥接与 SOTA 控制/规划/深度模块集成为可复现真机管线。
- **硬件：** 基于 OmniNxt（全向视觉 + GPU）；BOM 约 **2300 CHF/机**（论文撰写时）；提供原子化组装说明与视频教程。
- **软件编排（Ansible）：** `drone_setup` / `drones_update` / `drones_preflight` / `drones_postflight` 等 playbook 实现并行配置、预飞检查、日志回收；Host PC 仪表盘（`:8080`）监控延迟与电量。
- **自主栈（机载）：**
  - **HDSM**（High-speed Decentralized and Synchronous Motion planner）：体素安全走廊 + 去中心化时间感知规划 + 延迟鲁棒多机协调；映射采用 log-likelihood 占据更新。
  - **自适应 MPC**（acados）：100 Hz 跟踪 HDSM 参考轨迹，输出 collective thrust / body rate 至 PX4；按垂直跟踪误差缩放推力补偿电池压降。
  - **S2M2** 立体匹配深度：机载 GPU 推理（约 7 Hz 吞吐）；四机障碍实验启用。
- **系统架构：** 各机 **规划/建图/控制机载**；邻机通过 **domain bridge** 交换计划轨迹实现去中心化互避碰；**全局位姿由动捕经 Host PC 注入**（论文明确：规划算法去中心化，系统整体仍依赖中心动捕）。
- **实验（8×8×4 m 室内动捕）：**
  1. **6 机空场：** 圆周换位 + 随机漫游；约 2 h 累计飞行、0.2% 丢包与通信延迟下 **零碰撞**；最大速度约 10 m/s 约束内。
  2. **4 机障碍场：** 机载深度 + 保守建图/规划调参；30 min **零机间/障碍碰撞**（其余机体当时未完成鱼眼标定）。
- **算力：** 启用深度时 GPU ~95%、CPU ~54%；映射/规划 worst-case 满足 100 ms 预算；MPC worst-case 40.4 ms（超 10 ms 预算，PX4 保持上一 setpoint，最大跟踪误差 0.301 m < 0.45 m 规划安全半径）。

## 核心摘录（面向 wiki 编译）

### 1) 平台对比动机（Table I）

| 维度 | 商用（如 Mavic 3E） | 微四轴（Crazyswarm 类） | 研究机（FLA/Agilicious） | Starling 2 | SwarmNxt |
|------|---------------------|-------------------------|--------------------------|------------|----------|
| 开源/可改 | 否 | 是 | 是 | 部分 | **是** |
| GPU + 深度+规划并行 | 闭源黑盒 | 算力不足 | ROS 1 扩展难 | 缺 swarm 框架 | **ROS 2 + Orin** |
| Fleet 部署脚本 | N/A | 有 | 弱 | 文档原子化 | **Ansible 全套** |

**对 wiki 的映射：** [`wiki/entities/paper-swarmnxt.md`](../../wiki/entities/paper-swarmnxt.md)

### 2) Ansible 蜂群编排

- 单机 setup → 并行 `drones_update` pull/build → `drones_preflight` 启动 micro XRCE + 自主栈 + chrony/相机检查 → 飞行 → `drones_postflight` 集中 rosbag。
- Post-flight 自动汇总 MPC 跟踪误差、栈计算时延、最小机间距。

**对 wiki 的映射：** 工程实践节 + 源码运行时序图

### 3) 开源状态（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| SwarmNxt 软件 + 文档 | **已开源** | [GitHub](https://github.com/lis-epfl/swarm-nxt)，GitHub Pages 文档站 |
| OmniNxt 硬件设计 | **已开源** | [HKUST-Aerial-Robotics/OmniNxt](https://github.com/HKUST-Aerial-Robotics/OmniNxt) |
| HDSM / MPC / S2M2 模块 | **集成开源依赖** | 通过 ansible 安装 acados、perception 等；子模块见 `.gitmodules` |
| 动捕全局定位 | **外部依赖** | 非平台自带；论文承认 CVIO 尚不足以支撑敏捷蜂群 |
| 许可证 | 仓库根目录未标注 SPDX | 使用前请查阅各子包 LICENSE |

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-swarmnxt.md`](../../wiki/entities/paper-swarmnxt.md)**
- 交叉：**[`wiki/overview/multirotor-simulation-planning-control-stack.md`](../../wiki/overview/multirotor-simulation-planning-control-stack.md)**
- 谱系：**[`wiki/entities/crazyswarm2.md`](../../wiki/entities/crazyswarm2.md)**（室内动捕蜂群）、**[`wiki/entities/ego-planner-swarm.md`](../../wiki/entities/ego-planner-swarm.md)**（规划器）、**[`wiki/entities/px4-autopilot.md`](../../wiki/entities/px4-autopilot.md)**（飞控）
