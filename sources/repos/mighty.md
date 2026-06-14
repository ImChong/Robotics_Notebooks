# mighty（MIT ACL · Hermite 样条 UAV 轨迹规划）

> 来源归档

- **标题：** mighty
- **类型：** repo
- **来源：** MIT Aerospace Controls Laboratory (mit-acl)
- **链接：** <https://github.com/mit-acl/mighty>
- **License：** BSD-3-Clause
- **Stars：** ~200（2026-06，以 GitHub 为准）
- **入库日期：** 2026-06-14
- **一句话说明：** MIGHTY 论文官方实现——ROS 2 Humble 下的 **五次 Hermite 样条** 软约束四旋翼轨迹规划器，含 Docker/Gazebo 仿真、RViz2 交互选点与 Livox LiDAR 真机管线。
- **沉淀到 wiki：** [`wiki/entities/paper-mighty-hermite-spline-trajectory-planning.md`](../../wiki/entities/paper-mighty-hermite-spline-trajectory-planning.md)、[`wiki/overview/multirotor-simulation-planning-control-stack.md`](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**MIGHTY**（*M*IT *I*ntegrated *G*radient-based *H*ermite-spline *T*rajectory planner for quadrotors, *Y*…）是 RA-L 2026 论文 [arXiv:2511.10822](https://arxiv.org/abs/2511.10822) 的 **完整开源栈**：规划后端 + ROS 2 仿真/真机集成。

```
mighty/
├── docker/              # Docker 一键仿真（Linux / Mac + Xpra）
├── scripts/run_sim.py   # 统一仿真入口（interactive / gazebo）
├── setup.sh             # 原生 Ubuntu 22.04 依赖与 colcon 构建
└── mighty.repos         # 锁定 DecompROS2、Livox 等依赖版本
```

---

## 功能摘要（README）

| 维度 | 内容 |
|------|------|
| 平台 | Ubuntu 22.04 + **ROS 2 Humble** |
| 安装 | Docker（Linux Engine / Mac Desktop）、原生 `setup.sh` |
| 仿真 | RViz2 **2D Goal Pose** 交互；Gazebo 场景（`easy_forest` / `hard_forest` 等） |
| 感知硬件 | **Livox LiDAR**（`livox_ros_driver2`）、LiDAR 定位与 ESDF 建图管线 |
| 规划表示 | 五次 Hermite spline，联合优化 knot 位姿/速度/加速度与段时长 |
| 论文状态 | **IEEE RA-L 2026** 录用（doi: 10.1109/LRA.2026.3681187） |

---

## 快速上手

### Docker（Linux）

```bash
git clone --depth 1 https://github.com/mit-acl/mighty.git
cd mighty/docker
make build
make run-interactive          # RViz2 点选目标
make run-gazebo ENV=hard_forest
```

### 原生（Ubuntu 22.04）

```bash
mkdir -p ~/code && cd ~/code
git clone https://github.com/mit-acl/mighty.git mighty_ws/src/mighty
cd mighty_ws/src/mighty && ./setup.sh
source ~/.bashrc
cd ~/code/mighty_ws
python3 src/mighty/scripts/run_sim.py --mode interactive --setup-bash ~/code/mighty_ws/install/setup.bash
```

---

## 与 EGO-Planner 等基线的关系

- 论文 benchmark 直接对比 **EGO-Planner、RAPTOR、MINCO/SUPER、多项式** 等软约束 UAV 规划器；MIGHTY 在 **联合时空优化 + Hermite 局部导数控制** 上取得更短飞行时间与更快求解。
- 工程栈同为 **ROS 2 + ESDF 碰撞软惩罚 + Offboard 设定点** 范式，可作为 [EGO-Planner Swarm](../../wiki/entities/ego-planner-swarm.md) 路线的 **表示与优化层升级参考**。

---

## 参考来源

- [mighty_arxiv_2511_10822.md](../papers/mighty_arxiv_2511_10822.md)
- <https://github.com/mit-acl/mighty>
- <https://arxiv.org/abs/2511.10822>
