---
type: entity
tags:
  - paper
  - uav
  - multirotor
  - trajectory-planning
  - active-perception
  - esdf
  - prior-map-free
  - minco
  - zju
status: complete
updated: 2026-06-28
arxiv: "2606.17630"
venue: arXiv 2026
related:
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ./ego-planner-swarm.md
  - ./paper-mighty-hermite-spline-trajectory-planning.md
  - ./px4-autopilot.md
  - ./mavsdk.md
sources:
  - ../../sources/papers/flap_arxiv_2606_17630.md
summary: "FLAP（arXiv:2606.17630）：ZJU 无先验地图 3D UAV 规划——传感器系 FOV 主动感知惩罚 + 可优化 AP 子段 + MINCO 联合时空优化；窄垂直 FOV 与竖向机动仿真/真机验证。"
---

# FLAP（FOV 约束主动感知 · 无先验地图 3D 导航）

**FLAP**（*FOV-Constrained Active Perception Planning for Prior-Map-Free 3D Navigation*，arXiv:2606.17630，[论文](https://arxiv.org/abs/2606.17630)）是浙江大学提出的 **无先验地图四旋翼轨迹优化** 框架：把 **主动感知** 写成轨迹优化中的 **可微惩罚项**，在 **传感器坐标系** 从 UAV 动力学推导 **FOV 可见性约束**，并用 **速度触发** 机制与 **可优化主动感知（AP）子段** 在安全性与航时效率之间自适应折中。

## 一句话定义

**在双 ESDF 划分的已知/未知空间中，用 MINCO 联合优化飞行轨迹，同时优化「何时、如何」把未知边界观测点 $\boldsymbol{p}_v$ 纳入有限 FOV**——避免依赖昂贵感知感知前端，也避免全程保守限速。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FLAP | FOV-Constrained Active Perception Planning | 本文框架：视场约束下的主动感知轨迹规划 |
| FOV | Field of View | 传感器视场角，含水平/垂直区间与量程 |
| ESDF | Euclidean Signed Distance Field | 欧氏符号距离场；本文用双 ESDF 区分已知安全与条件可通行未知区 |
| UAV | Unmanned Aerial Vehicle | 无人机；本文聚焦微分平坦四旋翼 |
| MINCO | Minimum Control | 最小控制努力轨迹类；本文用 $h=3$ 表示位置+yaw |
| AP | Active Perception | 主动感知子轨迹段，时长固定、起始时刻可优化 |

## 为什么重要

- **补齐多旋翼栈「感知—规划」耦合缺口**：相对 [EGO-Planner Swarm](./ego-planner-swarm.md) / [MIGHTY](./paper-mighty-hermite-spline-trajectory-planning.md) 等以 **几何碰撞 + 动力学** 为主的软约束规划器，FLAP 显式建模 **传感器 FOV 几何** 与 **未知空间进入时机**，面向 **无先验地图** 巡检、搜救、工业竖向结构。
- **3D 竖向机动是硬场景**：加速度致 pitch 偏转、有限 **垂直 FOV** 使「yaw 对准速度方向」类启发式在爬升/下降时失效；FLAP 在传感器系统一处理 Livox Mid-360 非对称垂直 FOV 与深度相机对称 FOV。
- **工程取向**：仅需 **简单加权 A* 前端** 而非专用 perception-aware path generator；约束与惩罚均为 **可微**，便于梯度法联合优化。

## 核心结构

| 模块 | 作用 |
|------|------|
| **双 ESDF 地图** | $E_s$：未知+障碍皆不安全；$E_u$：仅障碍不安全 → $\mathcal{D}_s$（已知安全）、$\mathcal{D}_a$（条件可通行未知） |
| **前端 A*** | 在 $\mathcal{D}_s$ 内优先接近目标高度，减少反复竖向主动感知 |
| **可见性约束** | 边界平面 $\mathcal{B}$ 上定义 $\boldsymbol{p}_u$ 与偏移观测点 $\boldsymbol{p}_v$；传感器系 FOV/量程/最小观测角 |
| **风险感知惩罚** | $\mathcal{L}_a(\mathcal{C}_{sd})$：制动距离满足时弱化感知项，风险升高时激活 |
| **AP 子段** | $\boldsymbol{S}_p\subset\boldsymbol{S}_s$，优化起始 $\rho$ + 末端 safeguard，鼓励 **提前观测** |
| **MINCO 后端** | 平坦输出 $[\boldsymbol{p},\psi]$ 联合时空优化；边界点与终端速度/yaw 松弛 |
| **重规划** | ESDF 边界更新、AP 观测失败、规划延迟 $T_R$ 下的轨迹切换 |

### 流程总览

```mermaid
flowchart LR
  S["深度/LiDAR\n占据栅格"] --> D["双 ESDF\nDs / Da"]
  D --> A["加权 A*\n前端路径"]
  A --> B["边界 pu / 观测点 pv"]
  B --> M["MINCO 轨迹\n+ AP 子段优化"]
  M --> P["传感器系 FOV\n主动感知惩罚"]
  P --> O["L-BFGS 类\n梯度优化"]
  O --> T["轨迹跟踪\nPX4 / MAVSDK"]
```

### 与相邻规划器的分界

| 规划器 | 先验地图 | 主动感知 / FOV | 轨迹表示 | 备注 |
|--------|----------|----------------|----------|------|
| [EGO-Planner Swarm](./ego-planner-swarm.md) | 在线建图 ESDF | 不显式 FOV 优化 | B-spline | ROS 工程生态、swarm |
| [MIGHTY](./paper-mighty-hermite-spline-trajectory-planning.md) | 在线 ESDF | 不显式 FOV 优化 | Hermite 联合时空 | 单机效率导向 |
| SUPER / FASTER 类 | 在线 | 双轨迹备份 | 多种 | 竖向 FOV 盲区内备份易失效 |
| **FLAP** | **无先验、在线** | **嵌入优化** | **MINCO** | 窄垂直 FOV 与 3D 机动 |

## 实验要点（索引级）

- **水平窄道（LiDAR）**：垂直 FOV 从 90° 收紧至 0.2°；FLAP 在 SUPER 全失败、FM 极窄 FOV 不可行时仍 **6/10** 成功，且 AP 段占比随 FOV 收紧上升（约 28%→41%）。
- **头顶障碍 / U 型迷宫**：需螺旋爬升/下降；相对 SUPER 备份轨迹在竖向未知区更易误判。
- **深度相机**：~78° 水平 FOV、3 m/s 限速；窄道与竖向重叠障碍 passage 相对 RAPTOR/FM/NBV 报告更短航时。
- **真机**：Mid-360 水平/倾斜安装、深度相机；高墙翻越、L 形障碍竖向策略对比、室外非结构化导航。

## 常见误区或局限

- **误区：FLAP 替代 SLAM/建图** — 仍依赖 onboard 感知更新占据/ESDF；规划层解决 **如何利用有限 FOV 安全穿越未知区**。
- **误区：有 FOV 约束即绝对安全** — 可见性约束 **不建模遮挡**；依赖 AP 末端 safeguard 与重规划处理观测失败。
- **局限：前端质量敏感** — 论文指出 known–unknown 边界由前端决定，边界估计误差在极窄 FOV 下可致优化失败或急停。
- **局限：开源代码** — 截至 ingest 时 arXiv 页 **未列出公开仓库**；复现需对照论文实现 MINCO + 双 ESDF + 传感器系惩罚链。

## 关联页面

- [多旋翼仿真—规划—飞控栈总览](../overview/multirotor-simulation-planning-control-stack.md) — 本页在「无地图 + FOV 感知规划」层的定位
- [EGO-Planner Swarm](./ego-planner-swarm.md) — 同赛道 ESDF + 梯度优化工程基线
- [MIGHTY](./paper-mighty-hermite-spline-trajectory-planning.md) — 另一联合时空软约束 UAV 规划对照
- [PX4 Autopilot](./px4-autopilot.md) · [MAVSDK](./mavsdk.md) — 常见执行层

## 参考来源

- [FLAP 论文摘录（arXiv:2606.17630）](../../sources/papers/flap_arxiv_2606_17630.md)
- Zhang et al., *FLAP: FOV-Constrained Active Perception Planning for Prior-Map-Free 3D Navigation*, arXiv:2606.17630, 2026. <https://arxiv.org/abs/2606.17630>

## 推荐继续阅读

- [EGO-Planner 原版论文与仓库](https://github.com/ZJU-FAST-Lab/ego-planner) — ZJU FAST Lab 同机构经典局部规划基线
- [MIGHTY（arXiv:2511.10822）](https://arxiv.org/abs/2511.10822) — 联合时空软约束 UAV 规划对照
