# Ultra-Fusion：传感器退化与时空扰动下的韧性紧耦合多传感器融合 SLAM（arXiv:2606.21223）

> 论文来源归档（ingest）

- **标题：** Ultra-Fusion: A Resilient Tightly-Coupled Multi-Sensor Fusion SLAM Framework under Sensor Degradation and Spatiotemporal Perturbation for Intelligent Transportation Systems
- **类型：** paper / slam / multi-sensor-fusion / localization / its / lidar / vio / gnss
- **arXiv：** <https://arxiv.org/abs/2606.21223> · PDF：<https://arxiv.org/pdf/2606.21223>
- **项目页：** <https://sjtuyinjie.github.io/ultrafusion-web/>
- **代码（论文承诺）：** <https://github.com/sjtuyinjie/Ultra-Fusion>（接受后发布）
- **扩展基准：** <https://github.com/sjtuyinjie/M3DGR>
- **机构：** 北京理工大学、重庆大学、四川大学、西北工业大学、上海交通大学（通讯：Jie Yin）
- **入库日期：** 2026-06-24
- **一句话说明：** 提出 **统一滑窗估计器**，在 **同一优化窗口** 内按时间戳排序并接纳 **WIO / VIO / LIO / LVIO** 及可选 **轮速 / GNSS** 因子；配合 **可观测性感知初始化**、**因子级可靠性调度（FRS）** 与 **在线 LiDAR–IMU 时空标定（OSC）**，在 M3DGR、M2DGR-Plus、KAIST、GrandTour、MARS-LVIG 上对 **60+** 开源 SLAM 系统做大规模鲁棒性评测，覆盖轮式、腿式与 UAV 平台。

## 核心摘录（面向 wiki 编译）

### 1) 统一滑窗估计器与可配置传感器栈

- **要点：** 异构异步测量按时间戳排序，在同一滑窗内转为 **可选因子**；共享状态表示、因子准入、边缘化与标定逻辑，支持 **WIO、VIO、LIO、LVIO** 及 **轮速 / GNSS** 增广，无需为不同传感器组合重写整条管线（相对 Ground-Fusion++ 等子系统耦合方案）。LiDAR 以 **点级残差** 直接进入图优化，而非仅注入外部 LiDAR 里程计先验。
- **对 wiki 的映射：** [`wiki/entities/paper-ultra-fusion-multi-sensor-slam.md`](../../wiki/entities/paper-ultra-fusion-multi-sensor-slam.md)

### 2) 因子级可靠性调度（Factor-Wise Reliability Scheduling）

- **要点：** 在优化问题内对 **LiDAR / 视觉 / IMU / 轮速 / GNSS** 因子做 **退化感知门控与降权**，应对弱光照、LiDAR 几何退化、轮速打滑、GNSS 拒止；消融报告 LiDAR FRS 平均 ATE **−75.3%**、视觉 FRS **−36.2%**、轮速 FRS **−41.3%**。隧道/长廊等强退化场景相对 FAST-LIVO2、R3LIVE 等基线优势显著。
- **对 wiki 的映射：** 同上实体页；[`wiki/concepts/sensor-fusion.md`](../../wiki/concepts/sensor-fusion.md)

### 3) 可观测性感知初始化与在线时空标定

- **要点：** 启动阶段按 **可观测性** 选择引导模式；全模型平均初始化延迟 **0.153 s**（无自适应 **4.642 s**），前 20 s 平均 ATE **0.483 m**（无自适应 **16.808 m**）。在线标定在 **激励充分且传感器可靠** 时估计 **时间偏移与旋转外参**；注入 **±300 ms** IMU 时间偏移时全模型 RMSE 仍约 **厘米级**，外参旋转扰动 **10°** 时相对 FAST-LIVO2（**940 m**）保持稳定。
- **对 wiki 的映射：** 同上实体页；[`wiki/comparisons/lidar-slam-lio-vio-selection.md`](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)

### 4) 大规模基准与跨平台验证

- **要点：** 扩展 **M3DGR**（含仿真退化轨迹），系统对比 **60+** 代表 SLAM 系统；轮式 **M2DGR-Plus** 上 LVWIO 配置平均漂移 **0.59% / 0.24 m**（FAST-LIVO2 **2.32% / 1.48 m**）；**KAIST** 高速城市场景、**GrandTour** 四足、**MARS-LVIG** UAV 均报告领先或前列；单步优化 **5.48–10.73 ms**（i9-14900K，实时配置）。
- **对 wiki 的映射：** 同上实体页；[`wiki/overview/topic-state-estimation.md`](../../wiki/overview/topic-state-estimation.md)

### 5) 与既有融合 SLAM 的对照定位

- **要点：** 相对 **FAST-LIVO2**（固定 CIL、无在线时空标定）、**Ground-Fusion / Ground-Fusion++**（GNSS 可选但模式与子系统绑定），Ultra-Fusion 强调 **可配置 WIO/VIO/LIO/LVIO + 可选 W/G**、**图内退化调度** 与 **在线时空标定**，并支持 **高斯泼溅（GS）** 彩色建图输出。
- **对 wiki 的映射：** [`wiki/comparisons/lidar-slam-lio-vio-selection.md`](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 实体页与对比/概念页交叉引用
- [ ] 代码仓库公开后补充 `sources/repos/ultra_fusion.md`
