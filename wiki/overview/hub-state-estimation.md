---
type: overview
tags: [hub, hub-state-estimation, slam, odometry, ekf, perception]
status: complete
updated: 2026-09-04
summary: "状态估计知识链汇总：本体感知融合、SLAM/VIO/LIO 选型与 Kalman/优化估计框架，服务 locomotion 与导航中的位姿与速度估计。"
related:
  - ../entities/paper-slamformer-infinity.md
  - ../entities/paper-glob3r.md
  - ../methods/lingbot-map.md
  - ../queries/slam-second-spring-embodied.md
---

# 状态估计（知识链汇总）

> **知识链汇总**：本页是相关概念/方法的统一入口；对应策展纵深见图谱 [路线视图](../../docs/graph.html?depth=navigation) 与 [路线页](../../roadmap/depth-navigation.md)。

## 一句话定义

**状态估计** 从 **IMU、关节编码器、相机、LiDAR、雷达等传感器** 融合出机器人位姿、速度与接触/地形状态，是感知式 locomotion 与导航的控制输入基础。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EKF | Extended Kalman Filter | 非线性系统常用滤波框架 |
| UKF | Unscented Kalman Filter | 无迹卡尔曼变体 |
| VIO | Visual-Inertial Odometry | 视觉-惯性里程计 |
| LIO | LiDAR-Inertial Odometry | 激光-惯性里程计 |
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| CFAR | Constant False Alarm Rate | 雷达恒虚警率检测，抑制杂波虚警 |

## 为什么重要

- **盲走 vs 感知走**：估计质量决定能否在复杂地形稳定行走。
- **多传感器时间对齐**：与 [通信/时钟同步](./hub-communication.md) 强相关。
- **Sim2Real 感知 gap**：仿真传感器噪声模型与真机不一致会拖垮策略。

## 本知识链覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 概念 | 状态估计总览 | [State Estimation](../concepts/state-estimation.md) |
| 融合 | 多传感器融合 | [Sensor Fusion](../concepts/sensor-fusion.md) |
| 对比 | KF vs 优化估计 | [Kalman vs Optimization Estimation](../comparisons/kalman-filter-vs-optimization-based-estimation.md) |
| 对比 | LiDAR SLAM 选型 | [LiDAR SLAM / LIO / VIO Selection](../comparisons/lidar-slam-lio-vio-selection.md) |
| 导航栈 | SLAM 与自主导航 | [Navigation SLAM Autonomy Stack](./navigation-slam-autonomy-stack.md) |

## 与其他知识链的关系

- **[Locomotion](./hub-locomotion.md)**：感知式越障依赖状态估计。
- **[Sim2Real](./hub-sim2real.md)**：感知域随机与噪声建模。
- **[视觉骨干](./hub-vision-backbone.md)**：VIO 依赖视觉特征质量。

## 关联页面

- [AERIS-10（开源相控阵雷达）](../entities/aeris-10-plfm-radar.md) — 10.5 GHz PLFM 主动测距 + GPS/IMU 点迹修正；非 SLAM 栈，可作 **雷达测距** 硬件参考
- [Ultra-Fusion（多传感器 SLAM）](../entities/paper-ultra-fusion-multi-sensor-slam.md) — 统一滑窗 LVIO/LVWIO、退化调度与在线时空标定（arXiv:2606.21223）
- [X-IONet（跨平台惯性里程计）](../entities/paper-x-ionet-cross-platform-inertial-odometry.md) — 单 IMU 行人/四足 IO + EKF（IEEE RA-L 2026）
- [FOCUS（连续足部 FK 可靠度）](../entities/paper-focus-foot-observation-confidence.md) — 人形本体 EKF；接触 ≠ FK 可信（A3 Ultra，arXiv:2609.02222）
- [Query：具身时代 SLAM 精华与糟粕](../queries/slam-second-spring-embodied.md) — 深蓝沙龙纪要：留下时空基准，丢掉给人看的中间图
- [Glob3R（全局 SfM + 3D 基础模型）](../entities/paper-glob3r.md) — 冻结 Pi3X + tracks → 运动平均/BA；离线高精度建图与渲染
- [SLAMFormer-∞（无界 dense mono SLAM Transformer）](../entities/paper-slamformer-infinity.md) — memory condition + PGGO 联合长程位姿与 pointmap；官方仓占位（arXiv:2608.03429）
- [PanoLOG / G²PS](../entities/paper-panolog-ggps.md) — ERP 全景户外大规模 3DGS 划分重建（位姿下游 novel-view 资产）
- [LingBot-Map](../methods/lingbot-map.md) — 流式前馈 3D 重建（在线几何对照）
- [Contact Estimation](../concepts/contact-estimation.md)
- [Terrain Latent Representation](../concepts/terrain-latent-representation.md)
- [3D Spatial VQA](../concepts/3d-spatial-vqa.md)

## 参考来源

- 本库归纳自 [State Estimation](../concepts/state-estimation.md) 及 SLAM/VIO 对比页
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`state-estimation` 命中规则）
- 上游原始资料（本链概念页共同的 ingest 来源）：[线性 / 扩展卡尔曼滤波一手资料索引](../../sources/papers/kalman_filter_ekf_primary_refs.md)、[机器人状态估计核心论文](../../sources/papers/state_estimation.md)、[Ultra-Fusion：韧性紧耦合多传感器融合 SLAM（arXiv:2606.21223）](../../sources/papers/ultra_fusion_arxiv_2606_21223.md)
