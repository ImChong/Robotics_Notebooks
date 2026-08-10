---
type: entity
tags:
  - paper
  - slam
  - vslam
  - monocular
  - dense-reconstruction
  - geometric-transformer
  - pointmap
  - foundation-model
  - tsinghua
  - state-estimation
status: complete
updated: 2026-08-09
arxiv: "2608.03429"
venue: "arXiv 2026"
related:
  - ../methods/lingbot-map.md
  - ./paper-glob3r.md
  - ../concepts/state-estimation.md
  - ../overview/hub-state-estimation.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
  - ./orb-slam3.md
sources:
  - ../../sources/papers/slamformer_infinity_arxiv_2608_03429.md
  - ../../sources/sites/tsinghua-mars-lab-slamformer-infinity.md
  - ../../sources/repos/slamformer_infinity.md
summary: "SLAMFormer-∞（清华 IIIS/MARS，arXiv:2608.03429）：memory-conditioned 几何 Transformer 支撑无界前端与后端；PGGO 联合精炼长程位姿与 dense pointmap；KITTI/Waymo 优于 VGGT-Long，演示 >17 km；官方仓占位，推理代码待发布。"
---

# SLAMFormer-∞（Infinite SLAM Transformer）

**SLAMFormer-∞**（*Infinite SLAM Transformer for Unbounded Frontend and Backend Processing*，[arXiv:2608.03429](https://arxiv.org/abs/2608.03429)，[项目页](https://tsinghua-mars-lab.github.io/SLAMFormer-Infinity)）由 **清华大学 IIIS / MARS Lab**（Zhijian Fang、Weicheng Zheng、Yijun Yuan、Hang Zhao 等）提出：在统一几何 Transformer 上用 **memory condition** 定义局部坐标系与尺度，使 **流式前端** 保持有界计算，同时让 **后端 PGGO** 在长程图上联合迭代精炼相机轨迹与 dense pointmap——针对「有界联合推理」与「只对齐位姿、拼接局部几何」两条既有路线的结构取舍。

## 一句话定义

**用条件块锚定局部坐标系，让同一 transformer 既能流式跟踪，又能在回环/序列末对位姿与稠密几何做无界联合精炼。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| PGGO | Pose-Geometry Graph Optimization | 本文后端：位姿与几何联合图优化 |
| ATE | Absolute Trajectory Error | 绝对轨迹误差（文中 RMSE，米） |
| VGGT | Visual Geometry Grounded Transformer | 多视图前馈几何 foundation；长序列常需拼接 |
| KV | Key-Value（cache） | Transformer 注意力缓存，前端复用局部历史 |
| SE(3) | Special Euclidean Group in 3D | 刚体位姿群；文中阻尼插值更新 |

## 为什么重要

- **长程稠密单目仍是结构瓶颈：** 纯数据驱动 transformer SLAM（如前作 [SLAM-Former](https://arxiv.org/abs/2509.16909)）易被训练轨迹尺度卡住；VGGT-Long 类方法可拉长位姿，但 **局部 pointmap 常碎片化**。
- **前端算力与后端一致性可同模型：** Infinity 用条件局部系做有界推理，用 PGGO 做无界联合更新，避免「两套系统、两套几何」。
- **城市尺度证据：** 项目页与论文展示 **自采约 17 km / 45 min** 城市驾驶一致地图；KITTI / Waymo 上相对 VGGT-Long 的 ATE 与稠密几何均有提升。
- **选型对照清晰：** 在线流式几何看 [LingBot-Map](../methods/lingbot-map.md)；离线高精度 SfM 看 [Glob3R](./paper-glob3r.md)；**无界学习型 dense mono SLAM + 联合后端** 看本页（注意代码占位）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 作者 | Zhijian Fang*, Weicheng Zheng*, Yijun Yuan*†, Weibang Wang, Zhuoguang Chen, Chang Sun, Junhao Huang, Kenan Li, Minghui Qin, Hang Zhao† |
| 机构 | 清华大学交叉信息研究院（IIIS）· MARS Lab |
| 出处 | arXiv:2608.03429（2026-08-04） |
| 项目 | <https://tsinghua-mars-lab.github.io/SLAMFormer-Infinity> |
| 输入 | 流式 **单目 RGB**（calibration-free 评测协议） |
| 输出 | 关键帧位姿 \(SE(3)\) + dense pointmap |
| 训练 | 自 SLAM-Former 初始化；室内 12 帧·518 / 户外 36 帧·224；48×A100 · 10 epochs |
| 开源（截至 2026-08-08） | **部分开源（占位仓）**：[`SLAMFormer-Infinity`](https://github.com/Tsinghua-MARS-Lab/SLAMFormer-Infinity) 仅 README + gh-pages 站点；**无可运行训练/推理**。前作 [SLAM-Former](https://github.com/Tsinghua-MARS-Lab/SLAM-Former) 已开源，勿混 |

## 方法与核心结构

| 模块 | 作用 |
|------|------|
| **Memory condition** | \((\mathcal{I}_C,\mathcal{X}_C)\) 定义参考系与尺度；active 预测落在条件局部系 |
| **Conditional Frontend** | 关键帧检测 + 有界局部上下文 + KV；在线增量位姿与 pointmap |
| **Local Backend** | 每固定关键帧数精炼最近窗口，回写后续 frontend 的 memory |
| **PGGO（Global Backend）** | 回环或序列末：节点=位姿∪几何；迭代 transformer 精炼 + 阻尼位姿更新 |
| **四模式训练** | 共享权重，仅注意力掩码/条件模式不同，对齐测试时 frontend / backend / fine |

### 流程总览

```mermaid
flowchart TB
  rgb[ 流式单目 RGB ]
  fe[ Conditional Frontend\n关键帧 + 局部 KV ]
  mem[ Memory / 条件块\n坐标系与尺度 ]
  lb[ Local Backend\n周期窗口精炼 ]
  graph[ Pose-Geometry Graph\n前端 + 回环边 ]
  pggo[ Global PGGO\n联合位姿 + pointmap ]
  out[ 全局一致轨迹与稠密地图 ]
  rgb --> fe --> mem
  mem --> fe
  fe --> lb --> mem
  fe --> graph
  lb --> graph
  graph --> pggo --> out
```

## 源码运行时序图

**不适用**（截至 2026-08-08）：项目页与 [`Tsinghua-MARS-Lab/SLAMFormer-Infinity`](https://github.com/Tsinghua-MARS-Lab/SLAMFormer-Infinity) **未提供** 可辨识训练/推理入口（`main` 仅 README；`gh-pages` 为站点与 demo 媒体）。代码放出后应补：RGB 序列 → frontend/local backend →（可选）pose-graph 初值 → PGGO fine → 导出轨迹/点图 的 `sequenceDiagram`。工程试用可临时参考前作 [SLAM-Former `slam/demo.py`](https://github.com/Tsinghua-MARS-Lab/SLAM-Former)，但 **不能复现 Infinity 论文设定**。

## 工程实践

| 项 | 建议 / 论文设定 |
|----|----------------|
| **何时用** | 需要 **公里级单目稠密地图**，且不满足于「只对齐轨迹、拼接局部几何」时跟进本路线 |
| **何时不用** | 短室内轨迹、强特征/匹配先验场景：论文承认 [SLAM-Former](https://arxiv.org/abs/2509.16909) / MASt3R-SLAM 等仍可能更优 |
| **传感器** | 文中主线为 **calibration-free monocular RGB**；多传感器 ITS 退化场景另见 [Ultra-Fusion](./paper-ultra-fusion-multi-sensor-slam.md) |
| **算力参考** | 训练 48×A100；推理显存边界以论文 OOM 基线（单卡 RTX 4090）为对照，Infinity 自身延迟未给完整实时预算表 |
| **开源跟进** | 盯占位仓与项目页；放出前勿把 demo 视频当可部署包 |
| **源码运行时序图** | **不适用**（原因见上节） |

## 实验与评测（论文报告摘要）

| 基准 / 场景 | 对照 | 主要结论 |
|-------------|------|----------|
| **KITTI Odometry 00–10** | VGGT-Long 等 | Avg ATE **26.358 → 23.011 m**；calibration-free dense |
| **Waymo 城市片段** | VGGT-Long | Avg ATE **1.996 → 1.813 m** |
| **Waymo pointmap vs LiDAR** | VGGT-Long | Acc/Comp/Chamfer **1.182/2.860/2.021 → 0.949/2.777/1.863** |
| **7-Scenes** | VGGT-SLAM | ATE **0.068 → 0.046 m**；几何 Acc/Comp/Chamf 同步改善 |
| **TUM / Replica** | 多基线 | 与论文（2026-08）所列最强基线竞争；短室内 **SLAM-Former** 常仍最佳 |
| **Fine stage 消融** | w/o fine | Replica ATE **0.061 → 0.052**；定性表面更干净 |
| **自采长程** | VGGT-Long | **>17 km** 城市驾驶：对照崩溃，本文保持一致大地图（定性） |

## 结论

**SLAMFormer-∞ 的关键动作是把「局部坐标系条件」做成前后端共用的结构原语，再用 PGGO 把长程修正从「只拧位姿」升级为「位姿与稠密几何一起拧」。**

1. **真影响：memory condition** — 有界局部计算 ↔ 无界序列长度，不再把坐标行为绑死在首帧/训练轨迹长度上。
2. **真影响：PGGO 联合更新** — 相对 VGGT-Long 的位姿对齐，Waymo 上轨迹与 pointmap（Acc/Chamfer）同步变好；KITTI 全序列 ATE 亦降。
3. **真影响：城市场景尺度** — 论文/项目页展示 **17 km** 级一致地图，直指学习型 dense mono SLAM 的长程失效模式。
4. **次要代价：图连通外生** — 回环与边来自 frontend/检测，非端到端学出；连通差会拖累 PGGO。
5. **部署读法：** 长户外驾驶/建图优先；短室内勿默认碾压 SLAM-Former。
6. **工程读法：代码占位** — 今日只能读方法与看 demo；可跑通的是前作 SLAM-Former，不是 Infinity。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [SLAM-Former](https://arxiv.org/abs/2509.16909) | 统一 transformer SLAM 前作；全局精炼依赖增长状态/训练分布。∞ 用条件局部系 + PGGO 攻无界长程 |
| VGGT / VGGT-SLAM / VGGT-Long | 强局部几何或窗级缝合；Long 侧重位姿对齐。∞ 强调 **几何随位姿一起精炼** |
| [LingBot-Map](../methods/lingbot-map.md) | 流式前馈 + GCA/Paged KV，偏在线几何状态；∞ 保留显式 frontend/backend 节奏与图优化 |
| [Glob3R](./paper-glob3r.md) | 离线全局 SfM（tracks → 运动平均/BA）；∞ 面向 **在线 SLAM 管线** 的学习后端 |
| [ORB-SLAM3](./orb-slam3.md) | 稀疏特征经典栈；∞ 是学习稠密 pointmap 路线，工程成熟度与生态不同 |

## 局限与风险

- **图质量外生：** 作者自陈 PGGO 吃预定义连通，质量未从数据端到端学习。
- **短序列未必最优：** 室内表上 SLAM-Former / 匹配驱动方法可更强；勿用室内榜单外推长程优势反过来也成立。
- **开源未落地：** 官方仓为占位；今日无法复现论文数字或接入 Nav2。
- **户外 fine 初值：** PGGO fine stage 位姿初值依赖与 VGGT-Long 对齐的 pose-graph 配置——联合精炼仍吃经典初始化质量。

## 关联页面

- [LingBot-Map](../methods/lingbot-map.md) — 流式前馈 3D 重建对照
- [Glob3R](./paper-glob3r.md) — 离线全局 SfM + 基础模型对照
- [State Estimation](../concepts/state-estimation.md) — 视觉几何估计在控制链上游
- [状态估计知识链](../overview/hub-state-estimation.md) — SLAM / VIO 入口
- [导航·SLAM 开源栈总览](../overview/navigation-slam-autonomy-stack.md) — 经典与学习型视觉栈分层
- [LiDAR / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md) — 传感器栈选型（本方法为纯视觉稠密）
- [ORB-SLAM3](./orb-slam3.md) — 稀疏视觉经典基线

## 参考来源

- [slamformer_infinity_arxiv_2608_03429.md](../../sources/papers/slamformer_infinity_arxiv_2608_03429.md) — 论文摘录与开源核查
- [项目页归档](../../sources/sites/tsinghua-mars-lab-slamformer-infinity.md)
- [官方仓归档（占位）](../../sources/repos/slamformer_infinity.md)
- Fang et al., *SLAMFormer-∞* — <https://arxiv.org/abs/2608.03429>
- 项目页：<https://tsinghua-mars-lab.github.io/SLAMFormer-Infinity>
- 占位仓：<https://github.com/Tsinghua-MARS-Lab/SLAMFormer-Infinity>

## 推荐继续阅读

- 项目页 demo 与点云对比：<https://tsinghua-mars-lab.github.io/SLAMFormer-Infinity>
- 前作 SLAM-Former（可运行代码）：<https://github.com/Tsinghua-MARS-Lab/SLAM-Former> · <https://arxiv.org/abs/2509.16909>
- VGGT-Long（长序列对照，ICRA 2026）：项目页对比基线；论文引用 Deng et al.
- LingBot-Map（流式几何）：<https://arxiv.org/abs/2604.14141>
