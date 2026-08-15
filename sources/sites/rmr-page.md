# RMR（Optimization-Based Rig Unification 项目页）

> 来源归档（ingest）

- **标题：** Robust and Expressive Humanoid Motion Retargeting via Optimization-Based Rig Unification
- **类型：** site / project-page
- **URL：** <https://tmjeong1103.github.io/RMR/>
- **入库日期：** 2026-08-15
- **配套论文：** IROS 2025，pp. 21619–21626，DOI [10.1109/IROS60139.2025.11246607](https://doi.org/10.1109/IROS60139.2025.11246607)
- **可运行实现：** 无独立 RMR 仓库；算法已并入 [CoRe](https://github.com/tmjeong1103/CoRe) 的 **DMR / common-rigging** 阶段 — 归档见 [`sources/repos/core_retarget.md`](../repos/core_retarget.md)

## 一句话摘要

高丽大学 / CINAMON / Rainbow Robotics / NAVER LABS / UIUC 的 **RMR** 官方站点：把异构人体骨架先统一到 **canonical rig**（pre-rigging + post-rigging），再用 **方向向量 + IK** 映射到多人形，强调噪声视频估计也可出稳定上身表达。

## 公开信息要点（截至入库日）

- **录用：** 2025 IEEE/RSJ IROS，Hangzhou
- **作者：** Taemoon Jeong、Taehyun Byun、Jihoon Kim、Keunjun Choi、Jaesung Oh、Sungpyo Lee、Omar Darwish、Joohyung Kim、Sungjoon Choi
- **评测：** 12 台仿真机器人；真机 **AMBIDEX、THORMANG、JF2**（足固定约束下的上身表达）
- **代码：** 项目页未列独立 GitHub；CoRe README 写明软件「builds on」本页方法
- **无 arXiv：** 截至 2026-08-15 未检索到公开预印本

## 管线（项目页 Overview）

1. **采集** — MoCap 或单目 3D 姿态估计（可含噪声）
2. **Common-rigging**
   - Pre-rigging：各人体骨架 IK 到带质量/惯量的统一 rig
   - Post-rigging：足–地接触、质心对齐、自碰消除
3. **方向向量重定向** — 按机型 JOI 与连杆长度缩放方向向量，再 IK（限位 / 速度 / 自碰）并轨迹优化

## 为何值得保留

- 解释 CoRe 软件中 **DMR** 阶段的论文来源，避免把 v0.1.0 误读成「只有接触精炼、没有跨骨架统一」。
- 真机 AMBIDEX 舞蹈与实时视频→机器人执行演示，补 IEEE 付费墙。

## 对 wiki 的映射

- [`wiki/entities/paper-rmr.md`](../../wiki/entities/paper-rmr.md)
- [`wiki/entities/core-retarget.md`](../../wiki/entities/core-retarget.md)
