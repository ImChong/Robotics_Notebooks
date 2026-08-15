# RMR（IROS 2025）

> 来源归档（ingest）

- **标题：** Robust and Expressive Humanoid Motion Retargeting via Optimization-Based Rig Unification
- **类型：** paper
- **来源：** 项目页摘要 / IEEE Xplore 元数据 / CoRe 软件文档
- **原始链接：**
  - 项目页 <https://tmjeong1103.github.io/RMR/>
  - IEEE <https://doi.org/10.1109/IROS60139.2025.11246607>
  - 实现并入 <https://github.com/tmjeong1103/CoRe>
- **作者：** Taemoon Jeong、Taehyun Byun、Jihoon Kim、Keunjun Choi、Jaesung Oh、Sungpyo Lee、Omar Darwish、Joohyung Kim、Sungjoon Choi
- **机构：** 高丽大学（Korea University）；CINAMON；彩虹机器人（Rainbow Robotics）；纳沃实验室（NAVER LABS）；伊利诺伊大学厄巴纳-香槟分校（UIUC）
- **会议：** 2025 IEEE/RSJ IROS，Hangzhou，pp. 21619–21626
- **入库日期：** 2026-08-15
- **一句话说明：** 用带物理属性的 **canonical rig** 统一异构人体骨架（含噪声视频估计），再以方向向量缩放 + IK 映射到多人形，在限位与自碰约束下保留上身表达并稳住足接触。

## 核心摘录

### 1) 问题

人形要「像人」且物理可执行。源数据骨架不一、视频估计有噪声；直接按机型写重定向易碎。

### 2) Common-rigging + 方向向量重定向

- **Pre-rigging：** 各人体骨架 IK 到统一 rig（含质量 / 惯量，便于自碰与噪声姿态修正）
- **Post-rigging：** 足–地接触、质心对齐、消除不可执行伪影
- **JOI + 方向向量：** 按目标连杆长度缩放方向向量得目标姿态，再 IK（关节限位、速度界、自碰）并轨迹优化跟踪源运动

### 3) 评测（项目页）

- **仿真：** 12 台不同运动学人形
- **真机：** AMBIDEX、THORMANG、JF2；MoCap 与 RGB 视频估计均可 **无额外调参** 部署
- **范围：** 强调 **足固定约束下的上身表达** 与接触一致，不是全身动力学 locomotion 跟踪论文
- **实时演示：** 单目视频 → 姿态估计 → common-rigging → 机型重定向 → 真机执行

### 4) 开源边界

- 项目页 **无独立 GitHub**
- CoRe v0.1.0 的 **DMR** 阶段是官方可运行实现（SOMA77 → 11 机）；RMR 原文评测机型（AMBIDEX 等）不在 v0.1.0 捆绑表内
- **无 arXiv：** 截至入库日以项目页 + IEEE 为准

```bibtex
@inproceedings{jeong2025robust,
  author    = {Jeong, Taemoon and Byun, Taehyun and Kim, Jihoon and
               Choi, Keunjun and Oh, Jaesung and Lee, Sungpyo and
               Darwish, Omar and Kim, Joohyung and Choi, Sungjoon},
  title     = {Robust and Expressive Humanoid Motion Retargeting via
               Optimization-Based Rig Unification},
  booktitle = {2025 IEEE/RSJ International Conference on
               Intelligent Robots and Systems (IROS)},
  year      = {2025},
  pages     = {21619--21626},
  doi       = {10.1109/IROS60139.2025.11246607}
}
```

## 对 wiki 的映射

- 升格 [RMR 论文实体](../../wiki/entities/paper-rmr.md)
- 软件实现 [CoRe](../../wiki/entities/core-retarget.md)
- 后续接触精炼 [CoRe 论文](../../wiki/entities/paper-core.md)
- 对照 [GMR](../../wiki/methods/motion-retargeting-gmr.md)、[SOMA Retargeter](../../wiki/entities/soma-retargeter.md)

## 当前提炼状态

- [x] 项目页摘要 + common-rigging + 真机三平台
- [x] 与 CoRe 软件 DMR 阶段对齐
- [ ] IEEE 全文表格数字（付费墙；有预印本后再补）
