# Joint Navigation and Manipulation Planning with 3D Interaction Chains（3D-IC）

> 来源归档（ingest）

- **标题：** Joint Navigation and Manipulation Planning with 3D Interaction Chains
- **类型：** paper / mobile-manipulation / ovmm / navigation / manipulation / planning / vlm
- **venue：** ICML 2026 Poster（[ICML 虚拟页 #809](https://icml.cc/virtual/2026/poster/61610)）
- **原始链接：**
  - ICML：<https://icml.cc/virtual/2026/poster/61610>
  - 代码：<https://github.com/kekeZ66/3D-IC>
- **作者：** Keming Zhang, Sixian Zhang†, Xinhang Song, Hongyu Wang, Yiyao Wang, Yingjie Wang, Shuqiang Jiang
- **机构：** 中国科学院计算技术研究所 人工智能安全全国重点实验室；中国科学院大学；中国科学院计算技术研究所
- **入库日期：** 2026-07-10
- **一句话说明：** **3D Interaction Chains（3D-IC）** 面向 **开放词汇移动操作（OVMM）**，用 **共享 3D 特征图** 统一导航与操作规划，生成 **阶段对齐的交互路点** 并串联为 **多阶段交互链**；**分层策略** 联合 **VLM 可行性推理** 与 **转移代价** 选链，闭环执行并在 Stretch 3 真机验证任务成功率与轨迹效率双提升。

## 摘要级要点

- **问题：** OVMM 需要未见环境中的 **长视界导航** 与 **以物体为中心的操作**；主流 **分阶段** 管线常出现 **导航终点不利于操作**，或 **利于操作的姿态全局路径低效** 的错配。
- **核心表示：** 维护 **导航与操作共享的 3D 特征图**；为每个子阶段生成 **interaction waypoint（交互路点）**，将多阶段路点 **链接为候选交互链（interaction chain）**。
- **决策：** **分层策略（hierarchical policy）** 对候选链打分——**可行性** 由 **VLM 对路点中心 3D 特征的推理** 估计，**转移代价** 衡量阶段间移动开销；在 **成功率与路径效率** 间取最优折中。
- **执行：** 机器人执行 **下一个路点**，随新观测 **在线重规划**（receding-horizon / replanning）。
- **验证：** 仿真与 **Hello Robot Stretch 3** 真机；报告 **任务成功率** 与 **轨迹效率** 一致提升。

## 核心摘录（面向 wiki 编译）

### 1) 分阶段 OVMM 的「手递手」错配

- **链接：** <https://icml.cc/virtual/2026/poster/61610> Abstract
- **摘录要点：** 将导航与操作 **割裂为先后两阶段** 时，导航模块优化的 **区域到达** 未必等于操作模块需要的 **可达姿态与视角**；反之，局部 **操作友好位姿** 可能在全局上绕远。OVMM 的长视界特性使这种错配在长链任务中累积。
- **对 wiki 的映射：**
  - [3D-IC（论文实体）](../../wiki/entities/paper-3d-ic-joint-navigation-manipulation-planning.md) — 问题定义与分阶段局限

### 2) 共享 3D 特征图与交互路点

- **链接：** ICML Abstract；作者组 VIPL 相关工作（TrajRAG、Multi-Scale Gaussian-Language Map 等）共享 **几何–语义 3D 地图** 叙事
- **摘录要点：** 3D-IC 不为导航与操作维护两套独立场景表示，而在 **同一 3D feature map** 上为两技能生成 **stage-aligned waypoints**——每个路点同时承载 **几何可达性** 与 **操作交互语义**（物体中心、开放词汇目标）。
- **对 wiki 的映射：**
  - [3D-IC](../../wiki/entities/paper-3d-ic-joint-navigation-manipulation-planning.md) — 共享地图与路点表示

### 3) 交互链候选与分层打分

- **摘录要点：** 多阶段路点 **链接为候选链**；**高层** 用 **VLM** 在 **waypoint-centric 3D 特征** 上推理 **链级可行性**（能否完成开放词汇操作子目标），**低层/代价项** 评估 **阶段间转移**；联合选 **成功率–效率 Pareto 最优** 链。
- **对 wiki 的映射：**
  - [3D-IC](../../wiki/entities/paper-3d-ic-joint-navigation-manipulation-planning.md) — 分层策略与 VLM 可行性

### 4) 闭环执行与真机 Stretch 3

- **摘录要点：** 执行 **下一交互路点** 而非一次性开环全链；新观测到达后 **重规划**，适应部分可观测与动态遮挡。真机 **Stretch 3** 验证 **成功率与轨迹长度/效率** 同步改善。
- **对 wiki 的映射：**
  - [3D-IC](../../wiki/entities/paper-3d-ic-joint-navigation-manipulation-planning.md) — 闭环 replanning 与真机证据
  - 交叉：[REALM](../papers/realm_last_3_meter_vln_arxiv_2607_03792.md) — 同 Stretch 平台、互补关注 **VLN 末段实例接地**

## 对 wiki 的映射（总览）

- 实体页：[paper-3d-ic-joint-navigation-manipulation-planning.md](../../wiki/entities/paper-3d-ic-joint-navigation-manipulation-planning.md)
- 交叉任务页：[loco-manipulation.md](../../wiki/tasks/loco-manipulation.md)、[vision-language-navigation.md](../../wiki/tasks/vision-language-navigation.md)、[manipulation.md](../../wiki/tasks/manipulation.md)
- 代码归档：[sources/repos/3d-ic.md](../repos/3d-ic.md)

## BibTeX（待 arXiv 公开后补全 eprint）

```bibtex
@inproceedings{zhang2026_3dic,
  title={Joint Navigation and Manipulation Planning with 3D Interaction Chains},
  author={Zhang, Keming and Zhang, Sixian and Song, Xinhang and Wang, Hongyu and Wang, Yiyao and Wang, Yingjie and Jiang, Shuqiang},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
