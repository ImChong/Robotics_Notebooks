# PASSAGE（场景对齐人形感知穿越）

> 来源归档（ingest）

- **标题：** PASSAGE: Scaling Scene-Aligned Motion Learning for Perceptive Humanoid Traversal in Cluttered Environments
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.18732>
- **机构：** 银河通用（Galbot）；北京大学；清华大学；上海交通大学；南开大学；上海期智研究院（作者网络与 [HumanTracker](../papers/humantracker_arxiv_2608_13555.md) 重叠，含 He Wang†、Li Yi†）
- **作者：** Yuxuan Ma、Zicheng Zeng、Chunlin Peng、Zhoujian Li、Zetong Zhao、Zhikai Zhang、Yunrui Lian、Han Xue、Sikai Liang、Weiyi Zhu、Mulin Chen、Chenghuai Lin、Jiayuan Gu、Jilong Wang、Jingbo Wang、He Wang、Li Yi
- **入库日期：** 2026-09-18
- **一句话说明：** VR+动捕采集 **100 h / 1500**  clutter 场景对齐人体运动；条件 flow-matching **planner**（6.25 Hz）+ 感知 **whole-body tracker**（50 Hz）在 Jetson Orin 全 onboard 穿越，无技能标注。

## 核心摘录（MVP）

### 1) 单一 planner–tracker 覆盖多 traversal 行为

- **摘录要点：** 不依赖任务专用 RL 目标或分技能策略库；planner 从 motion history、局部目标与 robot-centric 多层 elevation map 生成短 horizon 参考，tracker 以几何反馈 50 Hz 执行；real-time chunking 保 chunk 间一致。
- **对 wiki 的映射：**
  - [PASSAGE](../../wiki/entities/paper-passage.md) — 架构
  - [楼梯与障碍感知 locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md) — 人形穿越任务轴

### 2) 数据 scaling 与 post-training

- **摘录要点：** 三 seed 下数据从 6 h → 100 h，held-out 场景 mean contact-free success **48.1% → 68.9%**；加 validated scene augmentation 达 **70.3%**。Planner 侧在冻结 tracker 下 RL post-training 进一步提升闭环。
- **对 wiki 的映射：**
  - [PASSAGE](../../wiki/entities/paper-passage.md) — scaling 读法
  - [SSR](../../wiki/entities/paper-ssr-humanoid-open-world-traversal.md) — 另一条 onboard 感知穿越对照

### 3) 全 onboard 实机

- **摘录要点：** 自中心 3D LiDAR、在线 occupancy mapping、**6.25 Hz** 规划 + **50 Hz** 控制，Jetson AGX Orin；**50** 个未见物理布局零预建图、无 offload 穿越。
- **对 wiki 的映射：**
  - [PASSAGE](../../wiki/entities/paper-passage.md) — 部署读法
  - [Humanoid locomotion](../../wiki/tasks/humanoid-locomotion.md)

### 4) 开源状态（截至 2026-09-18）

- **摘录要点：** **截至入库日 arXiv v1 未列官方项目页或 GitHub**；步骤 2.5 仅确认论文 PDF/摘要，无代码链接。后续 lint 跟进。
- **对 wiki 的映射：**
  - [PASSAGE](../../wiki/entities/paper-passage.md) — 局限节

## 当前提炼状态

- [x] arXiv 摘要已对齐
- [x] 无项目页 — 开源标「未列链接」
- [x] wiki 映射：`wiki/entities/paper-passage.md`
