# ZONDA: Zero-shot Object Navigation with Dynamic Avoidance in Multi-floor Environments（arXiv:2607.21025）

> 来源归档（ingest）

- **标题：** ZONDA: Zero-shot Object Navigation with Dynamic Avoidance in Multi-floor Environments
- **类型：** paper / ObjectNav / zero-shot / multi-floor / dynamic avoidance / VLM / Habitat
- **来源：** arXiv abs / PDF / HTML（v1，2026-07-23）
- **原始链接：**
  - <https://arxiv.org/abs/2607.21025>
  - PDF：<https://arxiv.org/pdf/2607.21025>
  - HTML：<https://arxiv.org/html/2607.21025v1>
- **作者：** Shaomin Liang, Xuanhong Liao, Shiyao Zhang（通讯作者：Shiyao Zhang）
- **机构：** 南方科技大学（SUSTech）；广东直驱科技有限公司（Direct Drive Tech）/ 华南理工大学（SCUT）；大湾区大学（Great Bay University）
- **入库日期：** 2026-07-24
- **一句话说明：** 零样本 ObjectNav 框架，用 **高度差可通行地图 + 启发式跨楼层规划**、**多视角 VLM 目标核验** 与 **行人轨迹预测避障**，在 HM3D / MP3D / 自建 HM3D-DYNA 上超过 ASCENT 等基线，并在 Direct Drive Tech **TITA** 轮腿双足上真机部署。

## 开源状态（核查 2026-07-24）

- **确认未开源 / 无项目页：** arXiv abs、HTML、PDF 均未列出 GitHub、Hugging Face、项目页或「code will be released」外链；公开 Web 检索未发现与本 ObjectNav 工作同名的官方可运行仓（无关同名仓库已排除）。
- **复现边界：** 方法依赖 SegFormer / SAM2 / RT-DETR·Grounding DINO·OWLv2、Qwen3-VL-Embedding-2B + Qwen3-VL-Flash、Habitat + HM3D/MP3D；无公开代码/权重时无法按官方入口复现；真机侧为 ROS 2 + 远端 RTX 5060 Ti + MPPI。
- **互指：** 升格实体页 [`wiki/entities/paper-zonda.md`](../../wiki/entities/paper-zonda.md)

## 摘要级要点

- **问题：** 现有零样本 ObjectNav 多假设 **静态单层**；跨楼层常绑平台相关 RL PointNav（如 ASCENT）；单视角确认易假阳性；静态图把行人当永久障碍。
- **三模块：** (i) Heuristic Multi-Floor Planner（高度差可通行 + 楼梯语义 + 拓扑图 \(\mathcal{G}_{\text{topo}}\)）；(ii) Multi-View Target Verification（多尺度观测缓冲 + VLM 联合判定）；(iii) Dynamic Pedestrian Avoidance（Kalman 恒速预测 \(T_{\text{pred}}=3\) s，占据膨胀 0.5 m）。
- **地图：** Object / Semantic / Traversable 三图，网格 **0.1 m**；语义图用 LLM 推房间类型 + VLM 余弦相似度 EMA（\(\alpha=0.9\)）。
- **规划：** 前沿 DBSCAN 分块 + 语义奖励选块 + 块内 TSP（最近邻启发式）；楼层切换靠楼梯候选（语义「stairs」且 \(\Delta h < H_{\text{agent}}\)）。
- **评测：** HM3D SR **66.5%** / SPL 33.0%；MP3D SR **48.2%** / SPL **21.5%**（相对 ASCENT +3.7 / +6.0 pp）；HM3D-DYNA SR **48.8%** vs ASCENT 30.9%。
- **消融：** 去掉多视角核验 → HM3D SR 66.5→**41.5%**；去掉跨楼层 → 57.8%；去掉启发式分块 → 62.6%。
- **真机：** Direct Drive Tech TITA；ROS 2 远端推理；MPPI 连续速度；办公场景避行人找垃圾桶。

## 核心论文摘录（MVP）

### 1) 动机：单层静态零样本 ObjectNav 的三缺口

- **链接：** <https://arxiv.org/abs/2607.21025> §I
- **摘录要点：** VLFM / ApexNav 等把 3D 压成 2D 网格，跨层失败；ASCENT 用 LLM 楼层感知但仍依赖平台绑定的 RL PointNav；单视角确认易把相似物/阴影当目标；MP3D/HM3D 默认静态，行人被写进永久占据。
- **对 wiki 的映射：**
  - [ZONDA](../../wiki/entities/paper-zonda.md)
  - [视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md) — ObjectNav 零样本 / 跨楼层分支
  - [VLN 四范式复现路径](../../wiki/overview/vln-open-source-repro-paradigms.md) — 与 VLFM「地图+语义前沿」对照

### 2) 方法：OST 地图 + 启发式跨楼层 + 多视角核验 + 行人预测

- **链接：** §III
- **摘录要点：** Object-Semantic-Traversable Map 供探索；\(H_{\text{agent}}\) 按平台穿越能力设定，楼梯用语义×可通行联合判定；多视角缓冲按质量 \(Q\) 取近远场送 VLM；行人 Kalman + 匈牙利关联，预测轨迹作临时障碍；离散 Habitat 动作用 A*→原语，连续场景用 MPPI。
- **对 wiki 的映射：**
  - [ZONDA](../../wiki/entities/paper-zonda.md)
  - [Habitat-Sim](../../wiki/entities/habitat-sim.md)
  - [Uni-LaViRA](../../wiki/entities/paper-uni-lavira.md) — 同为零样本 ObjectNav 但训练自由三层翻译对照

### 3) 评测：静态 SoTA + HM3D-DYNA + 真机 TITA

- **链接：** §IV；Table I–III；Fig. 5
- **摘录要点：** 静态多楼层上 HM3D/MP3D 零样本前列；动态榜相对 ASCENT 大幅回升；消融显示多视角核验对 SR 影响最大；真机用与 HM3D 相同的非平台参数，仅调 \(H_{\text{agent}}\) 与安全膨胀。
- **对 wiki 的映射：**
  - [ZONDA](../../wiki/entities/paper-zonda.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md) — Habitat→轮腿双足连续控制迁移读法

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-zonda.md`](../../wiki/entities/paper-zonda.md) — 主实体页
- [`wiki/tasks/vision-language-navigation.md`](../../wiki/tasks/vision-language-navigation.md) — 室内 ObjectNav / 跨楼层 / 动态避障分支
- [`wiki/overview/vln-open-source-repro-paradigms.md`](../../wiki/overview/vln-open-source-repro-paradigms.md) — 相对 VLFM 范式的扩展读法（本文暂未开源）
- [`wiki/entities/habitat-sim.md`](../../wiki/entities/habitat-sim.md) — HM3D/MP3D ObjectNav 仿真宿主
- [`wiki/entities/paper-uni-lavira.md`](../../wiki/entities/paper-uni-lavira.md) — 零样本 ObjectNav 对照
- [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md) — 仿真离散动作 → 真机 MPPI

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2607.21025>
