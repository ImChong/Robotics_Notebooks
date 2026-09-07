# ReViV：单目 egocentric 视频统一 4D 重建（viewer + view）

> 来源归档（ingest）

- **标题：** ReViV: Reconstructing the Viewer and the View in 4D from Monocular Egocentric Video
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2607.17790>
  - <https://reviv4d.github.io/>
  - <https://github.com/lvsean/reviv4d>
- **机构：** ETH Zürich · Delft University of Technology · Microsoft
- **会议：** ECCV 2026
- **入库日期：** 2026-09-07
- **一句话说明：** 首个从单目 egocentric RGB 统一重建 wearer 全身/双手/注视与场景深度、相机轨迹的 feed-forward 框架；Masked Generative Egocentric Transformer（MGET）在 7B unique tokens / 500B training tokens 上学习多模态联合分布，推理比 EgoAllo 快 100×、比 Dyn-HaMR 快 400×；代码与权重已开源（权重限非商用）。

## 核心摘录（MVP）

### 1) 问题：viewer 与 view 被割裂建模

- **摘录要点：** 既有 egocentric 方法要么只做场景几何（深度/相机），要么只做人体姿态，且常依赖 SLAM 轨迹、点云或外接手部模块；场景与人体时序不一致，推理慢。
- **对 wiki 的映射：**
  - [ReViV 论文实体页](../../wiki/entities/paper-reviv4d.md)
  - [EgoM2P 策展索引](../../wiki/entities/paper-sa-2506-07886-egom2p-egocentric-multimodal-multitask-pretraini.md) — 前序 scene-centric 多任务预训练

### 2) 方法：联合分布 + MGET + 统一 token 化

- **摘录要点：** 观测 \(\mathcal{X}=\{\text{RGB}\}\)，重建 \(\mathcal{Y}=\{\text{hand, body, gaze, depth, cam}\}\)。各模态经 VQ-VAE（body/hand 自训 transformer VQ-VAE；RGB/depth 用 Cosmos；gaze/cam 扩展 EgoM2P tokenizer）离散化后，T5 式 12+12 层 MGET 用随机 mask 学习 \(p(\mathcal{Z})\)；推理时以 RGB 为条件迭代解码全部模态。轻量 floor fitting（或可选 VIPE 度量深度锚）对齐 metric 4D 坐标。
- **对 wiki 的映射：**
  - [ReViV 论文实体页](../../wiki/entities/paper-reviv4d.md) — 流程总览与源码运行时序图

### 3) 数据引擎：4B → 7B tokens，统一运动表示

- **摘录要点：** 在 EgoM2P 场景数据上扩展 HoloAssist/HOT3D/ARCTIC/TACO/H2O/EgoGen/Nymeria 等的手与全身标注；Video Depth Anything 生成时序一致深度伪标签。手部用相机系、全身用重力对齐世界系两套 kinematic 表示。
- **对 wiki 的映射：**
  - [ReViV 论文实体页](../../wiki/entities/paper-reviv4d.md) — 数据与预训练

### 4) 评测：多任务 SOTA + 速度

- **摘录要点：** ADT 上全身 PA-MPJPE **88.6**、Similarity **0.751**、FID **0.442**，**0.7 s/clip**，优于需 VIPE/GT 相机的 EgoAllo/UniEgoMotion。四数据集手部 PA-MPJPE 全面领先 HaMeR/Dyn-HaMR（**0.7 s** vs 72–280 s）。相机 ATE **0.015**、注视 MSE **0.0211**；深度 Abs Rel **0.265**（弱于继承 UniDepth 的 EgoMono4D 但快 20×+）。
- **对 wiki 的映射：**
  - [ReViV 论文实体页](../../wiki/entities/paper-reviv4d.md) — 评测表

### 5) 开源状态（截至 2026-09-07）

- **摘录要点：** **已开源** — 项目页链 [lvsean/reviv4d](https://github.com/lvsean/reviv4d)；Apache 2.0 代码 + Sample Code License（**非商用**）权重；polybox 发布 `metric_depth/` 与 `reviv_500b/` 两套 checkpoint；Cosmos tokenizer 需 HF gated 下载；训练数据不随仓库分发。
- **对 wiki 的映射：**
  - [ReViV 项目页](../sites/reviv4d.md)
  - [ReViV 官方仓库](../repos/reviv4d.md)
  - [ReViV 论文实体页](../../wiki/entities/paper-reviv4d.md) — 工程实践

## 当前提炼状态

- [x] arXiv + 项目页 + GitHub README 对齐摘录
- [x] 步骤 2.5 开源核查：代码+权重已发布，权重非商用
- [x] wiki 映射：`wiki/entities/paper-reviv4d.md` 新建
