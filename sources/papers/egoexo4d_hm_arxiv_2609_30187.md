# Ego-Exo4D Human Meshes Dataset: 4D Human Motion Reconstruction for Ego-Exo Captures

> 来源归档（ingest）

- **标题：** Ego-Exo4D Human Meshes Dataset: 4D Human Motion Reconstruction for Ego-Exo Captures
- **类型：** paper
- **机构：** 德克萨斯大学奥斯汀分校（The University of Texas at Austin）
- **原始链接：**
  - <https://arxiv.org/abs/2609.30187>
  - PDF：<https://arxiv.org/pdf/2609.30187>
  - 项目页：<https://abhiram824.github.io/egoexo4d_human_meshes/>
- **入库日期：** 2026-09-26
- **一句话说明：** 在 Ego-Exo4D 同步多视角标定视频上，用改造 SLAHMR 管线（Mask R-CNN 选 wearer → ViTPose + HaMeR → 多视角三角化 → 冻结标定的 SMPL-H 优化）批量重建 4D 人体网格，经质量过滤后发布 **Ego-Exo4D-HM**（2,649 takes、104.59 h 重建运动；配套 MIT 代码与 Hugging Face 预计算 npz）。

## 核心论文摘录（策展）

### 1) 动机：Ego-Exo4D 有视频与稀疏 3D 姿态，缺可直接用的稠密 4D 人体运动

- **摘录要点：** Ego-Exo4D 提供同步 ego + 多 exo 技能活动视频，服务具身 AI、程序性活动理解等；原始发布仅 **稀疏** 3D 人体关键点标注，从多视角 RGB 恢复 **稠密、度量、带手** 的 4D 运动仍非平凡。机器人侧 increasingly 用人活动视频 / MoCap 训练人形控制器或 visuomotor 策略，但「只拍视频」与「有可执行 3D 监督」之间缺一层 **可复用中间表示**。
- **对 wiki 的映射：**
  - [paper-egoexo4d-hm](../../wiki/entities/paper-egoexo4d-hm.md) — 一句话定义与「为什么重要」。

### 2) 方法：三阶段 SLAHMR 改造 + Ego-Exo4D 标定/SLAM 先验

- **摘录要点：**
  - **单视角：** Mask R-CNN（RegNetY-4GF）检测人框，选包含 Aria SLAM 投影头位置的框为 wearer；ViTPose 身体 2D + HaMeR 手部 2D，共 **67** 关键点。
  - **三角化：** 多 exo 视角 + 标定外参，将 2D 观测三角化为 3D 点（与官方 sparse GT 同世界系）。
  - **SMPL-H 优化：** 在冻结相机参数下优化每帧根平移/朝向、身体+手 pose、每 take 单一 shape；**不用** SLAHMR 运动先验（三角化 3D 已约束全局运动）。输出 mesh 顶点与 67 关节。
- **对 wiki 的映射：**
  - [paper-egoexo4d-hm](../../wiki/entities/paper-egoexo4d-hm.md) — 流程总览 Mermaid + 核心机制。

### 3) 数据集规模与质量过滤

- **摘录要点：**
  - 约 **3,200** 条 Ego-Exo4D 视频跑管线；**551** takes（17.1%）因质量过滤剔除，保留 **2,649** takes、**104.59 h** 重建运动时长。
  - 过滤：**重投影自洽**（>10% 样本像素误差 >50 px 则弃）；**三角化覆盖率**（<50% 关键点成功三角化则弃）。
  - 每 take 四路 exo + 一路 ego，对应 **522.96 h** 原始视频（项目页概览常写 ~523 h）。
  - 发布：每 take 一个 merged **npz**（SMPL-H 参数、标定相机、3D 关节与各视角 2D 重投影）；HF **`Ego-Exo4D-HM/npz-datasets`**，2649 takes、~48GB。
- **对 wiki 的映射：**
  - [paper-egoexo4d-hm](../../wiki/entities/paper-egoexo4d-hm.md) — 数据集速查 + 工程实践。

### 4) 评测与机器人读法

- **摘录要点：** 对 Ego-Exo4D 提供的 GT 3D 身体/手关键点，在 **度量世界系** 下报告 **global MPJPE**（无 Procrustes 对齐）：身体 **56.21 mm**（845 annotated takes）、手部 **51.59 mm**（190 takes）。论文讨论人形 WBC、人视频→策略、动作条件世界模型等下游，但 **未** 在本工作中训练机器人策略验证「照着学」上限。
- **对 wiki 的映射：**
  - [paper-egoexo4d-hm](../../wiki/entities/paper-egoexo4d-hm.md) — 评测 + 结论 + 局限。

### 5) 开源边界（步骤 2.5 · 2026-09-26）

- **项目页：** Installation / Pipeline / Download 文档完整；声明 **MIT**，基于 SLAHMR；GitHub 链 `Abhiram824/egoexo4d_human_meshes`。
- **代码入口：** `scripts/run_pipeline.py`（5 阶段：去畸变 → 相机矩阵 → 检测+ViTPose+HaMeR → 三角化 → `slahmr/run_opt.py`）；可 `hf download Ego-Exo4D-HM/npz-datasets` 跳过自跑管线。
- **依赖：** 须先下载 **原始 Ego-Exo4D** 并设 `EGOEXO4D_DATASET`；渲染仍从 raw take 抽帧。
- **开放程度：** **已开源**（代码 + 预计算数据）；非「从零视频」——强绑定 Ego-Exo4D 生态与 GPU/CUDA 编译链。
- **对 wiki 的映射：**
  - [sources/sites/egoexo4d-hm-abhiram824.md](../sites/egoexo4d-hm-abhiram824.md)、[sources/repos/egoexo4d-human-meshes.md](../repos/egoexo4d-human-meshes.md)

## 对 wiki 的映射

- [paper-egoexo4d-hm](../../wiki/entities/paper-egoexo4d-hm.md)
- 姊妹：[paper-ego4d](../../wiki/entities/paper-ego4d.md)（上游语料）、[paper-human-as-humanoid](../../wiki/entities/paper-human-as-humanoid.md)（ego-exo → 机器人标签下游范式）、[paper-data-pyramid-embodied-manipulation](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)（Ego/Exo 数据层）

## BibTeX

```bibtex
@article{maddukuri2026egoexo4dhm,
  title={Ego-Exo4D Human Meshes Dataset: 4D Human Motion Reconstruction for Ego-Exo Captures},
  author={Maddukuri, Abhiram and Pavlakos, Georgios},
  journal={arXiv preprint arXiv:2609.30187},
  year={2026}
}
```
