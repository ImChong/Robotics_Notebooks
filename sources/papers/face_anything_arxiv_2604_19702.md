# Face Anything: 4D Face Reconstruction from Any Image Sequence

> 来源归档（ingest）

- **标题：** Face Anything: 4D Face Reconstruction from Any Image Sequence
- **类型：** paper
- **来源：** arXiv abs；项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2604.19702>
  - <https://kocasariumut.github.io/FaceAnything/>
- **作者：** Umut Kocasarı¹, Simon Giebenhain¹, Richard Shaw², Matthias Nießner¹（¹Technical University of Munich；²Huawei Noah's Ark Lab）
- **入库日期：** 2026-07-09
- **一句话说明：** 用 **canonical facial point prediction** 把动态人脸 **密集跟踪** 与 **4D 重建** 统一为单次前向的 **深度 + 规范面部坐标** 联合预测；Transformer + DPT 式头，在 NeRSemble 派生大规模监督上训练，相对 V-DPM / Pixel3DMM 等动态重建路线约 **3×** 更低对应误差、**16%** 更好深度精度且推理更快。

## 核心论文摘录（MVP）

### 1) 问题与核心表征：canonical facial coordinates

- **链接：** <https://kocasariumut.github.io/FaceAnything/>（Abstract）
- **摘录要点：**
  - 图像序列动态人脸同时面临 **非刚性形变、表情变化、视角变化**，几何与对应估计高度歧义。
  - 提出 **canonical facial point prediction**：为每个像素分配共享 **规范空间** 中的归一化面部坐标。
  - 将 **密集跟踪** 与 **动态重建** 转化为 **规范空间重建** 问题，单次前向模型内实现 **时序一致几何** 与 **可靠对应**。
  - **联合预测 depth + canonical coordinates** → 准确深度、时序稳定重建、密集 3D 几何、鲁棒面部点跟踪。
- **对 wiki 的映射：**
  - [Face Anything](../../wiki/entities/paper-face-anything-4d-face-reconstruction.md) — 表征定义与统一任务表述。
  - [视觉表征作为策略输入](../../wiki/concepts/visual-representation-for-policy.md) — 前馈几何/对应作为机器人感知上游的对照谱系。

### 2) 架构：Transformer 联合预测，跟踪即规范图预测

- **链接：** <https://kocasariumut.github.io/FaceAnything/>（Architecture）
- **摘录要点：**
  - **Transformer** 架构从一张或多张输入图 **联合预测 depth、ray maps、canonical facial maps**。
  - **不估计帧间运动**；跟踪表述为 **canonical map prediction**（每像素规范坐标）。
  - 多图输入经 **DPT-style head** 一次前向输出几何与对应。
  - 规范空间 **最近邻搜索** 得密集对应，实现高效、时序一致跟踪。
  - **两阶段训练**：先在 **DAViD** 预训练面部先验，再以 canonical 监督微调。
- **对 wiki 的映射：**
  - [Face Anything](../../wiki/entities/paper-face-anything-4d-face-reconstruction.md) — Mermaid 管线与训练阶段表。

### 3) 数据集：NeRSemble 多视角 → COLMAP 几何 → FLAME 规范对齐

- **链接：** <https://kocasariumut.github.io/FaceAnything/>（Dataset Creation）
- **摘录要点：**
  - 基于 **NeRSemble** 同步多视角、标定相机构建大规模数据。
  - 按 MediaPipe 估计做 **表情与姿态采样** 选帧。
  - 每时刻 **COLMAP** 重建高质量几何得 depth 与稠密点云。
  - **FLAME tracking** 将几何对齐到跨帧/跨身份的共享规范空间。
  - 将 FLAME 形变 **迁移** 到重建点生成 **canonical maps**，为对应学习提供稠密监督。
  - 最终样本含 **RGB、depth、canonical maps**，跨视角与时间几何/对应一致。
- **对 wiki 的映射：**
  - [Face Anything](../../wiki/entities/paper-face-anything-4d-face-reconstruction.md) — 数据管线分节。
  - [GVHMR](../../wiki/entities/gvhmr.md) — 人体视频重建上游对照（全身 SMPL vs 面部 4D）。

### 4) 实验：重建 + 跟踪 SOTA，相对动态重建基线显著加速

- **链接：** <https://kocasariumut.github.io/FaceAnything/>（Video / Results）
- **摘录要点：**
  - 评测：**NeRSemble、VFHQ** 等图像/视频基准。
  - 深度对照：**DAViD、Sapiens** 等面部深度方法。
  - 动态重建/跟踪对照：**V-DPM、Pixel3DMM**。
  - 相对既有动态重建：**约 3× 更低 correspondence error**、**16% 深度精度提升**、**更快推理**。
  - 模型、数据集与代码 **将公开**（截至项目页声明）。
- **对 wiki 的映射：**
  - [Face Anything](../../wiki/entities/paper-face-anything-4d-face-reconstruction.md) — 结果摘要与基线对照表。
  - [Vision Banana](../../wiki/entities/vision-banana.md) — 另一路前馈 3D 几何/深度专家对照。

### 5) 相关技术谱系：前馈几何重建与参数化人脸

- **链接：** <https://kocasariumut.github.io/FaceAnything/>（Related Links）
- **摘录要点：**
  - **前馈几何重建：** Depth Anything 3 (DA3)、VGGT、Pi3。
  - **参数化人脸：** FLAME（规范空间对齐依赖）。
  - **数据：** NeRSemble、VFHQ、Ava-256、CelebV-HQ。
  - Face Anything 把 **面部专用前馈重建** 与 **规范对应** 绑定，区别于通用场景几何模型与纯优化式动态人脸跟踪。
- **对 wiki 的映射：**
  - [生成式视觉预训练](../../wiki/concepts/generative-vision-pretraining.md) — 前馈 3D 感知范式背景。
  - [humanoid-training-data-pipeline](../../wiki/queries/humanoid-training-data-pipeline.md) — 视频→3D 人体/面部上游在数据管线中的位置。

## 当前提炼状态

- **状态：** MVP 摘录完成；待代码/权重公开后补充工程入口与延迟数据。
