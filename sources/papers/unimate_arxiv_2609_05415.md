# UniMate（跨拓扑骨骼文本驱动动画）

> 来源归档（ingest）

- **标题：** UniMate: One Unified Model to Animate Diverse Skeletons
- **类型：** paper / 3D 角色动画 / 跨拓扑 text-to-motion / 扩散 Transformer
- **arXiv：** <https://arxiv.org/abs/2609.05415>（PDF：<https://arxiv.org/pdf/2609.05415>）
- **会议：** SIGGRAPH Asia 2026 Conference Papers（Kuala Lumpur）
- **项目页：** <https://linzhanmou.com/unimate/>
- **机构：** 普林斯顿大学（Princeton）、加州大学伯克利分校（UC Berkeley）、麻省理工学院（MIT）、南洋理工大学（NTU）
- **作者：** Linzhan Mou、Jiahui Lei、Zhiyang Dou、Chenyue Cai、Chaoyue Song、Adam Finkelstein、Szymon Rusinkiewicz
- **入库日期：** 2026-09-30
- **一句话说明：** 给定 **已 rig 的 3D 资产 + 文本 prompt**，单一 **TADiT（Topology-Aware Diffusion Transformer）** 在 **无 per-skeleton 微调、无 test-time optimization** 下为任意拓扑骨骼合成动作；配套 **UniML3D**（13,006 条跨物种/刚体关节序列 + 统一 canonicalization）。

## 摘要级要点

- **动机：** 自动 rigging 已规模化，但现有 learned animator 多绑定 SMPL/SMAL 等模板，或 inference 需参考 motion / per-skeleton 微调。
- **TADiT 三件套：** (1) 图距离 + 边类型 + 深度的 **graph-aware attention bias**；(2) 图 Laplacian 谱上的 **Spec-RoPE**；(3) rest-pose skeleton token **AdaLN-Zero 全局拓扑条件**。
- **训练：** **Flow matching**（`training.diff_model = "flow"`）+ masked L2 + geodesic rotation + velocity smoothness；**classifier-free guidance**；在线 skeleton augmentation（加关节、删叶、链池化、骨长扰动）。
- **UniML3D：** Truebones + Mixamo + Objaverse-XL 经 16 步 filter/annotate/canonicalize；3584 唯一文本 prompt；Hugging Face 发布处理管线与特征。
- **零样本应用：** cross-topology transfer、in-betweening、expansion、text-guided editing（固定部分 joint token 再采样）。
- **开源（步骤 2.5，2026-09-30）：** 项目页链 **GitHub**、**HF 数据集/权重**；官方仓 <https://github.com/Friedrich-M/UniMate>（2026-09-06 释训练/推理代码）。

## 核心论文摘录（MVP）

### 1) 统一表示与 TADiT 管线

- **链接：** arXiv §3；Fig. 3
- **摘录要点：** joint token 流联合 rest-pose 运动学与 motion manifold；文本经 T5（默认 `google/flan-t5-base`）编码。
- **对 wiki 的映射：**
  - [UniMate](../../wiki/entities/paper-unimate.md) — 方法与流程总览

### 2) UniML3D 与 canonicalization

- **链接：** 项目页 Dataset 表；论文 §3.1
- **摘录要点：** 13,006 clips；y-up、直径缩放、相对 rest-pose 旋转、特征归一化；Truebones 商业包本体需自购。
- **对 wiki 的映射：**
  - [UniMate](../../wiki/entities/paper-unimate.md) — 数据与工程实践

### 3) 实验与零样本编辑

- **链接：** 项目页 Applications；论文 Fig. 12–15
- **摘录要点：** 单模型跨 biped / quadruped / avian / marine / insect / serpentine / articulated rigid；in-between / expansion / partial joint 固定编辑。
- **对 wiki 的映射：**
  - [UniMate](../../wiki/entities/paper-unimate.md) — 结论与局限

### 4) 官方代码与权重

- **链接：** [GitHub README](https://github.com/Friedrich-M/UniMate)；[HF UniMate](https://huggingface.co/Linzhan/UniMate)
- **摘录要点：** `unimate.training.train` / `unimate.inference.sample`；released checkpoints 与 `outputs/<exp>/config.json` 布局一致；`data_process/` 五阶段到 GLB/FBX 动画导出。
- **对 wiki 的映射：**
  - [unimate-friedrich-m](../repos/unimate-friedrich-m.md)
  - [unimate-linzhanmou](../sites/unimate-linzhanmou.md)

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-unimate.md`](../../wiki/entities/paper-unimate.md)
- 项目页核查：[`sources/sites/unimate-linzhanmou.md`](../sites/unimate-linzhanmou.md)
- 代码归档：[`sources/repos/unimate-friedrich-m.md`](../repos/unimate-friedrich-m.md)
- 互链：[Awesome Text-to-Motion（Zilize）](../../wiki/entities/awesome-text-to-motion-zilize.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)、[Skeleton Action Recognition](../../wiki/methods/skeleton-action-recognition.md)
