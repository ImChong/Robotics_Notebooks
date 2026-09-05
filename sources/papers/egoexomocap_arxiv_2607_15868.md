# EgoExoMoCap: Distributed Ego-Exo Human Motion Capture

> 来源归档（ingest）

- **标题：** EgoExoMoCap: Distributed Ego-Exo Human Motion Capture
- **类型：** paper
- **机构：** Meta Reality Labs；苏黎世联邦理工学院（ETH Zürich）SIPLAB
- **venue：** ECCV 2026 Spotlight
- **原始链接：**
  - <https://arxiv.org/abs/2607.15868>
  - PDF：<https://arxiv.org/pdf/2607.15868>
  - 项目页：<https://siplab.org/projects/EgoExoMoCap>
  - 代码仓（占位）：<https://github.com/eth-siplab/EgoExoMoCap>
- **入库日期：** 2026-09-05
- **一句话说明：** 分布式 HMD 人体动捕：两名及以上佩戴智能眼镜的受试者互为观测者，将连续 egocentric 头/腕轨迹与间歇 exocentric RGB 观测在 wearer 坐标系下融合（EgoNet 粗定位 → ViTPose 2D 关键点 → DINOv3 置信门控射线 → Spatial/Temporal Transformer），在 Nymeria 与 EgoHumans 野外数据上优于纯 ego / 纯 exo 与朴素融合基线。

## 核心论文摘录（策展）

### 1) 问题：ego 与 exo 动捕长期割裂，多 HMD 互为观测未被利用

- **摘录要点：** 传统动捕依赖多相机棚或惯性服；纯 ego（AvatarPoser / EgoPoser 等）下肢不可见、遮挡下失真；纯 exo（PromptHMR 等）全局尺度与严重遮挡脆弱。现代 HMD 支持分布式信息交换与时间对齐，使「每人既是被跟踪者又是他人移动观测者」成为可能。
- **对 wiki 的映射：**
  - [paper-egoexomocap](../../wiki/entities/paper-egoexomocap.md) — 一句话定义与「为什么重要」。

### 2) 方法：EgoNet 引导 exo ROI + 射线规范化 + DINOv3 门控融合

- **摘录要点：**
  - **输入：** observer RGB + 双方头 6D 位姿；wearer 头（+ 可选双腕）轨迹；1-point / 3-point 两种配置。
  - **EgoNet：** 仅 ego 信号粗估 SMPL，投影到 exo 图得 ROI（优于 YOLO 遮挡场景）。
  - **Exo tokens：** ViTPose 13 关键点 → 按 observer–wearer 头距深度缩放 → 变换到 wearer 头局部坐标系（ray canonicalization）。
  - **DINOv3 gating：** 学习 per-keypoint 置信，遮挡/噪声时自动降权 exo。
  - **融合：** Spatial Transformer 聚合 ego+exo → Temporal Transformer 输出全局 SMPL 序列。
- **对 wiki 的映射：**
  - [paper-egoexomocap](../../wiki/entities/paper-egoexomocap.md) — 流程总览 Mermaid + 核心机制。

### 3) 评测：Nymeria 主训练 + EgoHumans 跨数据集；多 observer 可进一步降误差

- **摘录要点：**
  - **Nymeria（80/20）：** 3-point MPJPE **5.72 cm**（EgoPoser 7.74 / PromptHMR+EgoPoser 6.47）；1-point **8.28 cm**。
  - **EgoHumans：** 3-point MPJPE **7.62 cm**；多 observer 子集 **7.11 cm**（单 observer 约 8.8–11 cm）。
  - **消融：** 去 ego 或 exo 均显著退化；DINO 门控优于 ViT score / 硬 mask；EgoNet 加噪鲁棒（σ=10 cm 仅 +0.54 cm MPJPE）。
- **对 wiki 的映射：**
  - [paper-egoexomocap](../../wiki/entities/paper-egoexomocap.md) — 实验与评测 + 结论。

### 4) 局限与开源边界

- **摘录要点：** 聚焦野外 GT 采集而非在线实时；假设 Aria 标定/同步；体型需已知；wearer 被他人大面积遮挡时 DINO 门控可能失效；长时 out-of-view 时下肢精度向 ego-only 退化。
- **项目页 / 源码核查（2026-09-05）：** GitHub `eth-siplab/EgoExoMoCap` 已建仓（MIT），README 写明 **Code coming soon**，截至入库日无可运行训练/推理入口。
- **对 wiki 的映射：**
  - [sources/sites/egoexomocap-siplab.md](../sites/egoexomocap-siplab.md)、[sources/repos/egoexomocap.md](../repos/egoexomocap.md)

## 对 wiki 的映射

- [paper-egoexomocap](../../wiki/entities/paper-egoexomocap.md)
- 相关姊妹：[paper-notebook-egoposer](../../wiki/entities/paper-notebook-egoposer-robust-real-time-egocentric-pose-estima.md)、[paper-notebook-avatarposer](../../wiki/entities/paper-notebook-avatarposer-articulated-full-body-pose-tracking.md)、[ego-category-04-ego-exo-fusion](../../wiki/overview/ego-category-04-ego-exo-fusion.md)

## BibTeX

```bibtex
@article{jiang2026egoexomocap,
  title={EgoExoMoCap: Distributed Ego-Exo Human Motion Capture},
  author={Jiang, Jiaxi and Bhatnagar, Bharat Lal and Yang, Nan and Ma, Lingni and Starke, Sebastian and Kips, Robin and Bertsch, Nadine and Holz, Christian and Bogo, Federica},
  journal={arXiv preprint arXiv:2607.15868},
  year={2026}
}
```
