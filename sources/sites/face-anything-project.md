# Face Anything 项目页（kocasariumut.github.io/FaceAnything）

> 来源归档（ingest 配套站点）

- **URL：** <https://kocasariumut.github.io/FaceAnything/>
- **对应论文：** [Face Anything: 4D Face Reconstruction from Any Image Sequence](https://arxiv.org/abs/2604.19702)（arXiv:2604.19702，TUM + Huawei Noah's Ark Lab，2026）
- **入库日期：** 2026-07-09
- **一句话说明：** 官方落地页：NeRSemble/VFHQ 重建与跟踪可视化、与 DAViD/Sapiens/V-DPM/Pixel3DMM 对比视频、数据集构建六步说明、架构五要点与 BibTeX。

## 页面要点（2026-07 快照）

### 核心主张

**Unified 4D facial reconstruction and dense tracking from image sequences via joint prediction of depth and canonical facial coordinates.**

### 结果展示（项目页视频/图）

| 基准 | 展示内容 |
|------|----------|
| **NeRSemble** | 输入视频 → 带跟踪的重建 |
| **VFHQ** | 输入视频 → 带跟踪的重建 |
| **数据集样例** | RGB / depth map / canonical coordinate map 三联 |

### 对比实验（项目页 Video 节）

| 任务 | 对照方法 |
|------|----------|
| 深度/重建 | DAViD、Sapiens |
| 动态重建 | V-DPM |
| 跟踪 | Pixel3DMM、V-DPM |

### 数据集构建（六步）

1. NeRSemble 同步多视角 + 标定相机
2. MediaPipe 驱动的表情/姿态采样选帧
3. COLMAP 重建 depth 与稠密点云
4. FLAME tracking 对齐到共享规范空间
5. FLAME 形变迁移到重建点 → canonical maps
6. 输出 RGB + depth + canonical maps（跨视角/时间一致）

### 架构（五步）

1. Transformer 联合预测 depth、ray maps、canonical facial maps
2. 跟踪 = canonical map prediction（非帧间运动估计）
3. DPT-style head，多图单前向
4. 规范空间最近邻 → 密集对应
5. DAViD 预训练 → canonical 监督微调

### BibTeX

```bibtex
@article{kocasari2026face,
  title={Face Anything: 4D Face Reconstruction from Any Image Sequence},
  author={Kocasari, Umut and Giebenhain, Simon and Shaw, Richard and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:2604.19702},
  year={2026}
}
```

## 对 wiki 的映射

- 与 [sources/papers/face_anything_arxiv_2604_19702.md](../papers/face_anything_arxiv_2604_19702.md) 配对
- 实体页：[wiki/entities/paper-face-anything-4d-face-reconstruction.md](../../wiki/entities/paper-face-anything-4d-face-reconstruction.md)
