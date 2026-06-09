# Vision Banana 项目页（vision-banana.github.io）

> 来源归档（ingest 配套站点）

- **URL：** <https://vision-banana.github.io/>
- **对应论文：** [Image Generators are Generalist Vision Learners](https://arxiv.org/abs/2604.20329)（arXiv:2604.20329，Google DeepMind，2026-04）
- **机构出版物页：** <https://deepmind.google/research/publications/240658/>
- **入库日期：** 2026-06-09
- **一句话说明：** 官方落地页：交互式分割/深度/法线/点云演示、zero-shot benchmark 柱状图、贡献者名单与 BibTeX。

## 页面要点（2026-06 快照）

### Overview 三句话

1. Vision Banana 是 **图像理解与生成统一** 的 SOTA 模型。
2. **生成式视觉预训练** 是有效的视觉理解范式。
3. **图像生成** 可作为多样视觉任务的 **通用接口**。

### Capabilities（hover/tap 交互演示）

| 能力 | Prompt 模式示例 | 后处理 |
|------|-----------------|--------|
| **Semantic Segmentation** | 自然语言或 JSON 类–色映射；per-class 开放词汇 | RGB → 最近色类像素赋值 |
| **Instance Segmentation** | 「Each X is colored differently」+ 背景色 | 多阶段聚类提取实例 mask |
| **Referring Expression** | 动作/外观/多语言文本指代 | 单目标纯色 mask |
| **Metric Depth** | rainbow colormap 深度可视化 prompt | colormap 反演 → metric depth |
| **3D Point Cloud** | 由 metric depth + 相机内参 unproject | 项目页 WebGL 交互查看 |
| **Surface Normal** | 「Predict/Generate surface normal map」 | RGB 编码法线向量 |

### Results 柱状图（zero-shot transfer 主设定）

**2D：**

- Cityscapes mIoU：Vision Banana **69.9** vs SAM 3 **65.2**
- SA-Co/Gold cgF₁：Vision Banana + Gemini 3.1 Flash-Lite **47.5** vs OWLv2 **24.6**
- RefCOCOg cIoU：**73.8** vs SAM 3 + Gemini 2.5 Pro **73.4**
- ReasonSeg gIoU：Vision Banana + Gemini 2.5 Pro **79.3** vs SAM 3 Agent **77.0**

**3D：**

- Metric depth（6 benchmarks 平均 δ₁）：**0.929** vs Depth Anything 3 **0.918**（注明不用内参）
- Surface normal 平均角误差：**18.928°** vs Lotus-2 **19.642°**

### BibTeX

```bibtex
@article{visionbanana2026,
  title={Image Generators are Generalist Vision Learners},
  author={Gabeur, Valentin and Long, Shangbang and Peng, Songyou and ...},
  journal={arXiv preprint arXiv:2604.20329},
  year={2026}
}
```

## 对 wiki 的映射

- 与 [sources/papers/vision_banana_arxiv_2604_20329.md](../papers/vision_banana_arxiv_2604_20329.md) 配对
- 实体页：[wiki/entities/vision-banana.md](../../wiki/entities/vision-banana.md)
- 概念页：[wiki/concepts/generative-vision-pretraining.md](../../wiki/concepts/generative-vision-pretraining.md)
