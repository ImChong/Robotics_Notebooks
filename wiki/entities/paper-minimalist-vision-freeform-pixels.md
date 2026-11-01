---
type: entity
project_id: minimalist-vision-freeform-pixels
tags:
  - paper
  - computer-vision
  - computational-imaging
status: complete
updated: 2026-11-01
arxiv: "2501.00142"
related:
  - ../concepts/visual-representation-for-policy.md
  - ../overview/cv-best-papers-2023-2026-technology-map.md
sources:
  - ../../sources/papers/minimalist_vision_freeform_pixels_arxiv_2501_00142.md
  - ../../sources/blogs/wechat_shenlan_cv_best_papers_2023_2026_labs.md
summary: "Minimalist Vision（arXiv:2501.00142）：Minimalist Vision 用任意形状的自由形态像素（光电探测器 + 光学掩膜）构成极简相机，把相机硬件建模成神经网络第一层，用极少像素完成特定视觉任务。"
---

# Minimalist Vision（arXiv:2501.00142）

**Minimalist Vision**（*Minimalist Vision with Freeform Pixels*，[arXiv:2501.00142](https://arxiv.org/abs/2501.00142)）来自 [深蓝具身智能 2026-11-01 盘点](../../sources/blogs/wechat_shenlan_cv_best_papers_2023_2026_labs.md)，属「计算成像与极简视觉」组。

## 一句话定义

**Minimalist Vision 用任意形状的自由形态像素（光电探测器 + 光学掩膜）构成极简相机，把相机硬件建模成神经网络第一层，用极少像素完成特定视觉任务。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Freeform Pixel | Freeform Pixel | 任意形状的光电探测像素 |
| CAVE | Computer Vision Laboratory (Columbia) | 哥伦比亚大学计算机视觉实验室 |

## 为什么重要

- 少即是多：用极少像素降低功耗并从物理层面保护隐私（公众号归属：哥伦比亚大学 CAVE 实验室，作者 Klotz 与 Nayar）。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-01）。arXiv 摘要页未列代码或项目页链接；本次未做进一步检索，不能判定是否开源。
- 同期同组工作见 [从视觉编程到 4D 重建：CV 三大会 2023–2026 Best Paper 阅读坐标](../overview/cv-best-papers-2023-2026-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2501.00142](https://arxiv.org/abs/2501.00142) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 按任务训练网络得到各自由形态像素的形状；摘要称设计了监测室内空间（8 个像素）等极简相机原型（arXiv 摘要）。 |
| **文内指标** | 公众号转述：用几个到十几个像素完成统计房间人数、检测门是否打开等任务；摘要称室内监测相机仅用 8 个像素。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — arXiv 摘要页未列代码或项目页链接；本次未做进一步检索，不能判定是否开源。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：用几个到十几个像素完成统计房间人数、检测门是否打开等任务；摘要称室内监测相机仅用 8 个像素。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [从视觉编程到 4D 重建：CV 三大会 2023–2026 Best Paper 阅读坐标](../overview/cv-best-papers-2023-2026-technology-map.md)；同 arXiv 不重复造页。

## 结论

**Minimalist Vision：传感器与网络联合设计，像素越少越要把硬件当网络层来训。**

1. 开源边界：**项目页未列 GitHub** — arXiv 摘要页未列代码或项目页链接；本次未做进一步检索，不能判定是否开源。
2. 核心机制：按任务训练网络得到各自由形态像素的形状；摘要称设计了监测室内空间（8 个像素）等极简相机原型（arXiv 摘要）。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [visual-representation-for-policy](../concepts/visual-representation-for-policy.md)
- [cv-best-papers-2023-2026-technology-map](../overview/cv-best-papers-2023-2026-technology-map.md)

## 参考来源

- [minimalist_vision_freeform_pixels_arxiv_2501_00142.md](../../sources/papers/minimalist_vision_freeform_pixels_arxiv_2501_00142.md)
- [wechat_shenlan_cv_best_papers_2023_2026_labs.md](../../sources/blogs/wechat_shenlan_cv_best_papers_2023_2026_labs.md)
- [arXiv:2501.00142](https://arxiv.org/abs/2501.00142)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2501.00142)
