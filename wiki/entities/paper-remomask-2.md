---
type: entity
tags: ['paper', 'motion-generation', 'retrieval-augmented']
status: complete
updated: 2026-09-09
arxiv: "2609.08365"
venue: "arXiv 2026"
code: https://github.com/AIGeeksGroup/ReMoMask-2
related:
  - ../overview/visual-focus-data-efficiency-10-papers-technology-map.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/remomask_2_arxiv_2609_08365.md
  - ../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md
summary: "ReMoMask-2（arXiv:2609.08365）：在生成器预量化潜空间重建检索库，轻量投影器对齐文本查询；减少检索–生成表示鸿沟。"
---

# ReMoMask-2

**ReMoMask-2**（*Latent Retrieval-Augmented Masked Motion Generation*，[arXiv:2609.08365](https://arxiv.org/abs/2609.08365)，[项目/代码](https://github.com/AIGeeksGroup/ReMoMask-2)）— 详见 [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)。

## 一句话定义

文本到动作生成常把检索库和生成器潜空间分开——ReMoMask-2 在 masked motion 生成器的潜变量里直接做 retrieval-augmented 生成。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ReMoMask-2 | Retrieval-augmented Masked Motion Gen v2 | 本文框架 |
| T2M | Text-to-Motion | 文本条件动作生成 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| VAE | Variational Autoencoder | 动作潜空间量化 |

## 为什么重要

- 检索库建在生成器 pre-quantization latent，而非外部动作空间
- 轻量 text projector 对齐查询

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.08365](https://arxiv.org/abs/2609.08365) |
| **开源** | **已开源** |
| **项目/代码** | [https://github.com/AIGeeksGroup/ReMoMask-2](https://github.com/AIGeeksGroup/ReMoMask-2) |

## 核心原理

- 检索库建在生成器 pre-quantization latent，而非外部动作空间
- 轻量 text projector 对齐查询
- GitHub + 项目页已公开

## 源码运行时序图

官方仓 [https://github.com/AIGeeksGroup/ReMoMask-2](https://github.com/AIGeeksGroup/ReMoMask-2)（归档见 [remomask-2.md](../../sources/repos/remomask-2.md) 若已建）：

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Repo as 官方仓库
    Dev->>Repo: clone + 依赖安装
    Dev->>Repo: 按 README 训练/推理入口
    Repo-->>Dev: 指标/可视化输出
```

- **最短复现：** 以 README 训练/评测脚本为准。

## 实验与评测

- 指标与设置以原文 PDF / 项目页为准；上文 Highlights 来自公众号归纳 + 项目页摘要。
- 横向对照见 [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)。

## 结论

**ReMoMask-2 的可迁移主张已写入 Highlights；部署前以原文实验设定与开源边界为准。**

1. **真影响：** 见核心原理 bullets。
2. **次要代价：** 预印本/待开源项需独立复现验证。
3. **部署读法：** 已开源 — 先读 README 或项目页再接真机/智能体栈。

## 关联页面

- [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)
- [模仿学习](../methods/imitation-learning.md)

## 参考来源

- [remomask_2_arxiv_2609_08365.md](../../sources/papers/remomask_2_arxiv_2609_08365.md)
- [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- [arXiv:2609.08365](https://arxiv.org/abs/2609.08365)

## 推荐继续阅读

- [原文 PDF](https://arxiv.org/pdf/2609.08365)
- [项目/代码](https://github.com/AIGeeksGroup/ReMoMask-2)
