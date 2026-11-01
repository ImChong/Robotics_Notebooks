# 四年十篇顶会Best Paper，谁在持续霸榜？

> 来源归档（blog / 微信公众号）

- **标题：** 四年十篇顶会Best Paper，谁在持续霸榜？
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/CJki5-KhJ438RdAIhzNbCw
- **发表日期：** 2026-11-01
- **入库日期：** 2026-11-01
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_shenlan_cv_best_papers_2023_2026_labs.md`](../raw/wechat_shenlan_cv_best_papers_2023_2026_labs.md)
- **一句话说明：** 四年 Best Paper 的主线是计算机视觉从静态 2D 像素处理走向动态、符合物理规律的 3D / 4D 世界重构：视觉编程与可控生成之后，VGGT 把前馈几何做到统一，D4RT 再把它扩展到动态 4D；UniAD、Minimalist Vision、BrickGPT 分别代表端到端驾驶、极简传感与物理可行生成。 共 9 篇；**9/9 独立详情节点**（本 ingest **新建 7**、**复用 2**）。

## 9 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 视觉推理与可控生成 | VisProg | [2211.11559](https://arxiv.org/abs/2211.11559) | **已开源** | [paper-visprog](../../wiki/entities/paper-visprog.md) |
| 02 | 端到端自动驾驶 | UniAD | [2212.10156](https://arxiv.org/abs/2212.10156) | **已开源** | [paper-uniad](../../wiki/entities/paper-uniad.md)（**复用**） |
| 03 | 视觉推理与可控生成 | ControlNet | [2302.05543](https://arxiv.org/abs/2302.05543) | **已开源** | [paper-controlnet](../../wiki/entities/paper-controlnet.md) |
| 04 | 视觉推理与可控生成 | Generative Image Dynamics | [2309.07906](https://arxiv.org/abs/2309.07906) | **项目页未列 GitHub** | [paper-generative-image-dynamics](../../wiki/entities/paper-generative-image-dynamics.md) |
| 05 | 视觉推理与可控生成 | Rich Human Feedback | [2312.10240](https://arxiv.org/abs/2312.10240) | **已开源** | [paper-rich-human-feedback](../../wiki/entities/paper-rich-human-feedback.md) |
| 06 | 计算成像与极简视觉 | Minimalist Vision | [2501.00142](https://arxiv.org/abs/2501.00142) | **项目页未列 GitHub** | [paper-minimalist-vision-freeform-pixels](../../wiki/entities/paper-minimalist-vision-freeform-pixels.md) |
| 07 | 3D / 4D 几何重建 | VGGT | [2503.11651](https://arxiv.org/abs/2503.11651) | **已开源** | [paper-vggt-visual-geometry-grounded-transformer](../../wiki/entities/paper-vggt-visual-geometry-grounded-transformer.md) |
| 08 | 视觉推理与可控生成 | BrickGPT | [2505.05469](https://arxiv.org/abs/2505.05469) | **已开源** | [paper-brickgpt](../../wiki/entities/paper-brickgpt.md) |
| 09 | 3D / 4D 几何重建 | D4RT | [2512.08924](https://arxiv.org/abs/2512.08924) | **项目页未列 GitHub** | [paper-d4rt](../../wiki/entities/paper-d4rt.md)（**复用**） |

## 对 wiki 的映射

- **9/9 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[从视觉编程到 4D 重建：CV 三大会 2023–2026 Best Paper 阅读坐标](../../wiki/overview/cv-best-papers-2023-2026-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-11-01）
- [x] 9 篇独立节点（7 新建 / 2 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
