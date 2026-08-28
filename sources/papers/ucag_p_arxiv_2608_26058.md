---
title: "UCAG-P: Unified Cross-embodiment Action Geometry for Policy Learning"
authors:
  - "Xu, Yifan"
  - "Li, Yiming"
  - "Zhan, Xinyu"
  - "others"
year: 2026
venue: "arXiv"
url: "https://arxiv.org/abs/2608.26058"
pdf_url: "https://arxiv.org/pdf/2608.26058"
github: "https://github.com/Public-BOTs/UCAG-P"
project_page: "https://public-bots.github.io/UCAG-P/"
status: "待发布"
tags:
  - cross-embodiment
  - action-geometry
  - vla
  - camera-frame
  - qwen3-vl
---

# UCAG-P: Unified Cross-embodiment Action Geometry for Policy Learning

- **论文**：UCAG-P: Unified Cross-embodiment Action Geometry for Policy Learning
- **作者**：Yifan Xu、Yiming Li、Xinyu Zhan 等（小米具身智能 × 澳门大学）
- **年份**：2026
- **链接**：https://arxiv.org/abs/2608.26058
- **PDF**：https://arxiv.org/pdf/2608.26058
- **GitHub**：https://github.com/Public-BOTs/UCAG-P
- **项目页**：https://public-bots.github.io/UCAG-P/
- **一句话**：相机系双锚点运动共享动作空间，几何条件翻译器映到本体命令；单 checkpoint 跨仿真基准零微调。
- **开源状态**：待发布（仓为论文图与项目页；code coming soon）
- **核心内容**：
  - 锚点：p0 腕/末端、p1 抓取中心，相机系。
  - 骨干：Qwen3-VL-4B。
  - 数据：6374h（机器人 + 仿真 + 人手）。
  - 单 ckpt 无榜微调：LIBERO 98.3%、RoboTwin Easy/Hard 88.7/89.2、LIBERO-Plus 82.0、RoboCasa GR-1 62.0。
  - Piper 真机：面包/抽屉/碗 60/90/75%。
- **整理后去向**：
  - [wiki/entities/paper-ucag-p.md](../../wiki/entities/paper-ucag-p.md)
  - [wiki/queries/cross-embodiment-transfer-strategy.md](../../wiki/queries/cross-embodiment-transfer-strategy.md)
  - [wiki/entities/qwen-robot-manip.md](../../wiki/entities/qwen-robot-manip.md)
