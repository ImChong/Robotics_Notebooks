# Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car（arXiv:1704.07911）

- **类型：** paper / first-party original reading
- **标题：** Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car
- **作者：** Mariusz Bojarski、Philip Yeres、Anna Choromanska 等
- **原文：** https://arxiv.org/abs/1704.07911v1
- **HTML：** https://arxiv.org/html/1704.07911v1
- **PDF：** https://arxiv.org/pdf/1704.07911v1
- **核对范围：** 2026-11-09 阅读 arXiv v1 正文所有章节、图注，PDF可视算法和实验图；未独立复现原实验。
- **通读：** 已完成
- **一句话说明：** PilotNet 的平均卷积激活热图用于定位端到端转向网络的视觉依据。

## 原文方法和评测核对

每层 CNN feature maps 沿通道取平均，上层图反卷积上采样（权重=1、bias=0）后逐层乘低层平均图；生成归一化至[0,1]的可视热图。对热图选出的显著区域先膨胀约30像素，再单独平移显著区域、非显著区域和整图；显著区域移动对转向影响接近整图，非显著移动影响较小。图中显示道路边界、草地、停靠车辆等线索。

## 局限与复现边界

输入区域平移的敏感性检验不是端到端驾驶的因果安全证明；未复现实验。

## 对 wiki 的映射

- [论文实体](../../wiki/entities/paper-pilotnet-saliency.md)
- [11 篇阅读路线](../../wiki/overview/robot-learning-foundations-11-papers.md)
