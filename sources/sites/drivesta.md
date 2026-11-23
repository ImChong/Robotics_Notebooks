# DriveSTA 官方项目主页

- **标题：** Beyond a Single Human Demonstration: Learning from Scored Trajectory Alternatives for End-to-End Driving
- **网站：** https://rtwotwo.github.io/DriveSTA/
- **论文：** https://arxiv.org/abs/2611.27515（官方 2026-11-20 v1，21 页 PDF 全文已核对）
- **代码：** https://github.com/Rtwotwo/DriveSTA（已开源，可运行推理、训练与 NAVSIM v1 评分）
- **模型与数据入口：** https://huggingface.co/Rtwotwo/DriveSTA（官方引用，下载可用性未单独复核）
- **入库日期：** 2026-11-23
- **核对范围：** 官方项目页 Method、NAVTEST/HUGSIM 表、Results，与 arXiv v1 §3–5 和附录 A/B 交叉核对；项目视频未逐帧分析。
- **通读：** 已完成

## 主页证据与数字口径

- 项目将每个场景单条已记录轨迹转换为带评分的多种候选；项目页称 103.3k 真实场景挖掘获得 2.61M 高质量训练目标。
- Base/Scale 在 NAVTEST 的 Refinement PDMS 分别是 **95.14 / 95.55**；Scale Proposal 为 95.09。
- NAVTEST 四模型对照：DriveSTA-Scale Refinement 95.55、FPD-Drive-Pro **95.69**、ChainFlow-VLA 94.85、DrivoR 94.58；不可据此宣称 DriveSTA 在该表排名第一。
- HUGSIM 345 场景：Scale Refinement **RC 56.09**、**HD-Score 46.69**；HUGSIM 是闭环评测，不等同于 NAVSIM v1 的开环/非反应评分。
- 项目页的 41 段 NAVSIM 片段经过挑选（DriveSTA ≥95、三基线 <80），是**选择性定性示例**；全套 12.1k 场景评分才是量化报告口径。

## 原始资料分工

- [论文归档](../papers/drivesta_arxiv_2611_27515.md)：21 页 v1 全文方法、结果、附录精读
- [官方仓库归档](../repos/drivesta.md)：代码与复现入口
- [Wiki 方法归纳](../../wiki/entities/paper-drivesta.md)
