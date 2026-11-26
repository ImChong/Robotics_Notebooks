# DriveSTA 官方项目主页

- **标题：** Beyond a Single Human Demonstration: Learning from Scored Trajectory Alternatives for End-to-End Driving
- **网站：** https://rtwotwo.github.io/DriveSTA/
- **论文：** https://arxiv.org/abs/2611.27515（官方 2026-11-20 v1，21 页 PDF 全文已核对）
- **代码：** https://github.com/Rtwotwo/DriveSTA（已开源，可运行推理、训练与 NAVSIM v1 评分）
- **模型与数据入口：** https://huggingface.co/Rtwotwo/DriveSTA（[已核对发布清单](../models/drivesta-release-assets.md)，未下载大文件）
- **入库日期：** 2026-11-23
- **核对范围：** 官方项目页 Method、NAVTEST/HUGSIM 表、Results，与 arXiv v1 §3–5 和附录 A/B 交叉核对；项目视频未逐帧分析。
- **通读：** 已完成

## 主页证据与数字口径

- 项目将每个场景单条已记录轨迹转换为带评分的多种候选；项目页称 103.3k 真实场景挖掘获得 2.61M 高质量训练目标。
- Base/Scale 在 NAVTEST 的 Refinement PDMS 分别是 **95.14 / 95.55**；Scale Proposal 为 95.09。
- NAVTEST 四模型对照：DriveSTA-Scale Refinement 95.55、FPD-Drive-Pro **95.69**、ChainFlow-VLA 94.85、DrivoR 94.58；不可据此宣称 DriveSTA 在该表排名第一。
- HUGSIM 345 场景：Scale Refinement **RC 56.09**、**HD-Score 46.69**；HUGSIM 是闭环评测，不等同于 NAVSIM v1 的开环/非反应评分。
- 项目页的 41 段 NAVSIM 片段经过挑选（DriveSTA ≥95、三基线 <80），是**选择性定性示例**；全套 12.1k 场景评分才是量化报告口径。


- **原文 Table 3 / Appendix B.7 补核：** 作者在 H20、batch1、64 候选、12 步 DDIM 的对齐前向测试中报告 ChainFlow-VLA 平均 1294.68 ms、DriveSTA-Scale S2 349.30 ms（约 −73%）；不含预处理、数据传输及仿真执行。论文基线 Table 1 与官网另组复现对照不可混为一张榜单。

## 11 月 24 日演示更新

- [HUGSIM 演示区](https://rtwotwo.github.io/DriveSTA/#hugsim) 有 51 段高分闭环片段；[渲染清单](https://github.com/Rtwotwo/DriveSTA/blob/f89bcf98233e417bb44e515302b712fec80d3c3d/docs/assets/data/hugsim-trajectory-render.json) 记录其叠加来自已记录的逐帧 8 点预测轨迹，按原相机视角投影，视野外点裁剪。更新的是可视化，不是重新评测，也不是 51 个新增评测场景。
- 按 `(dataset, scene, mode)` 对齐，51 条展示记录和 51 条渲染记录一一对应，且都在 [HUGSIM 345 条公开逐场结果](https://rtwotwo.github.io/DriveSTA/assets/data/hugsim-scores.csv) 中；基于该表重算平均 RC **56.09**、HD-Score **46.69**。
- 用 [NAVTEST 12,146 条公开配对结果](https://rtwotwo.github.io/DriveSTA/assets/data/navtest-scores.csv) 重算展示筛选规则，共 **41** 条符合阈值，其中 **13** 条三基线都为零。以上是对已发布记录做统计核对，不是复跑两个 benchmark。

## 原始资料分工

- [论文归档](../papers/drivesta_arxiv_2611_27515.md)：21 页 v1 全文方法、结果、附录精读
- [官方仓库归档](../repos/drivesta.md)：代码与复现入口
- [Wiki 方法归纳](../../wiki/entities/paper-drivesta.md)
