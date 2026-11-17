# MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations

- **类型：** paper / 原文核读
- **论文：** https://arxiv.org/html/2310.17596
- **发表：** CoRL 2023；https://proceedings.mlr.press/v229/mandlekar23a.html
- **项目：** https://mimicgen.github.io/
- **官方代码：** https://github.com/NVlabs/mimicgen
- **核读日期：** 2026-11-17
- **核读范围：** 原论文方法（物体坐标系下的子任务片段复用与端点变换）、任务/数据实验、消融、泛化和失效边界；核对官方代码文档的 TaskSpec、DatagenInfo、DataGenerator、SelectionStrategy 和 Waypoint 调用职责。未运行 GPU/robosuite 数据生成。
- **通读：** 已完成

## 方法与实验边界

原文从少量人工示范中按交互物体切分子任务，在新物体姿态下变换演示末端轨迹，并以仿真控制器与成功检查拼成新轨迹，目标是自动扩展行为克隆的数据覆盖。论文研究报告约 200 条源示范扩展 50K+ 示范，覆盖 18 个任务；[官方公开仓](https://github.com/NVlabs/mimicgen) README 所列下载数据为 **48K+、12 任务**。两套口径分别是论文实验与公开数据集，不可混用。生成质量依赖任务划分、物体位姿、可重放控制及成功判定；重度接触与不可行几何会失败。

## 官方运行入口

[代码概览](https://mimicgen.github.io/docs/modules/overview.html) 确认 `mimicgen/scripts/generate_dataset.py` 为数据生成入口，读取任务模板配置与 `TaskSpec`；仿真 `Environment Interface` 产出 `DatagenInfo`；`DataGenerator` 利用 `SelectionStrategy` 选择演示片段，由 `Waypoint` 表示并执行变换后的末端目标。仅完成静态资料核对，没有声称完整训练复现。
