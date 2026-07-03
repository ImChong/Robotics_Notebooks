# NVIDIA Research — SimFoundry（GEAR Lab）

> 来源归档（ingest）

- **标题：** SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation
- **类型：** site（官方项目页）
- **发布方：** NVIDIA GEAR Lab；合作方含 Georgia Tech、Stanford、UT Austin、University of Toronto 等（页面作者列表）
- **原始链接：** <https://research.nvidia.com/labs/gear/simfoundry/>
- **配套论文：** arXiv:2606.28276 — 归档见 [sources/papers/simfoundry_arxiv_2606_28276.md](../papers/simfoundry_arxiv_2606_28276.md)
- **入库日期：** 2026-07-03
- **一句话说明：** 单段真机视频 → **交互式 sim-ready 数字孪生** + **digital cousins**；页面展示 **Real2Sim 策略评测**（相对真机 Pearson **0.911**）、**Sim2Real 零样本/协同训练**、**3DGS 背景 + 物体 mesh** 混合场景，以及与 **PolaRiS / SAM3D** 的并排对比。

## 摘录要点（与论文分工）

- **TL;DR：** 一段真实场景视频自动变成可交互仿真环境，用于规模化策略训练与评测。
- **重建场景 demo：** Desk / Kitchen / Dining / Toys / Outdoor 等可拖拽 3D 查看器；混合表示为 **3D Gaussian Splat 背景 + 带纹理物体 mesh**。
- **Real-to-Sim Eval：** 真机 vs SimFoundry 并排执行（Clear Table、Marker in Cup、Store Marker、Serve Fruits、Stack Dishware 等）；DreamZero 等策略示例；散点图显示 SimFoundry 预测贴近真机成功率并优于 **PolaRiS** baseline。
- **Sim-to-Real Policy Training：** Sim-only 训练策略在真机可达 **96–100%** 量级（页面任务卡片）；**unseen objects** 仍报告高成功率案例；**co-train**（sim + 少量 real）在 DROID 上进一步抬升 Stack Dishware / Store Marker 等。
- **Digital Twin & Cousin Generation：** 输入视频 → reconstructed twins → reconstructed cousins；**task cousins** 可提升 π₀.₅-DROID 微调：**28%→46%**（13 任务）、held-out **0%→29%**（7 任务）。
- **重建质量：** 点击真场景物体加载 **physics-ready mesh**；与 **SAM3D** 在 Easy/Medium/Hard 场景上的 Chamfer / Position Error 柱状对比；支持 **OCID** 等开源数据集与 **AI 生成图像** 输入。
- **背景重建：** 前景擦除 → background-only 视频 → 3DGS；多室内场景视频演示。
- **管线图（页面）：** 感知提取 → 2D-to-3D mesh → 物理参数标注与仿真 sanity check → 支持 object/scene/task 增强；**模块化**——基础模型升级可 **组件替换** 而无需重设计全流程。

## 论文 / 代码状态

- 论文：<https://arxiv.org/abs/2606.28276>（PDF：<https://arxiv.org/pdf/2606.28276v1>）
- 截至入库日，项目页**未给出**公开 GitHub 仓库链接；复现以论文附录与 NVIDIA GEAR 发布为准。

## 对 wiki 的映射

- [SimFoundry 论文实体](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) — 三阶段管线、digital cousins、评测与训练量化读点
- [Sim2Real](../../wiki/concepts/sim2real.md) — Real2Sim 资产与 sim2real 训练闭环
- [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md) — real-to-sim 相关性评测语境
- [NVIDIA GEAR Lab](../../wiki/entities/nvidia-gear-lab.md) — 研究组锚点
