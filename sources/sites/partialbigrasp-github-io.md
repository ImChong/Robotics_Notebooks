# partialbigrasp.github.io（PartialBiGrasp 项目页）

- **标题：** PartialBiGrasp — Inferring Hidden Local Geometry for Bimanual Grasping from Partial Views
- **类型：** site / project-page
- **URL：** <https://partialbigrasp.github.io/>
- **arXiv：** <https://arxiv.org/abs/2608.19188>
- **入库日期：** 2026-08-21
- **配套论文：** [PartialBiGrasp（arXiv:2608.19188）](../papers/partialbigrasp_arxiv_2608_19188.md)

## 一句话摘要

IIIT Hyderabad 提出的 **PartialBiGrasp** 官方站点：从 **单视角局部点云** 推断隐藏局部几何，生成力闭合 **双臂抓取对**；卷积占据网络 + FC pairing critic + 局部占据引导采样 refinement。

## 公开信息要点（截至 2026-08-21 核查）

- **机构：** IIIT Hyderabad Robotics Research Center。
- **方法：** 不重建完整 mesh；补全与接触相关的厚度、边缘、夹爪间隙。
- **评测：** DG16M 仿真 + 11 物体 RealSense D455 实机；~55% FC vs ~22% baseline。
- **代码（步骤 2.5）：** 链 [partialbigrasp/codebase](https://github.com/partialbigrasp/codebase)；README **in progress** — 架构已发，权重/训练/数据/推理 **TODO**。按 **部分开源** 处理。

## 关联资料

- 论文归档：[`sources/papers/partialbigrasp_arxiv_2608_19188.md`](../papers/partialbigrasp_arxiv_2608_19188.md)
- 代码归档：[`sources/repos/partialbigrasp-codebase.md`](../repos/partialbigrasp-codebase.md)
- 实体页：[`wiki/entities/paper-partialbigrasp.md`](../../wiki/entities/paper-partialbigrasp.md)
