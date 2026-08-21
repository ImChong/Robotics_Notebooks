# PartialBiGrasp（arXiv:2608.19188）

> 来源归档（ingest）

- **标题：** PartialBiGrasp: Inferring Hidden Local Geometry for Bimanual Grasping from Partial Views
- **类型：** paper / bimanual-grasping / 3d-perception / partial-point-cloud
- **arXiv abs：** <https://arxiv.org/abs/2608.19188>
- **PDF：** <https://arxiv.org/pdf/2608.19188>
- **项目页：** <https://partialbigrasp.github.io/>（归档见 [`sources/sites/partialbigrasp-github-io.md`](../sites/partialbigrasp-github-io.md)）
- **代码：** <https://github.com/partialbigrasp/codebase>（归档见 [`sources/repos/partialbigrasp-codebase.md`](../repos/partialbigrasp-codebase.md)）
- **机构：** IIIT Hyderabad（Robotics Research Center）
- **作者：** Ayush Kaura、Vignesh Vembar、Md Faizal Karim、Keshab Patra、K Madhava Krishna
- **发表 / 上传：** 2026-08-21（arXiv v1）
- **入库日期：** 2026-08-21
- **索引来源：** [具身智能小站 8 篇综述](../blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md)（<https://mp.weixin.qq.com/s/30hu9SRxbRNXJcGLnNwl_g>）

## 开源状态（步骤 2.5，2026-08-21）

- **部分开源：** 项目页链 [partialbigrasp/codebase](https://github.com/partialbigrasp/codebase)；README 标注 **in progress** — 已发布模型架构，**权重 / 训练代码 / 数据集 / 推理 notebook / 环境配置仍 TODO**。
- **结论：** 可审架构，**尚不可完整复现**。

## 摘录 1：问题与接口

- 大型/重型/几何复杂物体常只有少量可抓区域；真实 RGB-D 只能给出 **局部点云**。
- 目标不是重建完整 mesh，而是推断与 **接触决策** 有关的隐藏局部几何（厚度、边缘、夹爪间隙）。

## 摘录 2：方法

- **卷积占据网络** 隐式学习局部几何 → 可抓性、无碰撞接触区、物体厚度。
- 生成满足 **力闭合（force-closure）** 约束的双臂抓取对；**采样优化** 修正不完整几何歧义。
- 管线：单臂 grasp 生成 → FC pairing critic → **局部占据引导采样 refinement**。

## 摘录 3：评测

- 解析指标 + 大规模仿真（DG16M）+ 11 物体 RealSense D455 实机。
- 报告 DG16M 上 ~55% FC vs baseline ~22%。

**对 wiki 的映射：** [`wiki/entities/paper-partialbigrasp.md`](../../wiki/entities/paper-partialbigrasp.md)；交叉 [Manipulation](../../wiki/tasks/manipulation.md)、[Bimanual Manipulation](../../wiki/tasks/bimanual-manipulation.md)。

## 当前提炼状态

- [x] 项目页 + GitHub 开源核查（部分开源）
- [x] 升格 `wiki/entities/paper-partialbigrasp.md`
