# PlanContext 作者项目页（https://plancontext.github.io/）

> 一手项目站点归档，与 arXiv:2611.03067 属于同一项目，不创建重复实体。

- **正式标题：** PlanContext: Benchmarking In-Context Driving Decision-Making
- **项目主页：** <https://plancontext.github.io/>
- **论文：** <https://arxiv.org/abs/2611.03067>
- **论文 HTML：** <https://arxiv.org/html/2611.03067v1>
- **归档日期：** 2026-11-10
- **核对范围：** 核对官网页首作者/机构、Paper / Code / Datasets 资源区域、六类场景说明、23 模型总表、指标释义、主要发现、统计和 BibTeX；与 arXiv v1 §3–4 比对。
- **通读：** 已完成
- **对应实体：** [PlanContext](../../wiki/entities/paper-plancontext.md)

## 开放状态（2026-11-10）

官网明确显示 **“Code Coming soon”** 与 **“Datasets Coming soon”**，并非可点击的官方源码库或数据下载入口。因此为**计划开放、当前尚未发布**；未核实到 PlanContext 作者的训练/评测可运行代码及可下载构造后的数据集。站内提供模型表格 CSV 下载属于**结果展示材料**，不能等同数据集/完整实现已开源。

论文的底层 NAVSIM 基准有[独立公开仓库](https://github.com/autonomousvision/navsim)，但**NAVSIM 不等于 PlanContext 评测器源码**，不得将其冒充 PlanContext 复现入口。

## 官网核实内容

- 输入同一场景的观测与历史，只变更导航意图；每次预测未来 4 秒的 8 个自车姿态。
- 六类上下文：A、B1、B2、C、L、LC；对应跟随、等待、拒绝、保持等决策要求。
- 最终 9,561 scene–condition、5,911 物理场景、23 个 VA/VLA/WA 模型。
- 站点表 2 数据与论文一致：OneVL 的 NAS 22.46%、NCS 39.30%；mPDMS–NCS 相关系数 −0.542。
- **源码运行时序图：不适用。** 尚无可逐行阅读的 PlanContext 官方训练或评测入口；论文层面的评测流程图绘于 wiki 主节点。
