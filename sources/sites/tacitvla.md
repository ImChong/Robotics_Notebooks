# TacitVLA 官方项目页 / Project Site

> 来源归档（site），页面内容于 2026-11-11 核对。

- **标题：** TacitVLA: Reasoning Vision-Language-Action Models Without Reasoning Latency
- **类型：** site
- **链接：** <https://tacitvla.github.io/>
- **论文：** <https://arxiv.org/abs/2610.17540>
- **PDF：** <https://arxiv.org/pdf/2610.17540>
- **入库日期：** 2026-11-11
- **核对范围：** 通读项目页头部、Method、Results、Latency、Benchmarks、Rollouts、BibTeX、资源按钮，并与论文原文 §4–6 交叉核实。
- **通读：** 已完成
- **沉淀到 wiki：** [TacitVLA 论文实体](../../wiki/entities/paper-tacitvla.md)

## 开源状态（2026-11-11）

**待发布。** 项目页资源栏列出 arXiv 与 PDF，代码位置显示 **Code soon**；未列官方可运行仓库、权重或独立数据下载入口。不得把论文使用的公开评测基准视为本文训练代码/权重已开源。

## 官方页面核对摘要

- **核心差异：** TacitVLA 与显式推理 VLA 共享训练序列和监督，动作专家的 attention mask 不允许读取当前生成的推理文本。
- **非记忆任务：** LIBERO、PhysicalAI-AV 部署时跳过推理文本生成。
- **记忆任务：** RoboMME、RMBench 在 action chunk 执行期间生成 ledger 条目，供下一次调用读取。
- **关键结果：** LIBERO-Plus **72.9%**、RoboMME Counting **82.7%**、RMBench 四任务 **77.5%**、PhysicalAI-AV 闭环 **52.5%** scene score；最大 **3.4×** 动作就绪时延对照。

## 互链

- [原始论文通读归档](../papers/tacitvla_arxiv_2610_17540.md)
- [主知识节点](../../wiki/entities/paper-tacitvla.md)
