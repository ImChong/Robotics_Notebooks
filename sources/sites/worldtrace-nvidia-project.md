# WorldTrace — NVIDIA Spatial Intelligence Lab 官方项目页

- **类型：** 研究项目原始页面
- **标题：** WorldTrace: Addressable Memory for Video World Models
- **机构：** NVIDIA（英伟达）、普林斯顿大学、多伦多大学、Vector Institute
- **项目页：** https://research.nvidia.com/labs/sil/projects/WorldTrace/
- **主论文：** https://arxiv.org/abs/2608.07408
- **论文归档：** [arXiv:2608.07408v2](../papers/worldtrace_addressable_memory_arxiv_2608_07408.md)
- **核查日期：** 2026-11-23
- **核对范围：** 项目页摘要、机制图文字、Slot indexing/Field/Landmark/Combined、LoopBench 表、资源与页面末尾链接、代码开放状态。
- **通读：** 已完成

## 一手资料提要

- 给缓存摘要槽分配处于训练距离区间的虚拟时间位置；Field 在 canonical key 空间平均，Landmark 冻结场景进入帧；完整版由 attention-guided 选择。
- 项目页解释 12 项 LoopBench：Matrix-Game-2 上 0.587→0.778 DINOv2，组合方案较滑窗相对 +32.6%，但显存与推理耗时额外增加。
- 官方站点提供演示视频、方法/实验摘要、论文 BibTeX 和前期 F2S Workshop 报告说明。**未找到指向 WorldTrace 实现的官方 GitHub、可运行脚本、模型权重或数据集下载入口**，不能把“无需重训”误写为“官方实现已开源”。
- 不要沿用旧 workshop 版本的 PAC / TempSSIM 为 NeurIPS v2 的主结果；当前 v2 以 DINOv2 与 Return LPIPS 为主要回忆指标。

## 对 wiki 的映射

- [WorldTrace 独立论文详情](../../wiki/entities/paper-worldtrace-addressable-memory-video-world-models.md)
