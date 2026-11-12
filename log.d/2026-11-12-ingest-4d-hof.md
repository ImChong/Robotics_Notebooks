## [2026-11-12] ingest | 4D-HOF 与三篇基础文献

- 意图：补足 4D 手物体交互中，视觉初始化、条件流匹配、物体 6D 位姿、手部 MANO 之间的技术联系。
- 主要结果：新增 4D-HOF 主知识实体（论文/官网/实现统一身份）与双语阅读、Mermaid 流程图和真实源码推理时序；归档 Albergo et al.、WiLoR、FoundationPose 三篇一手文献。
- 开源边界：TAMU-Visual-AI/4D-HOF 当前公开 HO3D 的推理/评估，不等于完整训练与原始视频初始化。
- 关联：复用已有 WiLoR 方法页，新增与 4D-HOF 的互链。
- 通读边界：已完成 4D-HOF v1 全文；本次补读 WiLoR CVPR 2025 正文和 supplementary，并结合官方 demo.py 补全来源、流程和实验边界。Stochastic Interpolants 与 FoundationPose 尚属初步核验，未标记已通读。
