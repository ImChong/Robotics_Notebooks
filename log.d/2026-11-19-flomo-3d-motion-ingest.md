## [2026-11-19] ingest | sources/papers/flomo_arxiv_2611_05062.md — 精读 FloMo 原文并升级三维场景流 WAM 节点

- 意图：检验“预测 3D 运动是否比预测未来视频更有利于动作学习”，区分视频预训练与未来预测目标。
- 原文：核对 arXiv:2611.05062v2 的方法、实验、局限与附录；保留同论文已有唯一实体 `wiki/entities/paper-flomo.md`。
- 开源：官方 pairlab/flomo 已提供训练、预处理、推理入口和 MA/VA/VMA 权重；运行与真机集成尚未独立复现。
- 关键结果：同条件真机 MA 91%、VMA 51%、VA 40%；OOD MA 18/30、VA/VMA 各 2/30；RoboTwin Easy 五任务 MA 63.6%。
- 落点：补原文与代码核验、Mermaid 主流程和源码时序，挂入 WAM 纵深路线 Stage 5。
