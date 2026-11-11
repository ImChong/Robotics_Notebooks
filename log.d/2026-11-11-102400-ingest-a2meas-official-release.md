## [2026-11-11] ingest | A²-MEAS 论文已有深读节点：补核官方开源、实现时序和复现边界

- 目的：对照 arXiv:2611.09124v1、官方项目页和 `AutoLab-SAI-SJTU/A2MEAS` `evaluation` 分支，避免重复创建已有主节点。
- 开源结论：已公开 BEHAVIOR / RoboTwin 推理和评测实现、权重与一个 Adapter；完整纠错配置、部分任务特化策略及完整训练基础设施未在当前发布版开放。
- 关键区别：论文完整 BEHAVIOR Q-score 0.6411 / SR 45.40%；官方 README 发布版复现表 0.5287 / 32.40%，本次未执行 GPU benchmark。
- 产物：补充项目主页与源码两份来源归档，更新论文归档及唯一主节点，新增基于实际源码入口的 Mermaid 运行时序图。
