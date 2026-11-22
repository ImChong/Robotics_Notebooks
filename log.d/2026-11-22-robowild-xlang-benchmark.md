## [2026-11-22] ingest | RoboWild | 复杂仿真家务的 VLM-RUA 评测

- **意图：** 将 XLANG Lab 的 RoboWild（38 场景、4480 交互物体、341 任务–本体组合）补入具身评测路线，厘清“最强 13.28%”不是实机成功率。
- **结果：** 新增独立论文详情、官网/README/源码核查归档、Mermaid 项目流与源码时序，挂载评测路线 Stage 3 与四层评测枢纽。
- **开源：** RoboWild + RoboWild-Harness 仿真与评价代码公开，Hugging Face 资产公开；未执行 GPU 复现。
- **原文核对：** 已通过官方 PDF 直链通读 2026-11-20 版 51 页论文（含附录 A–G）；确认 239 个等权 setup / 341 个本体评测实例、G1 在本文 ScaleBFM + Harness 方案下五模型均无成功、深度与 Coding Agent 消融结果；论文节点更新到 complete。
- **关键页：** [RoboWild](../wiki/entities/paper-robowild.md)
