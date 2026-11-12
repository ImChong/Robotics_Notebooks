## [2026-11-12] ingest | Cosmos 3 Interactive World Simulation 官方报告复核

- 复用已存在的 Cosmos Interactive World Simulation 唯一实体，不拆分四个 checkpoint 或重复建页。
- 重新核对 NVIDIA 2026-11-04 技术报告的时序因果注意力、四阶段后训练和原表关键指标：Bimanual 1× GB300 48.1 FPS；Sim-Auto 7 路 RGB + LiDAR 16× GB200 16.7 FPS。
- 新增 NVIDIA Cosmos Lab 官方项目页一手来源归档，修正旧报告日期和“双向 / 流式”“实时 / 30 FPS”“视觉生成 / 动作策略”的混淆；保留现有项目 Mermaid 流程图与源码时序图。
- 权重与框架已公开，但完整端到端闭环、传感器适配、模型安全评估仍需开发者自己完成。报告全文及附录仍未逐页完成通读，PR 应保持草稿直至该项完成。
