## [2026-11-12] ingest | Cosmos 3 Interactive World Simulation 官方报告复核

- 复用已存在的 Cosmos Interactive World Simulation 唯一实体，不拆分四个 checkpoint 或重复建页。
- 重新核对 NVIDIA 2026-11-04 技术报告的时序因果注意力、四阶段后训练和原表关键指标：Bimanual 1× GB300 48.1 FPS；Sim-Auto 7 路 RGB + LiDAR 16× GB200 16.7 FPS。
- 新增 NVIDIA Cosmos Lab 官方项目页一手来源归档，修正旧报告日期和“双向 / 流式”“实时 / 30 FPS”“视觉生成 / 动作策略”的混淆；保留现有项目 Mermaid 流程图与源码时序图。
- 权重与框架已公开，但完整端到端闭环、传感器适配、模型安全评估仍需开发者自己完成。随后补充通读报告正文 §1–9、附录 A–D 的可读取文本与图表文字转写，并新增 Isaac Lab 策略闭环实验的正反例；唯一 wiki 节点补充指向原文归档的 `original` 字段，按三项结构健康规则核算为 3/3。PDF 高分辨率图片未独立目视审计，GPU 实验未复现。
