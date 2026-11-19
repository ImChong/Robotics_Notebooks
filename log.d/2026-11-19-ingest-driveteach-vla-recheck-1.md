## [2026-11-19] ingest | 复核 DriveTeach-VLA（ECCV 2026 / arXiv:2607.01658）论文全文及官方代码

- 查重后复用 `wiki/entities/paper-driveteach-vla.md`，不新增同一论文的重复实体。
- 通读论文 §1–5 与 Appendix 0.A–0.C，修正 NAVSIM navtest PDMS 90.4（单次）/ N=12 selector 92.7（不可直接混比）、nuScenes L2 0.30 m 与基准差别、消融、延迟和检测器噪声边界。
- 核对 `data_engine/main.py`、`data_engine/src/pipelines/runner.py`、`dvd/README.md` 及训练 YAML，按真实调用链完善 Mermaid 数据引擎运行时序，明确论文 `2×4` 与当前 DVD YAML `16×28` 的差异；GRPO 仍依赖 Curious-VLA。
- 补全 `sources/papers/` 归档结构化「核对范围 / 通读」及实体 original / project_id，健康度以 CI 同一脚本检测为准。
