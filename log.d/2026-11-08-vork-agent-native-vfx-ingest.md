## [2026-11-08] ingest | Vork Agent 原生 3D/VFX 渲染引擎 — 核对官方项目、源码、许可证与性能边界

- 意图：为知识库补充 agent-native GPU 渲染工具，并与 Sand3D / 3D-as-Code 和机器人仿真描述建立对照。
- 一手证据：voyager.so/vork、官方技术解读、voyager-so/vork 的 README、CLI/Shot 源码、Benchmark RESULTS 与 LICENSE。
- 核心判断：Vork 是 source-available、非开放源码许可证；L4 百万鸟 96.942 秒不计 setup/warmup，Blender 耗时为外推数据，不能宣传为 24 倍等画质提速；不是机器人动力学仿真器。
- 输出：新增 wiki/entities/vork.md 及 sites/repos 溯源归档、原文 Mermaid 流程图和基于真实 CLI 的运行时序图，链接至 3D-as-Code 选型页。
