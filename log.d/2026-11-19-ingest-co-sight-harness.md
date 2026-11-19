## [2026-11-19] ingest | Co-Sight Harness Studio

- 意图：独立收录 ZTE-AICloud/Co-Sight 单仓中的 Co-Sight Harness 子项目，不把 Co-Sight 主平台与 Co-Knowledge 误归为 Harness。
- 开源结论：官方框架、Drive 编排插件、hooks 与多项配套组件源码公开；CLI/模型仍需外部依赖。源码与文档核对，未运行本地端到端验收。
- 关键页：[Co-Sight Harness](../wiki/entities/co-sight-harness.md)；[官方源码归档](../sources/repos/co-sight-harness.md)。
- 关键风险：框架默认全网卡监听且无登录鉴权；工作流门禁不代替机器人设备安全。
