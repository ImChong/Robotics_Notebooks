## [2026-11-10] ingest | sources/repos/velxio.md — 核验 Velxio 开源嵌入式固件与 SPICE 协同仿真平台

- 意图：将真实固件编译、CPU 指令仿真、ngspice-WASM 数模联合求解的架构整理为独立知识节点。
- 一手资料：Velxio 官网与年付定价、官方 README / 架构文档 / 模拟器实现、作者在 Espressif Developer Portal 的技术文章。
- 事实校正：正确拼写 Velxio；官网 35 板/6 类与源码 README 40 板/5 CPU 家族统计不一致；Linux 不是 ISA。
- 开源结论：AGPL-3.0 核心开源，STM32 / 树莓派 Linux 等商业板卡后端及 AI Agent / 账号功能由私有 overlay 提供。
- 沉淀：新增 [Velxio](../wiki/entities/velxio.md)，并与 [Wokwi](../wiki/entities/wokwi.md) 双向互链。
