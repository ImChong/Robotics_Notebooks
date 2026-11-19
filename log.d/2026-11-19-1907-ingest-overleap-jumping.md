## [2026-11-19] ingest | Overleap 双自由度机械腿弹跳控制

- 意图：从比特Z《拆解 Overleap：机械狗腿的弹跳算法》与 Overleap 原始源码整理完整知识实体。
- 开源结论：Aaron de los Santos 的 Overleap 官方仓库公开 Arduino 库、ODrive 示例、装配与机械图。
- 关键判断：下蹲、收腿和抬腿使用 MCU 侧关节 PD；蹬地使用 200 ms sin² 期望足端力与 Jᵀ 力矩映射；FSR 只负责接触判断，并非足端力闭环。
- 修正细节：标明 200 ms 力曲线与代码中的 >400 ms 离地退出条件不匹配。
- 本次核心页：wiki/entities/overleap.md；关联 roadmap/depth-classical-control.md Stage 1。
- 验证边界：已做文档及官方源码静态核读，未进行实机复现。
