# Sim-to-Real: Learning Agile Locomotion For Quadruped Robots · 方法页一手资料复核

- **原文：** <https://arxiv.org/html/1804.10332v2>
- **通读：** 已完成
- **核对范围：** 通读 v2 §I–VII；核对 Minitaur 与 Bullet 平台、§V 执行器/时延建模、动态参数随机化、外扰及 §VI 迁移对照；参考文献仅浏览。
- **入库日期：** 2026-11-26

先辨识物理参数并建模电机与反馈延时；再加入动态参数分布和外部扰动，配合紧凑观测学习鲁棒控制。在 Minitaur 真机验证 trot 与 gallop；对照表明只在误差较大的原仿真器加入扰动不足以迁移。

- [关联方法页](../../wiki/methods/sim2real-transfer-workflow.md)
