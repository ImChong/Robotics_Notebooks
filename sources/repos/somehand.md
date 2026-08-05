# BotRunner64/somehand（跨形态灵巧手重定向）

- **标题**: somehand — universal dexterous-hand retargeting
- **类型**: repo / dexterous-hand / retargeting / teleoperation
- **作者**: BotRunner64
- **机构**: 西湖大学、上海创智学院（Teleopit 配套）
- **链接**: <https://github.com/BotRunner64/somehand>
- **项目页**: <https://botrunner64.github.io/teleopit-page/>
- **许可证**: Apache-2.0
- **默认分支**: `master`
- **收录日期**: 2026-08-05

## 一句话摘要

面向多厂商灵巧手的 **统一优化重定向**：MediaPipe / PICO Bridge / 录制流 → YAML 配置的机器人手；CLI + 可嵌入 Python API；Teleopit 论文中的 **归一化指方向 + 距离/拇指帧目标** 的工程实现入口。

## 为何值得保留

- Teleopit「跨手无需手型专用超参」主张的 **独立可复用仓**；支持 LinkerHand 系列、Unitree Dex5、Inspire、Sharpa Wave、BrainCo Revo2 等（见 README 表）。
- 输入通道覆盖 webcam、视频、PICO Bridge、hc_mocap UDP 与存档，便于脱离全身栈单独调试手。

## 对 Wiki 的映射

- 论文实体：[paper-teleopit.md](../../wiki/entities/paper-teleopit.md)
- 主仓：[teleopit.md](./teleopit.md)
- 概念交叉：[motion-retargeting](../../wiki/concepts/motion-retargeting.md)
