# PartyMimic — 官方项目主页

> 官方项目技术资料归档（不是转载全文）

- **标题：** PartyMimic: General Humanoid Motion Tracking in Minutes on One GPU（PartyMimic：单卡分钟级通用人形全身运动跟踪）
- **类型：** site / 官方项目主页与完整实验、配方、SuperMimic 消融说明
- **作者 / 团队：** RoboParty Lab
- **项目：** https://roboparty.github.io/PartyMimic/
- **代码：** https://github.com/Roboparty/PartyMimic
- **日期：** 2026-11-25（资料复核；旧快照为 2026-11-11）
- **核对范围：** 官方项目页全文（Overview、Results、Training data、MLP or attention、Sparse keypoints、Sim2sim、LAFAN speed、数据选择与采样、Terrain、Recipe、SuperMimic 消融及长训、命令延迟、实机与 Agent 演示）；逐项核对评测表、实验条件、消融与局限。
- **通读：** 已完成
- **开源结论：** 已开源；仓库含训练、数据处理/重定向、评测、ONNX 导出代码与 G1/RP1 两份默认策略 checkpoint；原始数据需遵守各自许可。
- **沉淀到 wiki：** [PartyMimic](../../wiki/entities/partymimic.md)
- **源码归档：** [PartyMimic GitHub](../repos/partymimic.md)

## 项目页提供的一手证据

- 训练库扩充为 10,745 条（约 41 小时），旧库加 14 条 XtremeMotion。项目方的 G1/RP1 MLP 主表分别用 28/29 分钟，五组平均 38.4/39.3 mm；attention 均约 43 分钟，平均 37.4/38.7 mm。主表与随仓 seed-0 权重不是同一统计。
- 测试五组中，前四组来自 MotionDecode，Extreme 由 KungfuAthlete 源视频整段留出。误差在相对航向对齐、已跟踪区间上计算，不能替代完成率；SONIC/Mimic Lite 的现成权重和同卡重训基线分开。
- RP1 的 `sparse5` MLP 五组均值 42.3 mm；名称指五个关键身体部位外加骨盆。LAFAN-40 速度实验在训练同集，不等于 41 小时泛化训练。
- 项目页新增地形、训练尖峰保护、腕部跟踪与部署时延的独立实验。G1 地形实验全库和地形子集的 3,000-iter 成功率分别为 99.05% 和 89.1%；这是过滤后的训练库回放。
- 部署默认加 5–20 ms 命令滞后，去掉排行榜配方的 clip allocation 和 `--obs_err kp`；RP1 慢主机 Mani 仿真中，脚滑 3.75 → 1.42 → 1.03 cm/s 分别对应主表配方、单加滞后、完整部署默认。
- 官方另附 G1/RP1 真机示例，以及 INTACT、VLM 的 RP1 仿真展示。最新低箱抓取画面是仿真中的接触抓取，作者说明没有焊接约束；不是新增实机成功率，不把展示层当作仓库训练或部署入口。

## 核查范围边界

本归档依据公开项目页面及其结果表，未运行训练或真机验证；不将官方测量误写成第三方复现结果。

## 链接

- [项目主页](https://roboparty.github.io/PartyMimic/)
- [官方 GitHub](https://github.com/Roboparty/PartyMimic)
- [仓库 README](https://github.com/Roboparty/PartyMimic/blob/main/README.md)
- [wiki 实体](../../wiki/entities/partymimic.md)

