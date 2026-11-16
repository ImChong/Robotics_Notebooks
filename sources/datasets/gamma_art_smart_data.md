# TeleEmbodied/SMART-Data — 合成铰接操作数据集

- **发布方：** TeleEmbodied（TeleAI / Gamma Robotics 项目论文关联）
- **官方数据：** https://huggingface.co/datasets/TeleEmbodied/SMART-Data
- **格式：** LeRobot v3.0
- **许可：** MIT（以官方仓库实际文件为准）
- **对应论文：** https://arxiv.org/abs/2610.07652
- **入库日期：** 2026-11-11

## 数据集内容

官方报告的合成数据规模超过 100 万条操作轨迹、约 5 亿帧、超过 4,600 小时；包含 2,507 个铰接资产、1,122 个场景、23 种操作技能、44 类原子任务，数据采样使用 5 类机器人配置。主要覆盖物体铰链、滑轨和不同操作几何的仿真视觉—动作监督，不是实采真机数据。

## 使用边界

- 数据量为作者公布规模，本归档未逐文件重新统计。
- LeRobot v3.0 与共享 PRTS 示例中的 LeRobot v2.1 入口不应直接假定兼容，使用前核对版本/转换接口。
- 原始数据不包含完整 SMART-Sim 数据生成平台的可复现源码。

## 交叉引用

- [SMART 原始论文归档](../papers/gamma_art_smart_arxiv_2610_07652.md)
- [官方项目页归档](../sites/gamma_art_smart_project.md)
- [SMART 实体详情](../../wiki/entities/paper-gamma-art-smart.md)
