# HandsOn 示范数据集来源

> 来源归档

- **类型：** dataset（LeRobot v2.1）
- **项目：** https://llltttwww.github.io/HandsOn-website/
- **论文：** https://arxiv.org/abs/2610.26765
- **数据集：** https://huggingface.co/datasets/Cyan666/HandsOn-Datasets
- **代码与读取器：** https://github.com/TsinghuaC3I/HandsOn
- **许可证：** MIT（数据卡；第三方资产另按各自许可）
- **入库日期：** 2026-10-25

## 已核实信息

- 数据卡列出 **18,000 条 episode**、**6,053,555 帧**、30 Hz、约 56.1 小时，以及每条 episode 三路同步 RGB 视频。
- 固定拆分为 **15,178 条训练 episode / 2,822 条验证 episode**，按采集 session 隔离；训练归一化统计只使用训练集。
- 采集平台为 ROKAE AR5-5_08 机械臂与 Wuji Hand 1。每帧状态和动作各为 52 维：左右手各 26 维，右侧在前；动作是绝对 world-palm command，不应误读为关节速度。
- 数据包包含 Parquet 状态/动作、同步视频、任务和 episode 元数据、语言指令、归一化统计、读取器与校验清单；数据卡列出完整下载约 111 GB。
- 数据卡将任务口径分为 Core-50 与 Transfer-16，并特别说明公开示范语料的来源身份、配额组、活动 task/prompt pair 与任务词典条目数不是同一计数口径。
- 该数据为模拟采集语料。数据卡明确指出，动作预测误差不能替代基于模拟器与任务检查器的闭环成功评估，也不能据此证明真机安全。

## Robotics_Notebooks 映射

- 项目实体：[HandsOn](../../wiki/entities/paper-handson.md)
- 论文来源：[handson_arxiv_2610_26765](../papers/handson_arxiv_2610_26765.md)
- 仓库来源：[handson](../repos/handson.md)
