# Contact-4：仿真接触技能示范

- **官方数据：** <https://huggingface.co/datasets/ntumars/Contact-4>
- **核对版本：** `37ed93d660428d47480461c8b6564cc97388f094`
- **许可证：** CC BY 4.0
- **核对范围：** 发布卡、接口及文件清单；核对六个目录各 50 个 Parquet 回合，未下载全量数据。
- **通读：** 已完成
- **论文：** [全文归档](../papers/contact_harness_arxiv_2611_29377.md)

六项任务共 300 条成功示范、93,195 帧、10 Hz、约 2.6 小时。Insert、Wipe、Write、Pivot 四个接触任务 200 条，Place 与 Stack 是 `reference/` 下的取放参考。种子从 100 起，与评测 0–9 分开；静止帧被移除，不能用演示时长作为真实节拍。

发布物采用 LeRobot v2.1，包括双 RGB 图像、13 维状态（末端位置与角度、夹爪、力、力矩）、7 维目标末端与夹爪动作，及每回合控制参数。姿态和力旋量使用机器人基座系。数据是仿真采集，不是 Rokae 或 Franka 真机数据；跨本体训练前需对照动作定义、符号和控制参数。

**对应 wiki：** [Contact Harness](../../wiki/entities/paper-contact-harness.md)
