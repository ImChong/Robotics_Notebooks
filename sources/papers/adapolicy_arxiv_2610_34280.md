# AdaPolicy（arXiv:2610.34280）

> 来源归档（paper）

- **标题：** AdaPolicy: Action-Driven Anticipatory Robot Policies via Next-Chunk Prediction
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.34280>
- **PDF：** <https://arxiv.org/pdf/2610.34280>
- **项目页：** <https://teee111.github.io/AdaPolicy/>
- **入库日期：** 2026-10-28
- **线索来源：** [人类视频怎样变成机器人能用的动作经验？](../blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)（具身智能小站，2026-10-28）
- **一句话说明：** 当前动作头同时输出动作与演化状态，训练时再由展开动作头预测下一段，使动作损失塑造未来感知。

## 开源状态

- **部分开源**（步骤 2.5 核查，2026-10-28）：公众号未给代码链接；项目页链接 teee111/ADWM-Code，git ls-remote 失败（仓库不存在或私有），不能当作已开源。

## 核心摘录（公众号转述，未核验原文表格）

- 预测下一段动作，获得隐式前瞻。无需编码未来观测，执行不增加展开分支，覆盖仿真与精细真机任务。
- 当前动作头同时输出动作与演化状态，训练时再由展开动作头预测下一段，使动作损失塑造未来感知。
- 指标：公众号转述：无需编码未来观测，执行不增加展开分支，覆盖仿真与精细真机任务。

## 对 wiki 的映射

- [paper-adapolicy](../../wiki/entities/paper-adapolicy.md)
- [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-28-59-papers-technology-map.md)
