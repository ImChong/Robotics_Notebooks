# DiscoDemo（arXiv:2610.32093）

> 来源归档（paper）

- **标题：** DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.32093>
- **PDF：** <https://arxiv.org/pdf/2610.32093>
- **项目页：** <https://davian-robotics.github.io/DiscoDemo/>
- **代码：** <https://github.com/DAVIAN-Robotics/DiscoDemo>
- **入库日期：** 2026-10-29
- **线索来源：** [机器人缺的，是更多示范，还是更多成功解法？｜附代码与项目入口](../blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)（具身智能小站，2026-10-29）
- **一句话说明：** 模仿学习缺的往往不是更多相似轨迹，而是多种可成功的解法。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-29）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 让强化学习生成多种成功示范。DiscoDemo把强化学习用于数据生成，以逆向课程和多样性目标扩展覆盖，再训练同一个视觉策略；四项仿真和真机任务提供了较清楚的对照。
- 模仿学习缺的往往不是更多相似轨迹，而是多种可成功的解法。
- 指标：公众号转述：新增机制是让成功门控的解法多样性参与生成器训练，再把收益落实到下游视觉模仿，而非仅优化RL控制器自身回报；过强多样性也会损害任务能力。

## 对 wiki 的映射

- [paper-discodemo](../../wiki/entities/paper-discodemo.md)
- [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-29-58-papers-technology-map.md)
