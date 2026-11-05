# Safe-Ballbot（arXiv:2611.03307）

> 来源归档（paper）

- **标题：** Safe-Ballbot: Tilt-Consistent MPC with Control Barrier Functions for Ballbot Navigation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.03307>
- **PDF：** <https://arxiv.org/pdf/2611.03307>
- **项目页：** <https://safe-ballbot.github.io/>
- **入库日期：** 2026-11-04
- **线索来源：** [机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口](../blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)（具身智能小站，2026-11-04）
- **一句话说明：** 用允许倾角推导规划加速度上界，连接上层MPC和下层平衡控制；结合控制屏障与预测误差膨胀避障，目前证据来自含轮接触和力矩饱和的仿真。

## 开源状态

- **部分开源**（步骤 2.5 核查，2026-11-04）：公众号未给代码链接；项目页链接 asRoBallet/asRoBallet_mujoco 经 git ls-remote 可达，是否即论文完整代码未核对。

## 核心摘录（公众号转述，未核验原文表格）

- 球形平衡机器人的避障不能忘了倾角。
- 用允许倾角推导规划加速度上界，连接上层MPC和下层平衡控制；结合控制屏障与预测误差膨胀避障，目前证据来自含轮接触和力矩饱和的仿真。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-safe-ballbot](../../wiki/entities/paper-safe-ballbot.md)
- [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md)
