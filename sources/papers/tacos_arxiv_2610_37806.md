# TACOS（arXiv:2610.37806）

> 来源归档（paper）

- **标题：** TACOS: Terrain-Aware COntact Supervision for Visual Footstep Planning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37806>
- **PDF：** <https://arxiv.org/pdf/2610.37806>
- **代码：** <https://github.com/norlab-ulaval/TACOS>
- **入库日期：** 2026-10-30
- **线索来源：** [机器人做错了，奖励为什么还在增加？｜附代码与项目入口](../blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)（具身智能小站，2026-10-30）
- **一句话说明：** 几何平整不代表地面可靠。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-30）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 先判断脚下会不会滑，再决定落脚点。TACOS 从状态估计因子图的接触残差构造不稳定性监督，再投影到此前图像，学习视觉落脚预测；真机森林路径展示了它对泥地、植被等非几何风险的处理。
- 几何平整不代表地面可靠。
- 指标：公众号转述：监督来源可比较：硬评分利用后验信息，软评分可因果获得；两者都由接触残差生成，而非凭直觉选择标签。

## 对 wiki 的映射

- [paper-tacos](../../wiki/entities/paper-tacos.md)
- [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-30-28-papers-technology-map.md)
