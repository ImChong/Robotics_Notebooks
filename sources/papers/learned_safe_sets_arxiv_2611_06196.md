# Learned Safe Sets（arXiv:2611.06196）

> 来源归档（paper）

- **标题：** Beyond Terminal Constraints: Reliable Robot Control with Learned Safe Sets
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06196>
- **PDF：** <https://arxiv.org/pdf/2611.06196>
- **代码：** <https://github.com/idra-lab/safe-mpc>
- **入库日期：** 2026-11-06
- **线索来源：** [只在虚拟房间练习，人形机器人走进陌生房间能做什么？](../blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)（具身智能小站，2026-11-06）
- **一句话说明：** 规划器在预测轨迹的中间节点检查能否回到安全集合，求解遇阻时可切换到备用动作。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-06）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 让机械臂提前给自己留一条安全退路。作者也在六轴 Z1 上测试了参考轨迹跟踪，安全结论依赖论文对集合的假设。
- 规划器在预测轨迹的中间节点检查能否回到安全集合，求解遇阻时可切换到备用动作。
- 指标：公众号转述：作者也在六轴 Z1 上测试了参考轨迹跟踪，安全结论依赖论文对集合的假设。

## 对 wiki 的映射

- [paper-learned-safe-sets](../../wiki/entities/paper-learned-safe-sets.md)
- [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)
