# Open-HumanWAM（arXiv:2611.06817）

> 来源归档（paper）

- **标题：** Open-HumanWAM: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06817>
- **PDF：** <https://arxiv.org/pdf/2611.06817>
- **项目页：** <https://dagroup-pku.github.io/open-humanscale/>
- **代码：** <https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale>
- **入库日期：** 2026-11-06
- **线索来源：** [只在虚拟房间练习，人形机器人走进陌生房间能做什么？](../blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)（具身智能小站，2026-11-06）
- **一句话说明：** Open-HumanScale 从第一人称视频里提取手部动作。

## 开源状态

- **部分开源**（步骤 2.5 核查，2026-11-06）：公众号称 Open-HumanScale，arXiv 最新题名为 Open-HumanWAM；DAGroup-PKU/HumanNet 的 open-humanscale 分支经 git ls-remote 可达（Apache-2.0，含 humanwam_vanilla / humanwam_goal 目录）。README 新闻栏（2026-11-05）写代码与文档仍在整理，Open-HumanWAM-50K 语料标 Coming soon，模型权重集合上传中，因此只算部分开源。

## 核心摘录（公众号转述，未核验原文表格）

- 人手操作视频能帮机器人适应新桌面吗。作者固定机器人预训练数据和每项任务的示范数，检验加入人类视频后，策略能否适应新的灯光与桌面。
- Open-HumanScale 从第一人称视频里提取手部动作。
- 指标：公众号转述：每项用 50 条干净示范微调后，RoboTwin 随机场景成功率从 70.80% 升至 74.36%（50 项仿真任务，各测 100 次）。

## 对 wiki 的映射

- [paper-open-humanwam](../../wiki/entities/paper-open-humanwam.md)
- [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)
