# EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models（arXiv:2605.25477）

> 来源归档（ingest · 多模空间公众号策展 + 2026-09-29 深化）

- **arXiv：** <https://arxiv.org/abs/2605.25477>
- **机构：** 斯坦福大学（Stanford）
- **评测：** 六项复杂单臂/操纵任务；**30/30** 成功率；平均 **19.1 min** 在线机器人交互
- **项目页：** <https://pd-perry.github.io/expo-ft>
- **代码：** <https://github.com/pd-perry/expo-ft>（**已开源**；与 Real-Time EXPO-FT 同仓）
- **备注：** CoRL 2026；基座 **π0.5** SFT；算法源自 [EXPO](./expo_arxiv_2507_07986.md)
- **入库日期：** 2026-09-29
- **一句话说明：** 预训练 VLA 上 EXPO 式 **chunk 采样 + edit + Q 选优 + 回灌** 在线 RL；真机高成功率、极少在线样本。

## 核心摘录

### 1) 系统 loop

- VLA Proposal 多个 action chunk → edit policy 修正 → **Q 选最优** → 执行；HIL 纠正进入 buffer；高回报轨迹 **IL 更新大 VLA**。

### 2) 代表性数字

- 插花：SFT **14/30** → EXPO-FT **30/30**（**14 min** 在线数据，公众号转述）。
- 全任务 **30/30**，平均在线 **19.1 min**（论文 abstract）。

## 公众号 / 博客

- [wechat_embodied_heart_expo_universal_post_training_2026-09-29.md](../blogs/wechat_embodied_heart_expo_universal_post_training_2026-09-29.md)
- [pd_perry_universal_post_training_robotics_2026-09.md](../blogs/pd_perry_universal_post_training_robotics_2026-09.md)
- [`wechat_duomo_vla_weekly_trends_2026-08-17_part1.md`](../blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md)

## 对 wiki 的映射

- [`wiki/entities/paper-expo-ft.md`](../../wiki/entities/paper-expo-ft.md)
- [`wiki/entities/paper-expo.md`](../../wiki/entities/paper-expo.md)
