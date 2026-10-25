# Real-Time EXPO-FT 项目页

> 来源归档（site）

- **标题：** Real-Time EXPO-FT: Reinforcement Learning for Real-Time Vision-Language-Action Policies
- **类型：** site
- **链接：** <https://pd-perry.github.io/real-time-expo-ft/>
- **arXiv：** <https://arxiv.org/abs/2609.18207>
- **机构：** 斯坦福大学（Stanford University）
- **入库日期：** 2026-09-17
- **最后复核：** 2026-10-26
- **代码：** <https://github.com/pd-perry/expo-ft>（已开源；EXPO-FT 与 Real-Time EXPO-FT 共仓）
- **一句话说明：** EXPO-FT + 慢生成/快 edit + Q 选 chunk 的实时 VLA 在线 RL 框架。
- **沉淀到 wiki：** [`wiki/entities/paper-real-time-expo-ft.md`](../../wiki/entities/paper-real-time-expo-ft.md)

## 开源状态

- **已开源**（复核于 2026-10-26）：页内 Code 链指向 `pd-perry/expo-ft`；该仓库 README 标明包含两篇 EXPO-FT 论文的代码，并给出 Real-Time 的分支与运行入口。

## 项目页要点

- **三模块：** 大 VLA 慢提案 chunk → edit policy 按执行时刻观测修正 → Q 值选最优候选。
- **演示：** Policy rollouts 与 training 曲线；强调 **1× wall-clock** 真机视频。
- **评测：** 各任务 w/ RTC 与 w/o RTC 成功率对比表。
- **任务：** object passing、ball balancing、table soccer、dynamic picking。
