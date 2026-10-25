# Real-Time EXPO-FT（arXiv:2609.18207）

> 来源归档（paper）

- **标题：** Reinforcement Learning for Real-Time Vision-Language-Action Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.18207>
- **PDF：** <https://arxiv.org/pdf/2609.18207>
- **项目页：** <https://pd-perry.github.io/real-time-expo-ft/>
- **机构：** 斯坦福大学（Stanford University）
- **作者：** Perry Dong, Kuo-Han Hung, Dorsa Sadigh, Chelsea Finn
- **入库日期：** 2026-09-17
- **最后复核：** 2026-10-26
- **代码：** <https://github.com/pd-perry/expo-ft>（已开源；与 EXPO-FT 共仓，Real-Time 使用 `real-time-expo-ft` 分支）
- **一句话说明：** 在 EXPO-FT 上解耦慢 VLA chunk 生成与快 edit policy，用 Q 值选 chunk，实现在线 RL 微调实时 VLA；Kinetix 10/10 环境最优，四项真机任务平均 42%→97%。

## 开源状态

- **已开源**（复核于 2026-10-26）：项目页现有 Code 链接指向 `pd-perry/expo-ft`；README 提供 Real-Time EXPO-FT 的 OpenPI 与 DROID 分支、训练入口及真机 client。

## 核心摘录

1. **问题：** 大 VLA 推理延迟使执行时观测 stale；RTC 类异步执行可保平滑，但模仿学习无法越出训练分布提升可靠性。
2. **方法：** **Real-Time EXPO-FT** — 慢 **base VLA** 提案 action chunk + 轻量 **edit policy** 按最新观测快速修正 + **Q-function** 在线选最优候选 chunk。
3. **Kinetix：** delayed policy 在 **10/10** 环境中为 delayed 与 non-delayed 方法里最佳。
4. **真机（4 任务，在线数据上限 10 min）：** robot object passing、ball balancing、table soccer kicking、dynamic object picking；平均 **42%→97%**，无需人工干预。
5. **与 SmoothRL / WAM-async 对照：** 同属「部署延迟 + 在线改进」族；本文强调 **RL fine-tune EXPO-FT** 与 **edit + Q 选择** 三件套。

**对 wiki 的映射**

- [paper-real-time-expo-ft](../../wiki/entities/paper-real-time-expo-ft.md)
