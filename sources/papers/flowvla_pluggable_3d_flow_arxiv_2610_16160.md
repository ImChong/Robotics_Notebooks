# FlowVLA（arXiv:2610.16160）

> 来源归档（paper）

- **标题：** FlowVLA: Enabling World Dynamics Understanding through Pluggable 3D Flow Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.16160>
- **PDF：** <https://arxiv.org/pdf/2610.16160>
- **项目页：** <https://mmai-laboratory.github.io/flowvla/>
- **代码：** <https://github.com/MMAI-Laboratory/flowvla>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人动作为什么总在换块时卡顿？｜附代码与项目入口](../blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)（具身智能小站，2026-10-14）
- **一句话说明：** 只在训练时加入点级与末端执行器级三维流监督，让二维 VLA 对齐场景运动与控制信号；推理移除辅助头。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-14）：公众号只列项目页；项目页链接的 GitHub 仓库经 ls-remote 可达。

## 核心摘录（公众号转述，未核验原文表格）

- 二维 VLA 缺少动态理解；把三维知识留在训练阶段可避免推理成本上升。
- 可插拔 3D flow 头；监督来自点流与 EE 流；推理与原 VLA 等价。
- 指标：以原文为准。

## 对 wiki 的映射

- [paper-flowvla-pluggable-3d-flow](../../wiki/entities/paper-flowvla-pluggable-3d-flow.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
