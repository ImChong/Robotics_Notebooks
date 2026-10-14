# StreamingWAM（arXiv:2610.15585）

> 来源归档（paper）

- **标题：** StreamingWAM: Streaming World Action Modeling via Chunk-Wise Staircase Flow Matching
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.15585>
- **PDF：** <https://arxiv.org/pdf/2610.15585>
- **项目页：** <https://renshaojie233.github.io/StreamingWAM/>
- **代码：** <https://github.com/renshaojie233/StreamingWAM>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人动作为什么总在换块时卡顿？｜附代码与项目入口](../blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)（具身智能小站，2026-10-14）
- **一句话说明：** 把整块生成改为滚动动作缓冲 + 阶梯式分块噪声 + 历史条件，新观察到达后持续细化未来动作，同时压低响应延迟与跨块顿挫。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-14）：仓库可达；公众号称当前实现范围主要是 LIBERO-Plus，真机与其它基准以后续更新为准。

## 核心摘录（公众号转述，未核验原文表格）

- 世界动作模型部署时单次预测快还不够，相邻动作块接不上仍让机器人顿挫；首次暖启动与稳态更新的延迟须分开衡量。
- 滚动缓冲中的动作按时间距离分配不同噪声层级（staircase），前缀延续已承诺动作，历史记忆作为条件；每来一帧新观察即对缓冲做一次流匹配细化。
- 指标：作者报告覆盖 LIBERO、LIBERO-Plus、RoboTwin 与真机；公众号未核验表格，本页不引用具体数字。

## 对 wiki 的映射

- [paper-streamingwam](../../wiki/entities/paper-streamingwam.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
