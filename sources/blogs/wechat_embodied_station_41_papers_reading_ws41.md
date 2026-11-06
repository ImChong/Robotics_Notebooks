# 具身智能小站 10-14 期 41 篇公众号全文通读

> 来源归档（原文通读 · 支撑 `wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md`）

- **标题：** 机器人动作为什么总在换块时卡顿？｜附代码与项目入口
- **作者：** 具身智能小站（微信公众号）
- **类型：** blog
- **原文：** <https://mp.weixin.qq.com/s/pxBq6lFyzH05LGiBkkR9mQ>（2026-10-14；本库抓取全文落盘于 [`sources/raw/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md`](../raw/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)）
- **一句话说明：** 41 篇按「1 深读 + 2 跟进 + 38 扫读」分档，每篇一个判断；开篇点明动作块换块处的衔接与持续细化问题。
- **核对范围：** 通读抓取的公众号全文 41 篇条目（含阅读路线、StreamingWAM 深读、FreeBar 与 RoboCombine 跟进、04–41 号 38 篇扫读、各篇论文 / 代码 / 项目页链接），未逐篇打开被评论的 41 篇论文原文；文中编辑部声明「依据可读取的官方摘要和作者仓库整理，全文实验与排序仍待复核」，因此其中数字与结论属转述。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **阅读优先级：** 1 篇深读（StreamingWAM）、2 篇跟进（FreeBar、RoboCombine）、38 篇扫读；按方向覆盖 VLA 与动作生成、世界模型与记忆、导航及运动控制、多机器人协作与评测。
- **StreamingWAM（深读）：** 单次预测快不够，相邻动作块接不上仍顿挫；滚动缓冲 + 阶梯噪声 + 历史记忆；复现时区分首次暖启动与稳态更新延迟；评估应同时看跨块连续性、任务成功与重规划延迟；公开仓库实现范围主要是 LIBERO-Plus。
- **FreeBar（跟进）：** 异步执行下新动作块要接上正在执行的旧块；已承诺前缀严格约束、近未来柔性衔接、远未来保留自由度。
- **RoboCombine（跟进）：** 单技能成功不等于串联成功；把失败定位到子任务交接状态，区分理想触发与预测触发，提出 PRTI 扩展原子起始状态。
- **扫读（04–41）：** 每篇一句判断并附论文 / 代码 / 项目页链接，例如 FlowVLA 训练期 3D 流监督、推理移除辅助头；Δ-Flow 以近期动作历史替代高斯起点；AnyRank 序贯检验排序策略；ReCAP 只保留初始锚点与近期完整动作块；多篇文章明确提示结论只适用于仿真或特定设定。

## 对 wiki 的映射

- [`wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md`](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
