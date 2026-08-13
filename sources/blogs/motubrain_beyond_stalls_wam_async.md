# 告别卡顿：世界动作模型实时部署（Motubrain 博客）

> 来源归档（ingest 配套博客）

- **标题：** 研究｜告别卡顿：世界动作模型实时部署的实战经验与总结
- **类型：** blog
- **URL：** <https://www.motubrain.com/zh/research/beyond-stalls-deploying-world-action-models/>
- **对应论文：** arXiv:2608.01880 — [`sources/papers/wam_realtime_async_arxiv_2608_01880.md`](../papers/wam_realtime_async_arxiv_2608_01880.md)
- **日期：** 2026-08-03
- **入库日期：** 2026-08-13
- **一句话说明：** 官方中文长文，与 arXiv 实证同构：六策略、硬件时间戳对齐、三任务真机视频与结论。

## 相对 PDF 的补充读点

- 明确 **硬件时间戳** 做观测–命令对齐；缺对齐则任何融合都难补。
- \(d_{\mathrm{est}}\) 取端到端延迟**中位数**，作者承认只是粗近似；在线精确延迟估计仍开放。
- async+blend 对齐 SmolVLA chunk fusion；simple 对齐 HoloBrain-0 SimpleRTC；infer 对齐 Black et al. RTC；train 对齐 Training-Time RTC。
- 真机视频按原速播放（传送带 / 插块 / 微波炉）。

## 对 wiki 的映射

- [`wiki/entities/paper-wam-realtime-async.md`](../../wiki/entities/paper-wam-realtime-async.md)
