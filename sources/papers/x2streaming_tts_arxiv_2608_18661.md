# X2Streaming-TTS（arXiv:2608.18661）

> 来源归档（ingest）

- **标题：** X2Streaming-TTS: Causal Token-Level Text-to-Speech from Streaming Text with Speech-State Inheritance
- **类型：** paper / streaming-tts / causal-generation / human-robot-interaction
- **arXiv abs：** <https://arxiv.org/abs/2608.18661>
- **PDF：** <https://arxiv.org/pdf/2608.18661>
- **论文引用代码：** <https://github.com/X-Square-Robot/X2Streaming-TTS>（**404**，2026-08-21）
- **机构：** X Square Robot（平方机器人）
- **作者：** Rime Wen、Zehan Liu、Shawn Qin、Lights Shi、Roy Gan、Hao Wang、Qian Wang（通讯）
- **发表 / 上传：** 2026-08-21（arXiv v1）
- **入库日期：** 2026-08-21
- **索引来源：** [具身智能小站 8 篇综述](../blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md)（<https://mp.weixin.qq.com/s/30hu9SRxbRNXJcGLnNwl_g>）

## 开源状态（步骤 2.5，2026-08-21）

- **待发布：** 论文写 implementation publicly available； cited GitHub **404**；无项目页。
- **结论：** **宣称将开源 / 待发布**。

## 摘录 1：严格流式接口

- 许多「流式 TTS」仍等句级文本；本文做 **令牌级因果合成**，前缀不确定时仍持续发声。

## 摘录 2：因果承诺 + 状态继承

- **Causal commitment：** 不确定性感知缓冲 + 容量自适应 + 标点感知分段处理歧义（如 “3” vs “3rd”）。
- **Causal speech-state inheritance：** 跨段携带 Code2Wav 全状态与部分 Talker KV 历史。

## 摘录 3：延迟与质量

- 多数主客观指标优于伪流式；单请求首音频令牌中位时延 **15.8 ms**；128 并发 **260.8 ms**；质量接近离线基线。

**对 wiki 的映射：** [`wiki/entities/paper-x2streaming-tts.md`](../../wiki/entities/paper-x2streaming-tts.md)；交叉 [Teleoperation](../../wiki/tasks/teleoperation.md)、[VLA](../../wiki/methods/vla.md)。

## 当前提炼状态

- [x] 论文 cited 仓库 404 核查
- [x] 升格 `wiki/entities/paper-x2streaming-tts.md`
