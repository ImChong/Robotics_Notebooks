# How a robot remembers（VideoDB）

- **类型：** 官方技术文章 / 产品架构观点
- **作者：** Ashutosh Trivedi
- **发布日期：** 2026-10-27
- **原文：** <https://videodb.io/blog/how-a-robot-remembers>
- **核对范围：** 2026-11-01 通读文章正文、示例、FAQ 和参考文献列表；页面未标注版本号。
- **代码状态：** 官方 [Python SDK](https://github.com/video-db/videodb-python) 公开；文章中的找瓶子流程为示意代码，并未提供完整机器人执行应用。SDK 与服务端实现不是同一开放范围。已阅读 SDK 快照 `9d6e55fb6115d6b210d7609105dd2601115b98ad` 的 `collection.py`、`rtstream.py`、`shot.py` 相关调用。

## 内容摘要

文章提出模型外的视频记忆：摄像头流经分析器写入带时间及摄像头来源的事件索引，任务触发语义查询和过滤，回取相关画面供模型判断，再将结论或新的分析结果建立索引。索引查不到时可针对原视频重跑分析。实时数据更新与模型内部控制频率的短时缓存不同。

示例将“我的瓶子”先通过人物饮水记录关联到蓝瓶，再查找最近出现位置，回看片段确认细节；这是说明性示例，不能视作真机成功率报告。文章引用 MEM、MemER 和 ReMEmbR 等研究，不应将这些论文的结果归因于 VideoDB 机器人系统。

官方报告同区域视频流生成耗时中位数约 433–447 ms（不超过一分钟的窗口），数字来自其[性能文章](https://videodb.io/blog/frame-access-that-keeps-research-moving)，不包含模型理解与机器人执行的端到端延迟。

## 对 wiki 的映射

- [机器人如何记忆](../../wiki/entities/article-videodb-robot-memory.md)
