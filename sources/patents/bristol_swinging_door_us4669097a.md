# Data Compression for Display and Storage（Swinging Door 专利）

> 来源归档：原始专利资料

- **标题：** Data Compression for Display and Storage
- **类型：** patent / time-series / lossy-compression
- **专利号：** US4669097A
- **发明人：** Edgar H. Bristol
- **原始受让人：** Foxboro Co.
- **申请日 / 优先权日：** 1985-10-21
- **公开日：** 1987-05-26
- **Google Patents 镜像：** <https://patents.google.com/patent/US4669097A/en>
- **USPTO Patent Center：** <https://patentcenter.uspto.gov/applications/06789531>
- **一句话说明：** 专利把工业过程数据流压缩为趋势线走廊的端点：从首点建立误差上下界，随新样本收紧可行斜率范围；当新点使走廊无法继续容纳时，输出当前段端点并开始下一段。

## 原文结构摘录

专利摘要和说明书将 Swinging Door 描述为在线处理工业过程数据的方式：首个数据点及其误差偏移形成上下边界；新样本逐步修正边界；违反走廊条件的点结束当前区间；压缩输出用区间端点替代区间内的多数原始样本。专利另讨论保留实际极值/误差界的实现选项，因此不能把所有实现概括成“始终逐点保存极值”。

## 对 wiki 的映射

- [Swinging Door Trending（摆动门趋势压缩）](../../wiki/concepts/swinging-door-trending-compression.md) — 以在线误差走廊解释算法、参数和机器人遥测适用边界。
- [AVEVA PI 压缩资料](../sites/aveva-pi-swinging-door-compression.md) — 历史数据库中的参数与工程行为。
- [Bristol 1990 年会议论文书目](../papers/bristol_swinging_door_trending_1990.md) — 同一发明者后续的 SDT 论文出处。

## 法律与来源说明

Google Patents 显示的专利法律状态注明其状态信息并非法律结论；此归档只用于技术溯源，不作法律状态或实施自由判断。专利原始记录可从 USPTO Patent Center 按申请号 06789531 查阅。

## 参考来源

- [Google Patents：US4669097A](https://patents.google.com/patent/US4669097A/en)
- [USPTO Patent Center：Application 06789531](https://patentcenter.uspto.gov/applications/06789531)
