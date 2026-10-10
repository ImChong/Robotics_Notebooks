# AVEVA PI：Swinging Door 压缩官方资料

> 来源归档：厂商官方文档与技术演示

- **类型：** site / documentation / presentation
- **官方参数文档：** <https://docs.aveva.com/bundle/pi-server-s-da-admin/page/1022865.html>
- **官方演示：** [Exception, Compression, and their Impacts on PI System Performance（AVEVA 用户大会 2023）](https://cdn.osisoft.com/osi/presentations/2023-AVEVA-San-Francisco/UC23NA-3PGK04-AVEVA_Bregenzer_Brent-Exception-Compression-and-their-Impacts-On-PI-System-Performance.pdf)
- **核查日期：** 2026-10-10
- **一句话说明：** AVEVA 将 swinging-door 算法用于 PI Data Archive 的历史值压缩，并说明 CompDev、CompDevPercent、CompMax、CompMin 等参数如何约束归档事件。

## 一手资料要点

官方演示把压缩解释为：在仍可由插值重建的范围内丢弃样本，以减少历史归档量；目标是滤除仪器/过程噪声，同时保留显著过程变化。演示展示斜率上下界逐步收窄的“门”形几何过程，并说明乱序（out-of-order）数据绕过该压缩路径。官方参数文档覆盖压缩偏差、最大/最小时间间隔的配置。

这些是 PI Server 产品行为的说明，不应被当作所有 swinging-door 实现的唯一规范；不同实现的边界判断、强制归档与错误处理可能不同。

## 对 wiki 的映射

- [Swinging Door Trending（摆动门趋势压缩）](../../wiki/concepts/swinging-door-trending-compression.md) — 用产品参数解释理论误差走廊如何落到历史库设置。
- [可观测性（Logs / Metrics / Tracing）](../../wiki/concepts/observability-logs-metrics-tracing.md) — 高速控制环数据和长期存档应分层处理。

## 参考来源

- [AVEVA PI Server System Management 官方文档：CompDev / CompMax / CompMin](https://docs.aveva.com/bundle/pi-server-s-da-admin/page/1022865.html)
- [AVEVA 官方演示 PDF（2023）](https://cdn.osisoft.com/osi/presentations/2023-AVEVA-San-Francisco/UC23NA-3PGK04-AVEVA_Bregenzer_Brent-Exception-Compression-and-their-Impacts-On-PI-System-Performance.pdf)
