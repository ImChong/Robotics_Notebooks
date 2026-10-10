---
type: concept
tags: [time-series, data-compression, telemetry, observability, robot-data, signal-processing]
status: complete
updated: 2026-10-10
related:
  - ./observability-logs-metrics-tracing.md
  - ./embodied-data-cleaning.md
  - ./robot-data-supervision-signal-types.md
  - ./embodied-data-flywheel-minimal-closed-loop.md
  - ../formalizations/control-loop-latency-modeling.md
sources:
  - ../../sources/patents/bristol_swinging_door_us4669097a.md
  - ../../sources/papers/bristol_swinging_door_trending_1990.md
  - ../../sources/sites/aveva_pi_swinging_door_compression.md
  - ../../sources/repos/swingingdoor-emrumo-reference-implementation.md
summary: "Swinging Door Trending（摆动门趋势压缩，SDT）在线维护一对误差斜率边界，用分段线性端点替代容差内的冗余时序样本；适合长期趋势/遥测存档，不等于无损压缩，也不应未经验证地放进硬实时控制环。"
---

# Swinging Door Trending（摆动门趋势压缩）

## 一句话定义

**Swinging Door Trending（SDT）** 是一种在线、按单变量样本流工作的有损分段线性压缩算法：它维护锚点周围的上下斜率边界；只要新样本仍能落在容差走廊内，就暂不归档，走廊被新样本突破时再输出区间端点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDT | Swinging Door Trending | 摆动门趋势压缩的常见名称 |
| SDCA | Swinging Door Compression Algorithm | 代码实现中常用的算法名称 |
| PI | Plant Information | AVEVA PI 历史数据库产品系列名称 |
| CompDev | Compression Deviation | 误差走廊容差；PI 中按工程单位设置 |
| CompMax | Compression Maximum Time | 允许两条已归档记录之间经过的最长时间 |
| OOO | Out-of-Order | 时间戳乱序样本；PI 官方资料说明此类数据会绕过其压缩流程 |

## 为什么重要

机器人通常以高频采集关节状态、电机电流、接触力和任务指标，原样保留所有观测会快速增加长期存档成本。SDT 能在保留“整体趋势在给定容差内”的条件下，把平滑片段压缩成较少的代表点，适合作为**长期遥测归档**或离线趋势查看的一种选择。

但原始控制/安全数据与可视化趋势不是一回事：压缩后仍可在容差内重建，不代表每个尖峰、接触瞬态或相位关系都保留下来。

## 核心原理

对上一已归档点 \(A=(t_A,y_A)\)，设置绝对偏差容差 \(\epsilon\)。对时间戳为 \(t_i>t_A\) 的新点 \(P_i=(t_i,y_i)\)，其允许的上下斜率为：

\[
m_i^{upper}=\frac{y_i+\epsilon-y_A}{t_i-t_A},\qquad
m_i^{lower}=\frac{y_i-\epsilon-y_A}{t_i-t_A}.
\]

算法维护截至当前候选端点仍可行的斜率区间：

- 新上界只会收窄：\(m_{upper}\leftarrow\min(m_{upper},m_i^{upper})\)。
- 新下界只会收窄：\(m_{lower}\leftarrow\max(m_{lower},m_i^{lower})\)。
- 若 \(m_{lower}\le m_{upper}\)，当前点仍可由某条端点连线在偏差 \(\epsilon\) 内表示；将其作为新快照候选继续处理。
- 若区间被突破，前一个仍在走廊内的快照成为分段端点，开始下一段并重置斜率边界。

因此采样点到重建折线的偏差受所选容差约束。它约束的是该实现的误差定义；不能据此推断下游控制、频谱、阈值事件或所有插值方式也保持等价。

\`\`\`mermaid
flowchart TB
    A["上一归档锚点 A"] --> P["读取新样本 P"]
    P --> S["计算允许的上下斜率"]
    S --> B{"上下界仍相交？"}
    B -->|"是"| U["收紧斜率走廊并更新快照"]
    U --> P
    B -->|"否"| E["输出上一快照作为分段端点"]
    E --> R["以端点重置锚点和斜率边界"]
    R --> P
\`\`\`

### 一个直观例子

假设容差 \(\epsilon=0.1\)：

| 时间 | 原始值 | 当前判断 |
|------|-------:|----------|
| \(t_0\) | 1.00 | 作为起始锚点归档 |
| \(t_1\) | 1.04 | 如果可被锚点到后续端点的直线在容差内表示，暂存为候选 |
| \(t_2\) | 1.08 | 若上下斜率范围仍有交集，继续延长同一趋势段 |
| \(t_3\) | 1.45 | 若使走廊失效，输出此前快照，开始新段 |

此例展示判断思路，不是严格的数值测试：是否保留 \(t_1,t_2\) 取决于时间间隔和后续样本共同定义的斜率区间，而不是只比较相邻点的幅值。

## 工程实践

1. **先分清数据用途。** 控制回路、故障取证、安全与动态辨识保留原始记录或经验证的无损/事件触发记录；SDT 主要用于长期趋势副本。
2. **容差按信号单位定义。** 对关节角、电流、力矩和接触力分别设置 CompDev；单一百分比/单一阈值不适合物理量程不同的信号。
3. **固定采样/时戳语义。** 按真实时间戳计算斜率；明确重复、缺失、乱序和跨时钟数据如何处理。AVEVA PI 将 OOO 数据绕过其压缩流程。
4. **设置最长归档间隔。** CompMax 限制长平稳段无新事件的时间；CompMin 可抑制过密输出，但可能扔掉短暂事件。需以故障/接触瞬态回放验证。参数行为与触发后归档点选择取决于实现。
5. **多通道同步要另行设计。** 各关节独立压缩可能产生不同的归档时间戳，损害姿态、相位与因果分析；对多维机器人状态，保留共享时间戳/联合触发点，或仅对派生指标压缩。
6. **评估原始与重建信号差。** 至少比较最大绝对误差、事件检出率、关键峰值/过零点偏差、频谱变化、存储节省率和 CPU 时间；误差限本身不保证这些任务指标不变。
7. **实时环隔离。** 不要在 1 kHz 电机闭环同步做日志编码或写盘；在低优先级记录线程/异步缓冲或后处理副本中做压缩，并测量 backlog 和 deadline miss。

## 与相近方法比较

| 方法 | 保留/判据 | 优点 | 主要代价 |
|------|-----------|------|----------|
| Swinging Door / SDT | 相邻锚点间分段线性重建误差不超过给定偏差 | 在线、状态小、能按趋势自适应地延长片段 | 有损；对尖峰/多变量同步与频域结论需单独评估 |
| 死区 / deadband | 当前值相对最后归档值的偏差是否超过阈值 | 简单、开销低 | 关注纵向差值，不显式约束长时间趋势插值误差 |
| 定时抽取 | 固定时间间隔保留样本 | 规则简单，保留采样时钟 | 可能漏掉两个时刻间的短峰值 |
| 原始记录 | 保存每个采样点 | 可复现和二次分析能力最好 | 存储、带宽与检索成本高 |

## 局限与风险

- **它是有损压缩，不是去噪或异常剔除。** 被压缩移除的样本可能是真实尖峰，也可能是噪声；算法自身不懂任务语义。
- **误差界不等于任务保真。** 接触冲击、保护阈值、跌倒前兆、控制稳定性和频谱分析可能依赖幅值很小但很短的结构。
- **逐通道压缩会破坏同步。** 不同关节/传感器不同步归档，可能制造并不存在的相位差或掩盖真实因果关系。
- **乱序/缺失/重复时间戳需特判。** 除产品特定行为（PI 对 OOO 绕过）外，不同实现处理并不统一。
- **参考代码有限。** emrumo/swingingdoor 是 MIT 社区实现，无 release、仅 2 commits；适合阅读算法状态机，不是工业级维护承诺。
- **历史来源层级不同。** Bristol 的 ISA 1990 会议论文本次未找到作者/ISA 可访问全文；原始专利和 AVEVA 一手文档可直接查阅，论文书目仅作经典出处索引。

## 结论

**判断：** Swinging Door 是一种适合连续趋势归档的低状态、在线有损压缩器，最安全的机器人用法是保留原始控制与安全数据，把它用于经误差/事件验证的长期遥测副本。

- 先选信号和分析任务，再定 CompDev、CompMin 与 CompMax；不要反过来用一个统一压缩参数覆盖所有通道。
- 以真实非均匀时间戳计算边界，明确乱序数据策略。
- 对接触、冲击和保护相关信号保留原始或独立事件记录。
- 多关节轨迹必须检查共享时间轴与重建后的相位/动力学量。
- 把压缩误差、事件漏检率、存储节省和 CPU/缓冲开销一起验收。

## 关联页面

- [可观测性（Logs / Metrics / Tracing）](./observability-logs-metrics-tracing.md) — 区分实时指标采集与低频历史归档。
- [具身数据清洗](./embodied-data-cleaning.md) — 先做时序对齐和语义校验，再决定是否产生有损压缩副本。
- [机器人数据：监督信号类型分流](./robot-data-supervision-signal-types.md) — 压缩不能把观测/动作/结果信号的时间关系混成同一标签。
- [具身数据飞轮：最小闭环](./embodied-data-flywheel-minimal-closed-loop.md) — 失败回放与回归验收需要可审计的数据来源。
- [控制环时延建模](../formalizations/control-loop-latency-modeling.md) — 压缩/落盘任务不应侵入硬实时路径。

## 参考来源

- [Bristol 原始专利 US4669097A](../../sources/patents/bristol_swinging_door_us4669097a.md) — 1987 年公开的 corridor/door 压缩描述。
- [Bristol 1990 年 ISA 会议论文书目](../../sources/papers/bristol_swinging_door_trending_1990.md) — Swinging Door Trending 经典出处；本次未找到作者/ISA 可访问全文。
- [AVEVA PI 官方参数与演示](../../sources/sites/aveva_pi_swinging_door_compression.md) — CompDev、CompMax、CompMin 和 PI 中的压缩路径。
- [emrumo/swingingdoor 参考实现](../../sources/repos/swingingdoor-emrumo-reference-implementation.md) — MIT Python 实现，不代表官方算法发行版。

## 推荐继续阅读

- [US4669097A 专利全文](https://patents.google.com/patent/US4669097A/en)
- [AVEVA PI Server 参数文档](https://docs.aveva.com/bundle/pi-server-s-da-admin/page/1022865.html)
- [AVEVA 2023 官方压缩演示](https://cdn.osisoft.com/osi/presentations/2023-AVEVA-San-Francisco/UC23NA-3PGK04-AVEVA_Bregenzer_Brent-Exception-Compression-and-their-Impacts-On-PI-System-Performance.pdf)
