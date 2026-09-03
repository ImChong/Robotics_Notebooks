# A Physics-Consistent Benchmark for Contact-Rich Human-Robot Interaction in Assistive Care（arXiv:2609.02402）

> 来源归档（ingest）

- **标题：** A Physics-Consistent Benchmark for Contact-Rich Human-Robot Interaction in Assistive Care
- **简称：** Physics-Consistent HRC Benchmark
- **类型：** paper / human-robot-interaction / benchmark / assistive-care
- **arXiv：** <https://arxiv.org/abs/2609.02402>
- **PDF：** <https://arxiv.org/pdf/2609.02402>
- **项目页/预览仓：** <https://anonymous.4open.science/r/Physics-Consistent-Benchmark_4_HRC-8DBF/> — 归档见 [`sources/sites/physics-consistent-hrc-benchmark.md`](../sites/physics-consistent-hrc-benchmark.md)
- **代码：** `benchmark/` **Coming soon**（截至 2026-09-03）
- **入库日期：** 2026-09-03
- **索引来源：** [具身智能小站 8 篇盘点](../blogs/wechat_embodied_station_8_papers_open_source_2026-09-03.md)
- **一句话说明：** 辅助洗浴接触丰富 HRC 基准：可变形被动响应人体 + 物理一致评分 + 冻结 vision-only 评估协议；T1–T7 每方法 140 次运行。

## 开源状态（步骤 2.5，2026-09-03）

| 组件 | 状态 |
|------|------|
| 匿名预览 README / appendix | 已上线 |
| benchmark 实现 / rollout / scoring | **Coming soon** |

**结论：部分开源 / 待发布** — 论文与附录可引用；可运行基准待发布。

## 核心摘录

### 摘录 1：评测哲学

- 任务成功之外还有 physics-aware scores、力安全、区域正确性。
- 方法侧仅 RGB-D + 机器人传感；评分侧可用完整仿真状态。

**对 wiki 的映射：** [paper-physics-consistent-hrc-benchmark](../../wiki/entities/paper-physics-consistent-hrc-benchmark.md)

### 摘录 2：冻结协议结果（文内）

- LLM-augmented state machine 任务成功 **72.9%** → 经区域+力安全筛查 **56.4%**。
- VoxPoser **27.9%**（接触更轻更稳）；zero-shot π₀.₅ **0.7%**。

**对 wiki 的映射：** [paper-physics-consistent-hrc-benchmark](../../wiki/entities/paper-physics-consistent-hrc-benchmark.md)

## 当前提炼状态

- [x] 匿名预览仓核查
- [x] wiki 映射
