# A Physics-Consistent Benchmark for Contact-Rich Human-Robot Interaction in Assistive Care（arXiv:2609.02402）

> 来源归档（ingest）

- **标题：** A Physics-Consistent Benchmark for Contact-Rich Human-Robot Interaction in Assistive Care
- **简称：** Physics-Consistent HRC Benchmark
- **类型：** paper / human-robot-interaction / benchmark / assistive-care
- **arXiv：** <https://arxiv.org/abs/2609.02402>
- **PDF：** <https://arxiv.org/pdf/2609.02402>
- **项目页/预览仓：** <https://anonymous.4open.science/r/Physics-Consistent-Benchmark_4_HRC-8DBF/> — 归档见 [`sources/sites/physics-consistent-hrc-benchmark.md`](../sites/physics-consistent-hrc-benchmark.md)
- **代码：** `benchmark/` 已上线匿名预览快照（2026-11-21 复核；不含 SMPL-X 人体资产、策略权重与 420 次 rollout 日志）
- **入库日期：** 2026-09-03
- **最后核对：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 全文通读 `arXiv:2609.02402`（T1–T7、140 次/方法冻结协议、Table V–VI、C1–C5 力安全/物理一致性指标）并核对匿名仓库 `Physics-Consistent-Benchmark_4_HRC-8DBF`（`benchmark/run.py`、`experiments/phase3_vlm/run_demo.py`、`experiments/phase5/ch4c_replay.py`、`docs/supplement.pdf`）
- **索引来源：** [具身智能小站 8 篇盘点](../blogs/wechat_embodied_station_8_papers_open_source_2026-09-03.md)
- **一句话说明：** 辅助洗浴接触丰富 HRC 基准：可变形被动响应人体 + 物理一致评分 + 冻结 vision-only 评估协议；T1–T7 每方法 140 次运行。

## 开源状态（2026-11-21 复核）

| 组件 | 状态 |
|------|------|
| 匿名预览 README / `docs/supplement.pdf` | 已上线 |
| `benchmark/`（`run.py`、任务与适配、phase-5 scorer） | 已上线快照 |
| SMPL-X 人体资产、$\pi_{0.5}$/VoxPoser/ReKep 权重、420 条日志 | 未包含在匿名快照中 |

**结论：部分开源** — `benchmark/run.py`、任务配置与评分器已可审查，完整闭环复现仍缺受限人体资产与基线权重。

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
