# SPARK-VLN（项目页）

> 来源归档（ingest 关联资料）

- **标题：** SPARK-VLN — Dynamic Social Vision-Language Navigation
- **类型：** site / project-page
- **项目页：** <https://hutslib.github.io/SPARK-VLN.dc.html>
- **论文：** <https://arxiv.org/abs/2607.16806>
- **机构：** 香港科技大学（HKUST）；香港科技大学（广州）（HKUST-GZ）；新加坡国立大学（NUS）；浙江大学（ZJU）
- **入库日期：** 2026-09-07
- **一句话说明：** 动态社会 VLN 快慢双系统官方入口——逐 token 隐状态流对接 flow-matching 快规划器，配套 Idealized / Realistic 人中心基准与 staleness 统计可视化。

## 开源核查（步骤 2.5，2026-09-07）

| 链接 | 状态 |
|------|------|
| arXiv / 项目页论文摘要 | **已发布** |
| 项目页 **Code** 按钮 | **`href="#"` 占位**，无 GitHub / HF / 数据集外链 |
| 模型权重 / 基准数据 / 训练代码 | **截至入库日未列** |

**结论：** **确认未开源**。论文写 Under Review；复现仅能读方法与图表，不能按官方入口跑训练或评测。

## 页面要点（与论文对齐）

- **核心对比图：** Reason-Then-Act（推理完才动）vs Blocking Dual-System（快系统只在慢系统结束后更新）vs **Token Streaming**（蓝箭头逐 token 更新快规划器）。
- **三模块叙事：** Token-Wise Hidden Streamer → Sequence-to-Slot Latent Bridge → Evolving Latent Conditioner + flow-matching expert planner。
- **基准轴：** Idealized（推理时暂停）vs Realistic（推理时行人继续动）；社会场景维度（正面、路口、跟随、转角）；显式 **staleness stats**。
- **作者机构脚注：** HKUST¹、HKUST-GZ²、NUS³；通讯作者 Junwei Liang。

## 对 wiki 的映射

- [`wiki/entities/paper-spark-vln.md`](../../wiki/entities/paper-spark-vln.md) — 论文实体
- [`sources/papers/spark_vln_arxiv_2607_16806.md`](../papers/spark_vln_arxiv_2607_16806.md) — 论文摘录
