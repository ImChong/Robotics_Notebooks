# Spa-Bench 项目页（justintiensmith.github.io/Spa-Bench）

> 来源归档

- **标题：** Spa-Bench: A Real-Robot Benchmark for Spatial Grounding in VLA Models
- **类型：** site（项目页 + 论文 PDF + 演示视频）
- **URL：** <https://justintiensmith.github.io/Spa-Bench/>
- **论文：** <https://arxiv.org/abs/2610.12696>（归档见 [spa_bench_arxiv_2610_12696.md](../papers/spa_bench_arxiv_2610_12696.md)）
- **数据：** <https://huggingface.co/datasets/Spa-Bench/spa-bench-data-information>
- **代码：** 页面未列 GitHub 仓库（截至 2026-10-12）
- **入库日期 / 核查日期：** 2026-10-12
- **一句话说明：** Imperial College London 项目页：SO-101 真机空间落地基准，6 任务、4,092 个 episode。

## 页面核查结论

- 页面汇总关键发现：ID 空间指令成功率 67.5–73.3%，OOD 为 40.7–52.0%；直接点名比配对空间指令高 28.3–40.0 个百分点。
- 声明模型、数据集、回合与表格可在 Hugging Face 取得；页面无 “coming soon” 字样。
- HF 页面（核对时）显示两个子集 `model_evaluation`（约 2.89k 行）与 `training_data`（约 1.2k 行），CSV 自动转 Parquet，字段含 Model、Task、Episode、Prompt、Scene Setup、Condition 等；许可与代码链接未在抓取内容中出现。
- 无训练/评测代码仓库链接，故不建 `sources/repos/`。判定：**部分开放（数据/评测记录，无代码）**。

## 对 wiki 的映射

- [Spa-Bench 详情节点](../../wiki/entities/paper-spa-bench-spatial-grounding-vla.md)
