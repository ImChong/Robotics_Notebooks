# RoboSynChallenge（arXiv:2608.12416）

> 来源归档（ingest）

- **标题：** RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills
- **类型：** paper / dexterous-manipulation / synthetic-data / benchmark / challenge
- **arXiv：** <https://arxiv.org/abs/2608.12416>
- **项目页：** <https://edem-ai.github.io/robosynchallenge.github.io/> / <https://robosyn-bench.net/>
- **代码：** <https://github.com/EDEM-AI/RoboSynChallenge>（归档见 [`sources/repos/robosynchallenge.md`](../repos/robosynchallenge.md)）
- **数据/权重：** [HuggingFace RoboSynChallenge](https://huggingface.co/RoboSynChallenge)
- **入库日期：** 2026-08-19
- **通读：** 已完成
- **核对范围：** 2026-11-21 核对 arXiv:2608.12416v1 PDF 全文 1,354 行及 `EDEM-AI/RoboSynChallenge@45d9b9c` `scripts/eval_policy.py`、`evaluation_results/README.md`
- **一句话说明：** 基于双 AgileX Piper 6-DoF 机械臂的 10 任务灵巧操作基准：EmbodiChain 每任务合成 1,000 条训练轨迹 + 每任务 60 条真机评测轨迹（CC BY-NC-SA 4.0），π0.5 在 sim/real 分别达 38.50% / 33.00%。

## 开源状态（步骤 2.5）

- **仓库核查（2026-08-19）：** EmbodiChain 安装、数据采集、PI0/PI0.5/Motus 训练评测、ACT/DP 包装；HF 21 套 sim/real 数据集 + 多任务 checkpoint。
- **结论：** **已开源、可运行**（框架 + 数据 + 部分权重）。

**对 wiki 的映射：** [`wiki/entities/paper-robosynchallenge.md`](../../wiki/entities/paper-robosynchallenge.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查
