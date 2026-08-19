# RoboSynChallenge（arXiv:2608.12416）

> 来源归档（ingest）

- **标题：** RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills
- **类型：** paper / dexterous-manipulation / synthetic-data / benchmark / challenge
- **arXiv：** <https://arxiv.org/abs/2608.12416>
- **项目页：** <https://edem-ai.github.io/robosynchallenge.github.io/> / <https://robosyn-bench.net/>
- **代码：** <https://github.com/EDEM-AI/RoboSynChallenge>（归档见 [`sources/repos/robosynchallenge.md`](../repos/robosynchallenge.md)）
- **数据/权重：** [HuggingFace RoboSynChallenge](https://huggingface.co/RoboSynChallenge)
- **入库日期：** 2026-08-19
- **一句话说明：** 合成 state-action 训练通用操作策略，**最终只在未见真实环境**评测；基线覆盖 Transformer / Diffusion / VLA / WAM。

## 开源状态（步骤 2.5）

- **仓库核查（2026-08-19）：** EmbodiChain 安装、数据采集、PI0/PI0.5/Motus 训练评测、ACT/DP 包装；HF 21 套 sim/real 数据集 + 多任务 checkpoint。
- **结论：** **已开源、可运行**（框架 + 数据 + 部分权重）。

**对 wiki 的映射：** [`wiki/entities/paper-robosynchallenge.md`](../../wiki/entities/paper-robosynchallenge.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查
