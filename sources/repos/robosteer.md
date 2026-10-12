# RoboSteer 数据集（Hugging Face）

> 来源归档

- **标题：** RoboSteer 基准数据集
- **类型：** repo（数据集）
- **来源：** 浙江大学 / 宇树科技 / 新加坡国立大学 / 中南大学
- **代码：** 无 GitHub 仓库（截至 2026-10-12）；数据集：<https://huggingface.co/datasets/YanCORANV/RoboSteer>
- **项目页：** <https://robosteer.github.io/>
- **论文：** <https://arxiv.org/abs/2610.10198>
- **入库日期：** 2026-10-12
- **一句话说明：** 773,716 条任务定义的行为可操控性基准，23 个 TAR 归档约 355.35 GiB。
- **沉淀到 wiki：** [`wiki/entities/paper-robosteer-behavior-steerability-benchmark.md`](../../wiki/entities/paper-robosteer-behavior-steerability-benchmark.md)

## 数据卡摘要

- 任务数：Level 1 630,623；Level 2 135,329；Level 3 7,764。
- 文件：`archives/`（含 SHA256SUMS）、`docs/`（附录图、任务图、JSON 字段说明）、`Data/`（Level1/2/3/Shared）、`Tasks/`（按层级的任务 JSON，含 `input` 与 `ground_truth`）；Hub 显示总大小约 382 GB；同时保留压缩与解压需约 707.40 GiB。
- 许可：数据卡未给统一许可，团队自有数据与第三方派生材料再分发条款仍在审查；Level 2 Order/Times 使用 AMASS 与 BABEL，带非商用/不得再分发条款。
