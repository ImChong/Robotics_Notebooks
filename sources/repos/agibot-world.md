# AgiBot-World

> 来源归档（国内具身开源全景）

- **标题：** AgiBot-World
- **类型：** repo
- **机构：** 智元机器人
- **链接：** https://github.com/OpenDriveLab/AgiBot-World
- **分类：** 数据集/Benchmark
- **入库日期：** 2026-09-06
- **一句话说明：** 智元机器人 开源项目 AgiBot-World（数据集/Benchmark），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/agibot-world-2026.md`](../../wiki/entities/agibot-world-2026.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/agibot-world-2026.md](../../wiki/entities/agibot-world-2026.md)

## 官方资源补核（2026-10-05）

- **代码（模型与原始数据的规范入口）：** <https://github.com/OpenDriveLab/AgiBot-World>
- **项目页：** <https://agibot-world.com/>（本次页面动态渲染未读到正文，以官方 README 补核）。
- **论文：** <https://arxiv.org/abs/2503.06669>
- **权重：** HF `agibot-world/GO-1`、`GO-1-Air`；后者无 Latent Planner。
- **数据：** <https://huggingface.co/agibot-world>，Alpha / Beta 与 2026 版本分开。
- **发布事件：** 2025-03-10 论文/研究博客；2025-09-19 GO-1 模型开源。
- **入口：** `scripts/visualize_dataset.py`、`go1/shell/train.sh`、`go1/configs/go1_sft_libero.py`、`evaluate/deploy.py`。
- **许可：** README 声明代码和数据为 CC BY-NC-SA 4.0，不能视作宽松商用授权。
- **映射：** [Colosseo / GO-1](../../wiki/entities/paper-sa-2503-06669-agibot-world-colosseo-a-large-scale-manipulation.md)。
