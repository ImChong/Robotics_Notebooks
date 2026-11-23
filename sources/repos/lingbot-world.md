# lingbot-world（Robbyant 官方 1.0 代码仓）

> 来源归档（ingest · 官方代码仓）

- **标题：** LingBot-World: Advancing Open-source World Models
- **类型：** repo
- **机构：** 蚂蚁灵波（Robbyant）
- **链接：** <https://github.com/robbyant/lingbot-world>
- **项目页：** <https://technology.robbyant.com/lingbot-world>
- **论文：** <https://arxiv.org/abs/2601.20540>
- **许可：** Apache-2.0（`LICENSE.txt`）
- **基座：** [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- **入库日期：** 2026-09-06（2026-11-21 补核 1.0 官方仓与论文原文）
- **核对范围：** 2026-11-21 核对 `robbyant/lingbot-world@a43bec7` 官方 `README.md`、`generate_fast.py`（参数与推理入口），并通读主论文 [arXiv:2601.20540v1](https://arxiv.org/html/2601.20540) HTML 全文（摘要、§1–§3 Data Engine 与三阶段训练流程、表 1 对比）；未运行模型推理（无本地权重与多卡 GPU）。
- **通读：** 已完成
- **一句话说明：** 蚂蚁灵波开源世界模型 1.0 主仓（Apache-2.0），包含 Base (Cam) / Base (Act) / Fast（`generate_fast.py` + `run_fast.sh`）推理脚本与权重索引；README 已声明停止积极维护，后续更新迁移至 [lingbot-world-v2](https://github.com/robbyant/lingbot-world-v2)。

## 开源状态与入口

- **已开源（Apache-2.0）：** 推理代码、HF/ModelScope 权重（`lingbot-world-base-cam`、`lingbot-world-fast`）及 NF4 量化社区版可用；官方 README 标注本仓库已不再积极维护，推荐使用 `robbyant/lingbot-world-v2`。
- **推理入口：** `generate.py`（Base Cam/Act，`--task i2v-A14B`、`--allow_act2cam` / `--action_string`）与 `generate_fast.py` / `run_fast.sh`（Fast 因果 KV 缓存加速，16 fps、<1 s 延迟）。

## 对 wiki 的映射

- [lingbot-world.md](../../wiki/entities/lingbot-world.md)
- [paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md](../../wiki/entities/paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md)
