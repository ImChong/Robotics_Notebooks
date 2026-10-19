# phai-lab/touchscale（TouchScale 官方代码）

> 来源归档（ingest · 官方源码仓库）

- **标题：** TouchScale: 500 Hours of Human Vision and Touch for Visual-Tactile Learning
- **类型：** repo / dataset-tools / robot-learning
- **维护方：** [PhAI Lab](https://github.com/phai-lab)
- **代码：** <https://github.com/phai-lab/touchscale>
- **项目页：** <https://touch-scale.github.io/>
- **论文：** <https://arxiv.org/abs/2610.10288>
- **数据集：** <https://huggingface.co/datasets/2077AIDataFoundation/TouchScale>
- **核查日期：** 2026-10-19
- **对应 wiki：** [TouchScale](../../wiki/entities/touchscale.md)

## 仓库内容与运行入口

官方 README 将此仓库定位为 TouchScale 的代码发布，包含：

- `qc/`：数据质量检查管线。
- `n0-vtla/`：TouchScale 视触觉 mid-training 与机器人 post-training 实现。
- 训练入口：在 `n0-vtla/` 环境安装完成后，运行 `bash train_stage1.sh`；后训练可用 `INIT=touchscale DATASET=... ASSET_ID=... bash scripts/posttrain.sh`。
- 训练依赖公开数据、归一化文件及 N0-VTLA 基础权重；运行步骤和目录约定以仓库 README 与 `n0-vtla/docs/` 为准。

## 开源与许可边界

- **代码：已公开。** README 声明仓库代码采用 MIT License。
- **例外：** `n0-vtla/` 派生自 NeoteAI 的 N0-VTLA，沿用 CC BY-SA 4.0；N0-VTLA 权重受 Gemma Terms of Use 管辖。
- **数据：部分公开。** 官方数据卡截至 2026-10-19 托管 100 小时；完整 500 小时仍是计划发布。数据卡采用 CC-BY-NC-4.0，且下载文件前须同意分享联系信息。
- **复现注意：** 代码开源不等于数据可无条件下载或可商业使用；应分别核对仓库、模型权重与数据集的许可条款。

## 关联归档

- [项目页核查](../sites/touch-scale-github-io.md)
- [论文来源](../papers/touchscale_arxiv_2610_10288.md)
- [数据集发布记录](../datasets/touchscale_huggingface.md)
