# TurboVLA 原文核对｜TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM

- **资料类型：** paper
- **论文：** https://arxiv.org/html/2607.27205v3
- **arXiv：** 2607.27205（v3，2026-09-27）
- **官方项目：** https://h-embodvis.github.io/TurboVLA/
- **官方代码：** https://github.com/H-EmbodVis/TurboVLA
- **模型：** https://huggingface.co/H-EmbodVis/TurboVLA
- **机构：** 华中科技大学（HUST）、华为技术有限公司（华为参与作者名单，论文页面作者抬头依版本可能简写）
- **核对范围：** 2026-11-14 核对 arXiv HTML v3 的 Abstract、§1–5、A/B/C 附录（网络架构、训练配置、LIBERO/RoboTwin 指标表、加速与时延定义、每任务结果），以及官方 README 的安装、权重、训练/评测入口；静态检查 `experiments/libero/train.py`、`experiments/libero/evaluate.py`、`turbovla/training/train_mixed.py`。
- **通读：** 已完成

## 方法与关键边界

- 传统 LLM 介入路径 V→L→A 计算大；本文改为独立视觉编码和轻量语言编码，经**双向 cross-attention**融合后，连同机器人状态送入连续 action chunk 解码器。
- LIBERO 使用 DINOv3 ViT-B/16、BERT 文本编码、ACT 风格三层动作解码器、12 步 chunk；RoboTwin 改用 ViT-L 与 50 步动作预测。不能将两组指标当成同一个 checkpoint。
- arXiv v3 表格中的 LIBERO 四套均值是 **97.6%**，模型规模 0.2B、RTX 4090 31.2 ms、约 0.9 GiB；论文摘要中的四舍五入写法可能不同，本站以该版表格为准。
- v3 RoboTwin 2.0 的 0.4B 模型均值为 Clean 88.02 / Random 88.10（均值 88.06）。GPU CUDA Graph 加速表另给 11.3 ms，配合进一步优化为 7.42 ms；不能把优化加速后的数字与基线数字互换。
- 论文明确局限：聚焦执行层具体操作，不是“清理房间”这类开放长程指令的高层任务分解系统。

## 源码阅读依据

- `experiments/libero/train.py` → `turbovla.training.train_mixed.main()`；后者解析混合数据集统计与训练配置。
- `experiments/libero/evaluate.py` → `vla_adapter.rollout.main()`；CLI 从官方 LIBERO README 传入 checkpoint、suite、render/runtime 配置。
- RoboTwin 的官方安装/训练/评测入口为 `scripts/robotwin/prepare_data.sh`、`scripts/robotwin/train.sh`、`scripts/robotwin/evaluate.sh`；已核对 README 路径，未在本环境执行。

## 对 wiki 的映射

- [TurboVLA 唯一论文详情](../../wiki/entities/paper-turbovla.md)
