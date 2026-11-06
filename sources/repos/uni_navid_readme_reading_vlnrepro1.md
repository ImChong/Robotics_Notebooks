# Uni-NaVid 官方仓库 README 通读（VLN 复现范式页支撑）

> 来源归档（原文通读 · 支撑 `wiki/overview/vln-open-source-repro-paradigms.md`）

- **标题：** Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks（官方 GitHub README）
- **作者：** Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, He Wang
- **类型：** repo / navigation / vla
- **原文：** <https://github.com/jzhzhang/Uni-NaVid>（2026-11 经代理获取 raw README 全文）；论文 <https://arxiv.org/abs/2412.06224>
- **一句话说明：** RSS 2025 的视频式导航 VLA 的微调与评测代码，统一 VLN、跟踪等具身导航任务。
- **核对范围：** 通读 README 全文（Release、Install、Preparation、Train、Evaluation、Citation、Acknowledgments）。未读论文及附录、未读源码，也未运行代码。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **发布内容：** 训练代码、离线评测代码、基准评测（VLN-CE、EVT-Bench）、VLN-CE RxR 小份数据。
- **环境：** conda Python 3.10；`pip install -e .`；`flash-attn==2.5.9.post1`。
- **模型：** 视觉编码器 EVA-CLIP；语言模型 Vicuna-7B v1.5；已微调权重 Uni-NaVid (7B)（Hugging Face）。
- **数据：** 仅提供论文数据的小子集（`open_uninavid_sampled_500.json` 与 nav_videos），来自 VLN-CE R2R/RxR 训练集、EVT-Bench、ObjectNav、EQA；因许可限制未使用 L3MVN 做 ObjectNav 模仿学习，ObjectNav 性能可能略降。
- **训练：** 改 `DATA_PATH`/`MODEL_PATH`；`uninavid_stage_1.sh` 从 Vicuna-7B 起训（需足量数据），`uninavid_stage_2.sh` 在 Uni-NaVid 基础上微调。
- **推理速度：** 评测时用在线 token merging（`run_type=eval`），单张 A100 约 5 Hz；量化等可再提速。
- **离线评测：** `python offline_eval_uninavid.py test_cases/vln_1 Ourpur_dir`（另有 `tracking_1` 样例），样例视频在 Hugging Face。
- **基准成绩：** VLN-CE R2R Val SR 51.8 / SPL 47.7；RxR Val SR 56.1 / SPL 44.5；EVT-Bench STT/DT/AT 的 SR 为 53.3/31.9/15.8。VLN-CE 评测代码在 NaVid-VLN-CE 仓库，EVT-Bench 在 TrackVLA 仓库。
- **致谢与说明：** 基于 LLaMA-VID 与 NaVid；这是开源版本，部分函数为规避许可重写。

## 对 wiki 的映射

- [`wiki/overview/vln-open-source-repro-paradigms.md`](../../wiki/overview/vln-open-source-repro-paradigms.md)
