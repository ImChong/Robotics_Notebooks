---
type: source
source_type: repo
title: "mr-nav: A minute-scale training framework for microrobot navigation"
url: "https://github.com/yinghansun/mr-nav"
project_id: minute-scale-microrobot-navigation
paper: "2608.00854"
license: BSD-3-Clause
status: public
updated: 2026-10-21
summary: "官方微型机器人导航训练实现，包含向量化环境、RL 算法、配置、训练/评估脚本和已训练模型。"
---

# mr-nav

官方代码仓：<https://github.com/yinghansun/mr-nav>

- **关联论文：** [Minute-scale training for microrobot navigation](../papers/minute_scale_microrobot_navigation_arxiv_2608_00854.md)，arXiv:2608.00854；*Nature Machine Intelligence*，2026-09-28，DOI: [10.1038/s42256-026-01305-w](https://doi.org/10.1038/s42256-026-01305-w)。
- **代码状态：** 已公开；仓库采用 BSD-3-Clause。
- **目录：** `alg/`（RL 算法）、`cfg/`（配置）、`env/`（环境）、`dataset/`（示例数据）、`scripts/`（训练与评估）、`saved_model/`（模型）、`utils/`。
- **训练入口：** `python ./scripts/train.py --env vessel_env --num_run 1 --num_epoch 500`。
- **评估入口：** `python ./scripts/play.py --env vessel_env --model_path ./saved_model/VesselEnv/model_499.pt`；README 也提供自由空间示例。
- **资源要求：** README 建议训练 GPU 至少 8 GB 显存，并报告 RTX 3060 上约 8–10 分钟完成一次训练；仅运行已训练策略可用 CPU。
- **依赖说明：** README 列出 Python 3.12.8，并建议按平台单独安装 PyTorch，再安装项目依赖。

仓库 README 对并行环境和采样吞吐的概述数值与论文摘要略有差别，性能引用以论文摘要为准；复现时应以具体配置和实测环境为准。
