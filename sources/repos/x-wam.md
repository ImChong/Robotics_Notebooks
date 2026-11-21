# X-WAM（sharinka0715/X-WAM）

> 来源归档（官方源码仓 · Apache-2.0）

- **标题：** X-WAM: Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising
- **类型：** repo
- **机构：** 小米集团 / Xiaomi Robotics 相关作者团队（`sharinka0715`）
- **代码仓：** <https://github.com/sharinka0715/X-WAM>
- **项目页：** <https://sharinka0715.github.io/X-WAM/>
- **论文：** <https://arxiv.org/abs/2604.26694>
- **权重与数据集：** [Hugging Face Checkpoints](https://huggingface.co/sharinka0715/X-WAM-checkpoints) · [X-WAM-RoboCasa](https://huggingface.co/datasets/sharinka0715/X-WAM-RoboCasa) · [X-WAM-RoboTwin](https://huggingface.co/datasets/sharinka0715/X-WAM-RoboTwin)
- **分类：** 世界动作模型（4D WAM）
- **入库日期：** 2026-09-06
- **核对范围：** 核对 `sharinka0715/X-WAM@72cfb86`（2026-06-10/2026-09-24 推送）仓库结构、`README.md`（全文 254 行）、`scripts/train_sft.py`（全文 158 行）、`evaluation/policy_broker.py`、`evaluation/policy_server.py`（前 100 行）、`evaluation/robocasa_client.py`（前 100 行）；同时对照 `arXiv:2604.26694` HTML 论文摘要与方法节。未在 GPU 环境运行训练或仿真评测。
- **通读：** 已完成
- **一句话说明：** 基于 `Wan2.2-TI2V-5B` 视频先验的统一 4D 世界动作模型官方仓库，通过轻量级交错深度分支（Lightweight Depth Adaptation）与异步去噪采样（ANS，50 步视频 vs 10 步动作）联合优化视频生成、3D 空间重建与动作执行，并在 RoboCasa（79.2%）与 RoboTwin 2.0（89.8% / 90.7%）发布后训练代码、权重与数据集。

## 开源状态（Apache-2.0）

- **已开源：** 发布 SFT 后训练脚本（`scripts/train_sft.py`，基于 PyTorch Lightning + DeepSpeed bf16-mixed）、ZMQ Broker–Server–Client 三进程在线评测框架（`evaluation/policy_broker.py`、`policy_server.py`、`robocasa_client.py`）、预处理与配置（`configs/`），以及 Hugging Face 预训练/微调权重与 RoboCasa / RoboTwin 后训练数据集；预训练基座使用 `Wan2.2-TI2V-5B`（需按 README 下载至 `checkpoints/wan22_5b/`）。

## 对 wiki 的映射

- [wiki/entities/cn-os-x-wam.md](../../wiki/entities/cn-os-x-wam.md)（源码实现实体节点）
- [wiki/entities/paper-rcl-2604-26694-unified-4d-world-action-modeling-from-video-prio.md](../../wiki/entities/paper-rcl-2604-26694-unified-4d-world-action-modeling-from-video-prio.md)（论文实体节点）
