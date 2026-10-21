# THUSI-Lab/PrismWAM

> 来源归档

- **标题：** PrismWAM: Decoupling State Representations in World Action Models
- **类型：** 官方代码仓库
- **链接：** https://github.com/THUSI-Lab/PrismWAM
- **项目页：** https://prism-wam.github.io/
- **论文：** https://arxiv.org/abs/2610.26564
- **模型：** https://huggingface.co/THUSI-Lab/PrismWAM
- **核查日期：** 2026-10-21
- **一句话说明：** 提供 PrismWAM 两阶段训练配方、基准适配、预训练策略加载及 HTTP 推理服务。

## README 要点
- **任务分解：** Semantic Planning 预测稀疏 RGB 目标；Physical Rollout 预测稠密深度序列；动作专家以单向 RGB → Depth → Action 注意力读取表征并生成 32 步动作块。
- **训练阶段：** Stage 1 以当前深度和指令预测未来深度；Stage 2 联合训练 RGB 历史、当前深度、机器人状态与指令，输出 RGB 目标、深度 rollout 和动作。
- **部署入口：** `prismwam.load_pretrained` 加载 robotwin、robodojo 或 libero 策略；`scripts/serve.py` 可启动 HTTP policy server。
- **模型与评测：** README 列出 RoboTwin 2.0、RoboDojo、LIBERO 三类模型，并链接到 Hugging Face 下载；README 报告 LIBERO 99.35%、LIBERO-Plus 79.44%、LIBERO-Pro 54.48%，不同基准不可横向比较。
- **环境：** README 给出的快速开始环境为 Linux、NVIDIA GPU、CUDA 12.8 与 Python 3.13。
- **许可：** 代码仓库为 MIT；README 指出第三方组件中包含 Apache-2.0 许可，模型权重、数据集和仿真器采用各自许可证。

## 开源边界
代码、训练配方、评测适配和部署服务已公开。运行仍需按各 benchmark 准备数据、仿真环境与初始化模型；公开仓库不等于这些外部依赖都已打包，也不代表特定真机硬件已有即插即用控制接口。

## 对 wiki 的映射
- 实体页：[PrismWAM](../../wiki/entities/paper-prismwam.md)
- 项目页：[PrismWAM 项目归档](../sites/prism-wam-project.md)
- 论文：[arXiv:2610.26564](../papers/prismwam_arxiv_2610_26564.md)
- WAM 概念：[世界–动作模型](../../wiki/concepts/world-action-models.md)
