# Qwen-Drive-1.0 官方代码仓库

> 来源归档（repo）

- **标题：** Qwen-Drive-1.0 — An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving
- **类型：** repo / autonomous-driving / vision-language / inference
- **GitHub：** <https://github.com/QwenLM/Qwen-Drive-1.0>
- **对应论文：** [arXiv:2609.00111](https://arxiv.org/abs/2609.00111)
- **项目文章：** <https://qwen.ai/blog?id=qwen-drive-1.0>
- **权重：** <https://modelscope.cn/models/Qwen/Qwen-Drive-1.0-4B>
- **机构：** Qwen Team；华中科技大学（HUST）
- **许可证：** Apache-2.0
- **入库日期：** 2026-10-14
- **一句话说明：** Qwen-Drive 推理、demo 与评测实现；以共享 Qwen3.5-4B 基座挂载 BEV 感知与规划权重。
- **开源状态：** **部分开源：** 推理 / 评测源码和 demo 数据已发布，权重页可访问；文件树中未见训练代码。

## 仓库核查

- **主要代码：** src/qwen_drive/、src/qwen_drive_perception/；含场景、图像读取、VQA / 规划推理、BEV 感知和可视化模块。
- **入口：** scripts/demo.py 可运行 VQA、direct planning、reasoning planning；scripts/run_perception.py 为感知入口。评测脚本覆盖 NAVSIM、Waymo 与 PhysicalAI。
- **demo 数据：** data/demo/ 提供规划场景和帧数据；README 给出规划结果与标注轨迹的 ADE / FDE 对照。
- **模型结构：** 根目录约 9.1 GB 为 VLM；planner-sft/、planner-rl/ 各约 2.1 GB；perception/ 约 0.5 GB。
- **运行环境：** README 以 Python 3.10 为例，建议 GPU 显存 24 GB+；pyproject.toml 依赖 PyTorch、Transformers 5.14.x、FlashAttention 2 等。
- **复现边界：** 开放 demo / 推理 / 评测与权重下载路径；未发现训练入口或完整训练配置，训练配方需依赖论文和项目文章说明。

## 与 wiki / sources 的关系

- 论文项目实体：[paper-qwen-drive-1-0](../../wiki/entities/paper-qwen-drive-1-0.md)
- 论文摘录：[qwen_drive_1_0_arxiv_2609_00111](../papers/qwen_drive_1_0_arxiv_2609_00111.md)
- 官方文章：[qwen-drive-1-0](../sites/qwen-drive-1-0.md)
