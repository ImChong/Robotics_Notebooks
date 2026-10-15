# RoboPrompt 官方代码仓库

- **仓库：** <https://github.com/yanwen-zou/Roboprompt>
- **项目页：** [RoboPrompt 项目页](../sites/roboprompt-project.md)
- **论文：** [RoboPrompt（arXiv:2610.10534）](../papers/roboprompt_arxiv_2610_10534.md)
- **数据集与示例检查点：** <https://huggingface.co/datasets/ywzou/Roboprompt_Play_Data>
- **核查日期：** 2026-10-15
- **开放状态：** 公开代码仓库，README 包含数据准备、训练、离线交互评测、策略服务和硬件配置入口。仓库 GitHub 元数据未声明许可证；公开可见不等于许可已明确。
- **一句话说明：** RoboPrompt 的实现仓库，包含 Evo-1 Phase I、π₀.₅ / Diffusion Policy / FastWAM Phase II 适配、提示标注、steering 组合和评测工具。

## 仓库入口

- `Evo-1/`：Phase I 提示条件动作模型及数据处理。
- `scripts/labeling/`：把 gripper-tip 轨迹投影到相机图像，生成点与轨迹提示标注。
- `scripts/realworld/train/evo1/`：Evo-1 两阶段训练脚本与配置。
- `scripts/realworld/eval/`、`steering/`：交互式评测、Phase I + Phase II 组合及策略适配器。
- `openpi/`、`diffusion_policy/`、`FastWAM/`：可接入的 Phase II 策略实现与配置。
- README 的 Quick Start 使用 `uv`；Evo-1 示例配置提供冻结视觉模型训练 5,000 步、随后联合训练 30,000 步的流程。README 还给出本地离线交互评测步骤。

## Robotics_Notebooks 映射

- 论文实体：[RoboPrompt](../../wiki/entities/paper-roboprompt.md)
- 项目页来源：[RoboPrompt 项目页](../sites/roboprompt-project.md)
- 论文来源：[arXiv:2610.10534](../papers/roboprompt_arxiv_2610_10534.md)
