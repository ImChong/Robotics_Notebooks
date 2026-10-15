# RoboPrompt 项目页

- **类型：** 论文项目页
- **标题：** RoboPrompt: Intuitive Robot Policy Steering with Sparse Human Input
- **链接：** <https://yanwen-zou.github.io/Roboprompt-Website/>
- **论文：** [RoboPrompt（arXiv:2610.10534）](../papers/roboprompt_arxiv_2610_10534.md)
- **代码：** [RoboPrompt 官方仓库](../repos/roboprompt.md)
- **核查日期：** 2026-10-15
- **开源状态：** 项目页直接提供论文与 GitHub 仓库链接；仓库 README 另提供公开数据集与示例检查点入口。仓库元数据未声明许可证，使用前需核对许可。
- **一句话说明：** 用稀疏的视觉提示（目标点、手绘轨迹、粗略方向）引导机器人策略，并通过两阶段动作生成与筛选后的部署 rollout 改进策略。

## 项目页要点

1. **用户输入：** 在相机视图中画出短轨迹、指定目标点，或给出粗略运动方向；无需专用遥操作设备。
2. **两阶段动作生成：** 0.77B Evo-1 把提示转为连续 6D 动作草案；Diffusion Policy、π₀.₅ 或 FastWAM 再通过 diffusion / flow matching 细化动作。
3. **基础策略复用：** 项目页强调不需要修改基础策略架构，也不需要为 steerability 微调基础策略。
4. **部署闭环：** 用 TOT 筛选 steered rollout，与原始离线演示共同用于 DAgger；项目页报告 π₀.₅ 三任务平均成功率由 80.0% 提升至 95.5%，人工干预由 2.86 次降至 1.60 次。

## Robotics_Notebooks 映射

- 论文实体：[RoboPrompt](../../wiki/entities/paper-roboprompt.md)
- 论文来源：[arXiv:2610.10534](../papers/roboprompt_arxiv_2610_10534.md)
- 代码来源：[官方仓库](../repos/roboprompt.md)
