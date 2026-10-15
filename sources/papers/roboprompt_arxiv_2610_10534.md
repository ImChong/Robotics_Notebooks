# RoboPrompt: Intuitive Robot Policy Steering with Sparse Human Input（arXiv:2610.10534）

> 来源归档（ingest）

- **标题：** RoboPrompt: Intuitive Robot Policy Steering with Sparse Human Input
- **类型：** paper
- **作者：** Yanwen Zou、Chenyang Shi、Guoxuan Xu、Wenye Yu、Wendi Chen、Ye Pan、Cewu Lu、Chuan Wen
- **机构：** 上海交通大学、上海创智学院、诺玛矩阵（Noematrix Ltd.）
- **arXiv：** <https://arxiv.org/abs/2610.10534>（v1，2026-10-07）
- **项目页：** <https://yanwen-zou.github.io/Roboprompt-Website/>
- **代码：** <https://github.com/yanwen-zou/Roboprompt>
- **数据与示例检查点：** <https://huggingface.co/datasets/ywzou/Roboprompt_Play_Data>（README 提供训练数据及 Evo-1、π₀.₅ 示例检查点的下载命令）
- **入库日期：** 2026-10-15
- **一句话说明：** 让用户通过图像上的点、轨迹或粗略方向提示引导已有机器人策略：Evo-1 将稀疏意图转成动作草案，扩散或流匹配策略再细化动作，并把筛选后的 rollout 用于 DAgger 改进策略。

## 核心摘录（面向 wiki 编译）

1. **两阶段 steering：** Phase I 是 0.77B 参数的意图转换器 Evo-1，接收当前相机观测、任务指令与点/轨迹/方向提示，生成时空连续的动作块；Phase II 复用已有基础策略，不改其架构，也不为 steerability 微调。
2. **适配不同生成策略：** 对 diffusion policy 使用 Progressive Biased Denoising（PBD），逐个动作块增加去噪程度，让控制从人工草案平滑转向策略先验；对 flow-matching 策略使用 Noise-Augmented Flow Reversal Steering（NA-FRS）。
3. **从 rollout 改进策略：** 使用 Time-weighted Optimal Transport（TOT）按与专家轨迹的时序接近程度筛选 steered rollout，再与离线演示共同用于 DAgger 式策略改进。
4. **跨策略验证：** 论文报告在 Diffusion Policy、π₀.₅ 与 FastWAM 上均可 steering；项目页称 prompt alignment 达到至少 95%，无需为 steering 微调基础策略。
5. **实验结果与口径：** π₀.₅ 在三个真实任务上经过 2–3 轮 DAgger 后，平均任务成功率由 80.0% 提升至 95.5%（+15.5 个百分点），人工干预次数平均由 2.86 降至 1.60（减少 44.0%）。在 Insert Bread 单任务跨三种基础策略的实验中，论文摘要报告平均成功率提升 21.3%，人工干预数减少 81.9%（2.60 降至 0.47）；两组结果的任务与策略范围不同，不应混作同一平均值。
6. **开放边界：** 项目页链接到公开代码仓库；仓库 README 提供数据、示例检查点、训练及离线交互评测入口。GitHub 仓库元数据未声明许可证，复用前应核实许可条件。

**对 wiki 的映射：** [RoboPrompt](../../wiki/entities/paper-roboprompt.md)

## 版本与引用

- arXiv v1，提交于 2026-10-07。
- 数据与数字以论文、项目页及官方仓库 README 为准。
- BibTeX：Zou et al., “RoboPrompt: Intuitive Robot Policy Steering with Sparse Human Input,” arXiv:2610.10534, 2026.
