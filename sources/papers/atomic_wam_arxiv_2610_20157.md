# Atomic-WAM: Execution-Grounded Atomic-Motion Reasoning for World Action Models（arXiv:2610.20157）

> 来源归档（ingest）

- **标题：** Atomic-WAM: Execution-Grounded Atomic-Motion Reasoning for World Action Models
- **arXiv：** <https://arxiv.org/abs/2610.20157>（PDF：<https://arxiv.org/pdf/2610.20157>）
- **代码（官方仓库，已核查）：** <https://github.com/ssj9596/Atomic-WAM>（推理与 RoboLab 评测；归档见本页「开源状态」）
- **项目页：** <https://ssj9596.github.io/Atomic-WAM-project/>
- **权重（README 链接，已核查）：** <https://huggingface.co/datasets/MochunniaN1/Atomic-WAM_Robolab>（WAM、Reasoner LoRA、tokenizer、visual processor、VAE；托管于 HF datasets 路径）
- **评测排行榜：** <https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html>（RoboLab 官方）
- **作者：** Shijun Shi、Zhe Liu、Zhi Hou、Yinan Chen、Tao Zhang、Kai Hu、Yong Liu、Jiangning Zhang
- **机构：** Zhejiang University；University of Hong Kong；ACE Robotics；Jiangnan University（来自仓库 README 作者单位）
- **入库日期：** 2026-10-15
- **一句话说明：** 视觉语言推理器先描述下一步局部原子动作，再交给固定的世界动作模型（WAM）联合预测视频与动作块；推理器以 EG-GRPO 用执行成功与动作合理性奖励后训练。主结果为 RoboLab 仿真的 real-to-sim 评测。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** 页面 Code 链接 `github.com/ssj9596/Atomic-WAM`，Weights 链接 HF（datasets 路径）；未标注 "coming soon"。Code / Data / Demos 区其余为通用第三方链接。
- **仓库核查（2026-10-15）：** README 发布计划：Paper 与项目页 ✔、WAM 权重与 Reasoner LoRA ✔、推理代码与 RoboLab 评测 ✔；**训练代码 ✘、原子动作标注与数据准备 ✘（未发布）**。提供 `scripts/setup_env.sh`、`scripts/download_weights.py`、`scripts/run_inference.sh` 及 `robolab/` 评测脚本。README 报告 RoboLab 组织方使用已发布权重完成复现（官方排行榜分数计算与论文略有不同）。
- **结论：** **部分开源**（推理、权重与 RoboLab 评测可运行；训练代码与标注未发布）。

## 摘录 1：方法（摘要与项目页）

- 任务级指令常未说明下一步物理交互；推理器先生成局部动作描述，WAM 据此预测未来视频与动作块。
- 原子动作标注：从多视角 DROID 示范分段机器人运动与夹爪变化，跨左 / 右 / 腕视角对齐、赋标签并校验（项目页）。
- EG-GRPO：WAM 固定，训练推理器；奖励 `r = r_task × (0.5 + 0.5 × r_exec)`（项目页）。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-atomic-wam.md`](../../wiki/entities/paper-atomic-wam.md)；与 [World Action Models](../../wiki/concepts/world-action-models.md)、[GRPO](../../wiki/methods/grpo.md) 互链。

## 摘录 2：结果

- **RoboLab-120，默认指令：** 39.6% SR，score 55.5（官方排行榜第二，Lakeside 为 51.0% SR / 65.3 score）。
- **RoboLab-120，模糊指令：** 31.6% SR，score 48.6（排名第一）。
- **论文消融（默认指令，三次运行）：** 监督推理器 36.7%；GRPO 无执行奖励 34.2%；EG-GRPO 39.2%。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-atomic-wam.md`（含流程总览与源码运行时序图）。
- 仅新增实体页，不改已有页。
