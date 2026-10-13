# SimVLA（arXiv:2610.11248）

> 来源归档（ingest）

- **标题：** SimVLA: Zero-Shot Sim-to-Real VLA Learning for Mobile Manipulation
- **类型：** paper / vla / sim-to-real / mobile-manipulation / synthetic-data
- **作者：** Kyoungin Baik、Youngwoon Lee
- **机构：** 延世大学、首尔大学
- **arXiv：** <https://arxiv.org/abs/2610.11248>
- **项目页：** <https://kyounginbaik.github.io/simvla/>
- **代码：** <https://github.com/kyounginbaik/SimVLA>（BSD-3-Clause；未包含完整训练配方与 checkpoint）
- **日期：** 2026-10-08（arXiv v1）
- **一句话说明：** SimAction + SimVQA 仿真预训练、SimDeploy rollout 后训练，学习无需真机示范的移动操作 VLA，并零样本迁移到真实环境。

## 核心摘录与 wiki 映射

1. **数据：** 程序化厨房由 Scene Synthesizer、Objaverse、Isaac Lab 等构建；35 个任务、100 个房屋、200k 轨迹（约 2,778 小时）。SimAction 提供动作轨迹，SimVQA 从仿真特权状态构造空间/几何/子任务问答。映射到[论文实体](../../wiki/entities/paper-simvla-zero-shot-mobile-manipulation.md)。
2. **训练：** SimAction + SimVQA 预训练后，以 SimAction + SimDeploy 做策略 rollout 后训练；论文使用冻结 PPO controller/技能监督，策略以 PaliGemma 3B 为底座并 flow-match 生成动作。
3. **评测：** 跨场景平均任务进度 54.4%，与每项任务 50 条真机示范微调的 π0.5 对照；数字只适用于论文测试设置。
4. **复现边界：** 官方仓库提供仿真和数据工作流，但当前未发布完整训练 recipe、SimDeploy policy-training adapter 与 checkpoint。

## 当前提炼状态

- [x] 核对 arXiv 摘要和 HTML 正文、项目页与代码入口
- [x] 区分论文方法与当前仓库可复现范围
- [x] 关联 VLA、manipulation 与 loco-manipulation 导航页
