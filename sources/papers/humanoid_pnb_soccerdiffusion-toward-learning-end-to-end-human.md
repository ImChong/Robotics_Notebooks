# SoccerDiffusion: Toward Learning End-to-End Humanoid Robot Soccer from Gameplay Recordings

> 来源归档（ingest · 原 Paper Notebooks progress 条目已升格）

- **标题：** SoccerDiffusion: Toward Learning End-to-End Humanoid Robot Soccer from Gameplay Recordings
- **类型：** paper
- **分类：** 05_Locomotion
- **arXiv：** <https://arxiv.org/abs/2504.20808>
- **项目页：** <https://bit-bots.github.io/SoccerDiffusion/>
- **代码：** <https://github.com/bit-bots/SoccerDiffusion>（MIT）
- **机构：** University of Hamburg · Hamburg Bit-Bots
- **入库日期：** 2026-06-11
- **再核日期：** 2026-07-28
- **一句话说明：** 从 RoboCup 真机比赛录像训练 transformer 扩散端到端控制并蒸馏单步推理；跌倒恢复真机 95%；已开源数据集/权重/代码。

## 核心摘录（策展，非全文）

- 数据：88 段录像 ≈15 h；关节 50 Hz / 图像 10 Hz → SQLite。
- 模型：多编码器晚融合 + 扩散解码关节轨迹；DDIM；蒸馏为 1 步学生。
- 评测：四方向跌倒恢复真机 95% / 仿真 100%；行走踢球偏定性。
- 定位：基座 IL，不宣称超越手写栈；高层战术有限。
- **开源核查（2026-07-28）：** bit-bots/SoccerDiffusion 已开源（MIT）。

## 对 wiki 的映射

- [paper-notebook-soccerdiffusion-toward-learning-end-to-end-human](../../wiki/entities/paper-notebook-soccerdiffusion-toward-learning-end-to-end-human.md)
- [humanoid-soccer](../../wiki/tasks/humanoid-soccer.md)
- 分类父节点：[paper-notebook-category-05-locomotion](../../wiki/overview/paper-notebook-category-05-locomotion.md)

## 参考来源（原始）

- 项目页：[bit-bots-soccerdiffusion.md](../sites/bit-bots-soccerdiffusion.md)
- 仓：[soccerdiffusion.md](../repos/soccerdiffusion.md)
- 论文：<https://arxiv.org/abs/2504.20808>
- [PROGRESS.md 历史锚点](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
