# ActGaze 官方项目页（4open.science 匿名站点）

- **类型：** 项目站点 / 研究展示
- **核对日期：** 2026-11-09
- **页面：** https://anonymous.4open.science/w/ActGaze/
- **独立附录：** https://anonymous.4open.science/api/repo/ActGaze/file/assets/appendix.pdf
- **论文原文：** https://arxiv.org/html/2609.28955v1
- **代码：** 截至核对日，页面仅提供附录、讲解视频和四任务 rollout，**未列可运行 GitHub 仓库或训练/推理脚本、权重、数据集下载**。
- **开源状态：** 论文/演示公开，**实现未公开**；不能将匿名 4open 项目站点本身当作源代码仓库。

## 页面核对内容

- 方法是 **policy initialization → action-to-gaze distillation → gaze-to-action refinement** 三阶段；在冻结的 VLA 上做相机级 mask、patch 级 blur，再把损失差蒸馏给 gaze head，最后把 gaze head 的分布约束回动作注意力。
- 主实验：Straw / Hose / LEGO / Connector 各 50 回合，π0.5 77.0%、ActGaze 91.5%。分项 74→88、68→92、88→96、78→90（单位：成功率 %）。
- 消融：仅 Straw/LEGO，两任务平均 92.5%（完整）、37.5%（无反事实监督）、62.5%（无分阶段训练）；与四任务主实验不可混作同一口径。
- 真实实验配置、细节和鲁棒性边界以论文和附录为准，不凭项目页演示推断推理延迟或线上性能。

## 交叉引用

- [论文原始来源归档](../papers/actgaze_arxiv_2609_28955.md)
- [唯一 Wiki 实体](../../wiki/entities/paper-actgaze.md)
