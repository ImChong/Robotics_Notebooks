# SteerAct：少样本时间维引导动作模型（arXiv:2610.25632）

> 来源归档（paper）

- **标题：** SteerAct: Unlocking Action Models through Few-Shot Temporal Steering
- **类型：** paper / action model / activation steering / few-shot adaptation
- **arXiv：** <https://arxiv.org/abs/2610.25632>
- **项目页：** <https://steeract.github.io>（由 arXiv 摘要页链接）
- **提交日期：** 2026-10-19
- **作者：** Kaichen Zhou、Zhiyan Li、Chongkai Gao、Yuhan Wang、Wenlong Huang 等
- **一句话说明：** SteerAct 在冻结动作模型中选择一个任务相关稀疏特征，基于其激活随时间的描述符，在推理时少样本调节激活。

## 摘要级要点

- 离线阶段使用成功与失败 rollout 选择可区分结果的稀疏特征，并建立时间描述符参考库；在线阶段据特征演化决定介入时机和幅度。
- 方法不更新动作模型权重；论文在 π0.5、Fast-WAM、RoboTwin 2.0、DexJoCo 与真机操作评测。
- 作者报告每任务 24 条带结果标签的 rollout 相比未引导模型将任务成功率提高约 10%；摘要未说明该数值适用于所有基准的相同统计口径。
- arXiv 摘要称代码与演示位于项目页；本次未取得可核验的 GitHub 仓库 URL，代码入口待复核。

## 对 wiki 的映射

- [SteerAct 论文实体页](../../wiki/entities/paper-steeract-few-shot-temporal-steering.md)
- [世界动作模型](../../wiki/entities/unifolm-world-model-action.md)
- [HINT 长程操作](../../wiki/entities/paper-hint-robot-manipulation.md)
