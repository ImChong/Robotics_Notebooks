# GALATEA（arXiv:2609.10050）

> 来源归档（paper）

- **标题：** GALATEA: Grounding Generated Video Plans in Simulation Towards Versatile Dexterous Controllers
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.10050>
- **PDF：** <https://arxiv.org/pdf/2609.10050>
- **项目页：** <https://boyuan-an.github.io/GALATEA/>
- **代码：** <https://github.com/boyuan-an/GALATEA>（截至 2026-09-19 仓库为 README/资产占位，RL 与 HOI 重建代码 **待 2026-11 前发布**）
- **入库日期：** 2026-09-19
- **一句话说明：** 生成 HOI 视频 → 立体深度/掩码/手追踪重建 metric 轨迹 → 仿真 SAPG 接触跟踪 → BC+DAgger 蒸馏跨物体灵巧控制器；2500 生成片段约 2000 可用、1500+ 仿真落地；闭环真机功能抓取/非抓取推/抓后姿态跟踪。

## 开源状态

- **部分开源 / 待发布**（步骤 2.5，2026-09-19）：GitHub 已建库但 README 明确 RL 跟踪与 HOI 重建代码 **2026-11 前公开**；项目页按钮写「Code (by Nov.)」。

## 核心摘录

1. **三阶段管线：** (1) 真实首帧+语言条件视频模型生成 HOI 参考；(2) 仿真中接触感知奖励 + SAPG + 域随机化训练多物体多轨迹 HOI tracker；(3) 类别专家 BC+DAgger 蒸馏为单一跨物体策略。
2. **HOI 重建：** 立体深度、物体掩码、手追踪与联合优化；约 2500 生成 clip 中 ~2000 可用参考，仿真 grounding SR 较基线 **+25 pp** 以上。
3. **机构：** 加州大学伯克利分校（UC Berkeley）、Sharpa、香港大学（HKU）。
4. **部署：** 推理时视频模型出 motion plan，已学 tracker 闭环执行。

**对 wiki 的映射**

- [paper-galatea](../../wiki/entities/paper-galatea.md)
- [generative-world-models](../../wiki/methods/generative-world-models.md)
- [manipulation](../../wiki/tasks/manipulation.md)
