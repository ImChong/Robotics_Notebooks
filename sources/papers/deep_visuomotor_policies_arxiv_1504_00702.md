# End-to-End Training of Deep Visuomotor Policies（arXiv:1504.00702）

- **类型：** paper / primary-source reading
- **标题：** End-to-End Training of Deep Visuomotor Policies
- **作者：** Sergey Levine, Chelsea Finn, Trevor Darrell, Pieter Abbeel
- **arXiv：** https://arxiv.org/abs/1504.00702
- **HTML：** https://arxiv.org/html/1504.00702v5
- **原文 PDF：** https://arxiv.org/pdf/1504.00702
- **核对范围：** arXiv v5 摘要、§4 的 BADMM/局部动力学与控制器优化以及结论/局限；全部实验、附录推导尚未完整逐节核实。
- **通读：** 未完成
- **一句话摘录：** 通过部分观测 Guided Policy Search，将使用全状态的局部轨迹优化与仅观测图像的深度策略监督学习交替进行，直接输出机器人电机力矩。

## 基于当前已核实原文的要点

- 局部轨迹分布与局部线性高斯动力学模型通过机器人轨迹样本拟合。
- 利用 KL 约束控制轨迹更新幅度；BADMM 协调轨迹控制器与感知策略。
- 在模拟和真机精细操作任务验证，比较端到端与固定视觉特征。
- 边界：训练时依赖充分可观察的状态信息；对于与训练差异很大的视觉干扰泛化有限。

## 开放资源核查

- [Guided Policy Search 官方关联项目](https://github.com/cbfinn/gps) 是 2016 年维护的 GPS 重实现，能作为算法复现入口；不代表完整还原论文中的真机、数据及网络设置。
- 补充一手入口：https://www.jmlr.org/papers/v17/15-522.html

## 对 wiki 的映射

- [paper-deep-visuomotor-policies.md](../../wiki/entities/paper-deep-visuomotor-policies.md)
- [11 篇论文路线](../../wiki/overview/robot-learning-foundations-11-papers.md)

## 未完成事项

- 继续阅读尚未核实的正文/附录、表格及可复现资产；完成后方可将“通读”改成“已完成”。
