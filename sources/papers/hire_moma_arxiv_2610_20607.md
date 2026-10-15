# Hierarchical Residual Policy Learning for Real-World Mobile Manipulation with Sparse Human Guidance（HiRe-MoMa，arXiv:2610.20607）

> 来源归档（ingest）

- **标题：** Hierarchical Residual Policy Learning for Real-World Mobile Manipulation with Sparse Human Guidance
- **缩写 / 框架：** **HiRe-MoMa**（Hierarchical Residual Mobile Manipulation）
- **类型：** paper / mobile-manipulation / residual-policy / human-in-the-loop / real-world-rl
- **arXiv：** <https://arxiv.org/abs/2610.20607>（Submitted 2026-10-14；PDF：<https://arxiv.org/pdf/2610.20607>）
- **项目页：** <https://hire-moma.github.io/>（页面标注 CoRL 2026）
- **代码：** 项目页与 arXiv 摘要页均未列出 GitHub / 权重链接（截至 2026-10-15）
- **作者：** Zhefei Gong、Hefeifei Jiang、Zehao Wang、Jiachen Li（通讯）
- **机构：** Northeastern University；University of California, Riverside；Georgia Institute of Technology
- **入库日期：** 2026-10-15
- **文章中的转述：** 见 [微信图文归档](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)（第 13 篇）
- **一句话说明：** 在冻结的基础策略（文中为 Diffusion Policy）之上，用与策略无关的残差加底盘 / 机械臂双头分层解耦，并以稀疏人工干预重塑 critic 价值，在 TIAGo++ 真机上约 30 分钟在线微调。

## 开源状态（步骤 2.5，核查日期 2026-10-15）

- 项目页 Code / arXiv 按钮无可用 GitHub 链接；页面外链只有作者个人主页、arXiv 与 mailto。
- 论文 HTML 未见代码或权重发布承诺。
- **结论：未开源**（截至入库日未列代码链接；论文未承诺开源）。

## 摘录 1：方法（§4）

- **分层残差架构：** 冻结基础策略，**Policy-Agnostic Residual** 在其动作上叠加加性修正。
- **Hierarchical Decoupling：** 残差拆成 mobile-base 与 manipulator-arm 两个头，采用单向条件，减轻模块间梯度干扰。
- **Expert-Guided Value Alignment：** 用低于 20% 训练覆盖的稀疏人工干预重塑 critic 的价值面，引导自主探索。

## 摘录 2：实验（§5，TIAGo++ 真机）

- **任务：** Place Only、Sweep Litter、Pick Place；每任务 20 次试验。
- **表 1（相对冻结 DP 基线，成功率）：** Place Only 50.0% → 95.0%；Sweep Litter 45.0% → 90.0%；Pick Place 22.5% → 40.0%。同表对比 +HG-DAgger 与 +Flat Residual（等价 HIL-SERL 残差）。
- **项目页摘要数字：** 约 30 分钟在线交互；平均成功率 39% → 75%；执行速度 1.29×；人工干预数据不足 20%。
- **消融（表 2）：** Flat Residual → +Hierarchical Decoupling → +Alignment 逐步叠加。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-hire-moma.md`](../../wiki/entities/paper-hire-moma.md)；与残差 RL、DP 基线、真机安全微调页互链。
