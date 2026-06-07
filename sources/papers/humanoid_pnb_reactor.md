# ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting.html>
- **分类：** 02_Motion_Retargeting
- **arXiv：** <https://arxiv.org/abs/2605.06593v1>
- **入库日期：** 2026-06-07
- **一句话说明：** 针对传统几何重定向需要 手工接触模板 / 大量调参、且仍产生 脚滑、自碰、动力学不可行 等问题，ReActor 提出 物理感知、RL 内嵌的双层框架：用户只给 稀疏语义刚体对应 与名义姿态对齐，系统自动搜索一组 有界偏移参数 把源运动 $\mathbf{m}_t$ 映到参数化参考 $\mathbf{g}_t(\mathbf{p})$；下层策略在仿真里跟踪 $\mathbf{g}_t$，上层最小化 $\mathbf{g}$ 与仿真 rollout 状态 $\mathbf{s}_t$ 的误差。论文在 两台人形 + 四足 上展示跨大差异本体的重定向，并分析近似梯度与泛化行为。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-reactor](../../wiki/entities/paper-notebook-reactor.md).

## 对 wiki 的映射

- [paper-notebook-reactor](../../wiki/entities/paper-notebook-reactor.md)
- 分类父节点：[paper-notebook-category-02-motion-retargeting](../../wiki/overview/paper-notebook-category-02-motion-retargeting.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting/ReActor__Reinforcement_Learning_for_Physics-Aware_Motion_Retargeting.html>
- 论文：<https://arxiv.org/abs/2605.06593v1>
