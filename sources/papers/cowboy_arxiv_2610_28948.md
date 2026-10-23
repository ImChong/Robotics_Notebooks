# COWBOY：Contextual Whole-Body Manipulation（CoRL 2026）

> 来源归档（ingest）

- **英文标题：** COWBOY: A Scalable Sim-to-Real Framework for Learning Contextual Whole-Body Manipulation
- **类型：** paper
- **作者：** Jiahui Yang、Jason Jingzhou Liu、Ray Muxin Liu、Andrew E. Wang、Kenneth Shaw、Deepak Pathak、Ruslan Salakhutdinov
- **机构：** Carnegie Mellon University（CMU；项目页展示 CMU 标识）
- **会议：** Conference on Robot Learning（CoRL）2026
- **arXiv：** <https://arxiv.org/abs/2610.28948>（2026-10-20 提交）
- **项目页：** <https://cowboy-manip.com/>
- **代码：** 项目主页截至 2026-10-23 未列出官方代码仓库；不能据此确认代码已公开
- **入库日期：** 2026-10-23
- **Wiki 映射：** [COWBOY 实体页](../../wiki/entities/paper-cowboy-contextual-whole-body-manipulation.md)

## 核心论文摘录

### 1) 全身控制器处理环境几何相关的接近动作

COWBOY 使用可向量化的环境感知全身控制器（WBC），在大量仿真场景中协调移动底座与手臂接近目标，并处理自碰撞、环境碰撞和主动视觉。论文将场景适配交给控制器，降低直接用强化学习从完整场景中探索的难度。

- **对 wiki 的映射：** [全身控制](../../wiki/concepts/whole-body-control.md)、[COWBOY 实体页](../../wiki/entities/paper-cowboy-contextual-whole-body-manipulation.md)

### 2) 局部接触任务由 RL 专家完成

到达物体附近后，局部强化学习策略专注于物体交互；局部策略可嵌入不同完整场景，复用为多教师示范。这把“穿过场景到达目标”和“接触后完成抓取”拆成不同训练问题。

- **对 wiki 的映射：** [抓取与操作任务](../../wiki/tasks/manipulation.md)、[COWBOY 实体页](../../wiki/entities/paper-cowboy-contextual-whole-body-manipulation.md)

### 3) 蒸馏到统一视觉全身策略并迁移真机

论文将 WBC 与局部 RL 教师组合出的行为蒸馏为单一视觉策略。策略根据点云观测协调底座、手臂、灵巧手和可动相机，并根据物体及环境几何改变底座位置、相机视角、手臂轨迹与抓取方式。项目页报告在 24 个真实场景、100 多种物体上达到 91% 成功率。

- **对 wiki 的映射：** [COWBOY 实体页](../../wiki/entities/paper-cowboy-contextual-whole-body-manipulation.md)、[移动操作](../../wiki/tasks/loco-manipulation.md)
