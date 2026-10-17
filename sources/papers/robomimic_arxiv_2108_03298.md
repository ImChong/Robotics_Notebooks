# What Matters in Learning from Offline Human Demonstrations for Robot Manipulation（arXiv:2108.03298）

> 来源归档（ingest）

- **标题：** What Matters in Learning from Offline Human Demonstrations for Robot Manipulation
- **类型：** paper
- **原始链接：**
  - arXiv：https://arxiv.org/abs/2108.03298
  - PDF：https://arxiv.org/pdf/2108.03298
  - CoRL 2022 正式论文与 PDF：https://proceedings.mlr.press/v164/mandlekar22a.html
  - 研究页：https://robomimic.github.io/study/
  - 代码：https://github.com/ARISE-Initiative/robomimic
- **作者：** Ajay Mandlekar、Danfei Xu、Josiah Wong、Soroush Nasiriany、Chen Wang、Rohun Kulkarni、Li Fei-Fei、Silvio Savarese、Yuke Zhu、Roberto Martín-Martín
- **会议：** Conference on Robot Learning (CoRL)，PMLR 164，2022
- **入库日期：** 2026-10-17
- **一句话说明：** 系统研究人工离线示范数据上的机器人操作策略学习，比较模仿学习与离线强化学习方法，并分析示范质量、历史观测、相机视角、超参数及策略评估对结果的影响。

## 核心论文摘录（MVP）

### 1) 研究设置：算法、任务与数据质量

- **链接：** https://proceedings.mlr.press/v164/mandlekar22a.html
- **摘录要点：** 论文在五个仿真、三个真实世界多阶段操作任务上比较六种离线方法：BC、BC-RNN、HBC、BCQ、CQL、IRIS。数据涵盖熟练操作者、多操作者混合质量数据和机器生成数据；观测包括低维状态与图像。
- **对 wiki 的映射：** [robomimic 框架实体](../../wiki/entities/robomimic.md)、[模仿学习](../../wiki/methods/imitation-learning.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)

### 2) 结论：人的决策历史对策略有用

- **链接：** https://robomimic.github.io/study/
- **摘录要点：** BC-RNN 与 HBC 等历史依赖方法在人工数据上优于不建模历史的策略。人类遥操作动作可能受到操控设备和此前动作影响，单帧观测未必构成充分决策状态。
- **对 wiki 的映射：** [robomimic 框架实体](../../wiki/entities/robomimic.md)、[模仿学习](../../wiki/methods/imitation-learning.md)

### 3) 结论：数据混合质量会改变离线 RL 的表现

- **链接：** https://robomimic.github.io/study/
- **摘录要点：** 批量离线 RL 对混合质量的机器生成数据较有能力，但面对混合质量的人类示范时更困难；即使对同一起始条件分别提供成功与失败示范，也会形成策略筛选挑战。
- **对 wiki 的映射：** [robomimic 框架实体](../../wiki/entities/robomimic.md)、[在线与离线 RL 对比](../../wiki/comparisons/online-vs-offline-rl.md)

### 4) 评估与工程含义：训练损失不等于任务成功

- **链接：** https://robomimic.github.io/study/
- **摘录要点：** 论文指出验证目标与策略执行评价之间可能不一致，最佳验证检查点可能明显弱于实际表现最佳的检查点；观测选择与超参数也会显著影响策略成功率。因此离线损失不能单独替代环境回放评估。
- **对 wiki 的映射：** [robomimic 框架实体](../../wiki/entities/robomimic.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)

## 对 wiki 的映射（汇总）

- [robomimic 框架实体](../../wiki/entities/robomimic.md) — 统一介绍框架、数据、算法、复现和适用边界
- [模仿学习](../../wiki/methods/imitation-learning.md) — BC、BC-RNN 与 HBC 的方法背景
- [强化学习](../../wiki/methods/reinforcement-learning.md) — BCQ、CQL、IRIS 等离线 RL 对照
- [在线与离线 RL 对比](../../wiki/comparisons/online-vs-offline-rl.md) — 固定数据学习的评估与分布偏移问题
