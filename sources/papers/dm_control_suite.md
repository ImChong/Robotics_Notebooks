# dm_control_suite

> 来源归档（ingest）

- **标题：** DeepMind Control Suite
- **类型：** paper
- **作者：** Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, Timothy Lillicrap, Martin Riedmiller
- **链接：** https://arxiv.org/abs/1801.00690
- **入库日期：** 2026-05-11
- **一句话说明：** 提出基于 MuJoCo 的连续控制 RL 基准套件：统一任务结构、可解释奖励与 Python/MuJoCo API，并给出多域基准与基线实验。

## 核心论文摘录

### 1) 与 Gym 连续控制域的定位差异

- **摘录要点：** Control Suite 专注连续控制；观测按物理意义分组（位置、速度、力等）而非简单拼接；奖励统一在 \([0,1]\)（LQR 域除外）以得到可解释学习曲线与跨任务汇总指标；强调文档完善、模式统一的代码库，并覆盖与 Gym 对等的多个域且扩展更多任务。
- **对 wiki 的映射：**
  - [dm-control](../../wiki/entities/dm-control.md)
  - [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)

### 2) MDP 设计：无限视界、评估与折扣

- **摘录要点：** 任务无显式终止状态或时限，属无限视界设定；内部智能体可用折扣和式，而评估用固定长度（文中为 1000 步）回合近似长期回报；因奖励在目标附近约为 1，总回报曲线纵轴可统一理解为 \([0,1000]\) 量级，便于比较。
- **对 wiki 的映射：**
  - [dm-control](../../wiki/entities/dm-control.md)

### 3) 任务验证：物理稳定与「可解性」

- **摘录要点：** 连续域不能像 Atari 那样靠人工通关验证难度；作者用多种学习智能体反复训练，迭代任务设计直至物理稳定、难以被利用漏洞刷分，且至少一种智能体能按预期方式求解；进入基准集与 extra 集的任务据此划分。
- **对 wiki 的映射：**
  - [dm-control](../../wiki/entities/dm-control.md)

### 4) RL API：`TimeStep` 与 `suite.load`

- **摘录要点：** `environment.Base` 定义 `action_spec` / `observation_spec`、`reset`、`step`；`step` 返回含 `step_type`、`reward`、`discount`、`observation` 的 `TimeStep`；`suite.load(domain_name, task_name)` 加载单任务，`suite.BENCHMARKING` 可遍历基准子集。
- **对 wiki 的映射：**
  - [dm-control](../../wiki/entities/dm-control.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
