# Progress Reward Modeling for Robotic Learning: A Comprehensive Survey

> 来源归档（ingest）

- **标题：** Progress Reward Modeling for Robotic Learning: A Comprehensive Survey
- **类型：** paper / survey
- **来源：** arXiv abs；配套 Awesome 索引仓
- **原始链接：**
  - <https://arxiv.org/abs/2607.21655>
  - <https://ar5iv.labs.arxiv.org/html/2607.21655>
  - <https://github.com/sterzhang/Awesome-Progress-Models>
- **作者：** Jianshu Zhang*, Keliang Wu*, Haoran Lu, Anbang Liu, Ce Zhang, Weijie Yin, Chengxuan Qian, Xiyuan Yang, Zhenyu Pan, Guo Ye, Han Liu
- **机构：** 西北大学（Northwestern）；卡内基梅隆大学（CMU）；威斯康星大学麦迪逊分校（UW–Madison）；加州大学圣巴巴拉分校（UCSB）；伊利诺伊大学厄巴纳-香槟分校（UIUC）
- **入库日期：** 2026-07-27
- **一句话说明：** 机器人学习 **过程奖励 / 进度模型** 综述：用统一接口（任务状态表示 · 目标规格 · 输出形态）+ 四种构造范式 + 数据/基准透镜，整理碎片化文献；配套 MIT 开源 Awesome 索引。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 论文项目入口 | arXiv comments 指向 Awesome 仓 |
| GitHub | <https://github.com/sterzhang/Awesome-Progress-Models> — **已开源**（MIT） |
| 内容 | 综述结构图 + 按范式整理的论文画廊（预览图、Code/Project、BibTeX） |
| 可运行训练代码 | **不适用**（索引/策展仓，非算法实现） |
| 结论 | **已开源（Awesome 索引）** |

## 核心论文摘录（MVP）

### 1) 问题：终局成功信号不够用

- **链接：** <https://arxiv.org/abs/2607.21655> §1
- **摘录要点：** 终端成功只回答「是否完成」；不问当前行为是推进、停滞还是回退。长时程、多策略路径下，过程奖励同时服务 RL 稠密信用分配、在线监控、轨迹重排、数据过滤与失败恢复。
- **对 wiki 的映射：**
  - [过程奖励建模（概念）](../../wiki/concepts/progress-reward-modeling.md)
  - [Progress Reward Survey（论文实体）](../../wiki/entities/paper-progress-reward-modeling-survey.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 2) 接口三维

- **链接：** arXiv §2
- **摘录要点：**
  - **当前状态：** 单观测 / 时序上下文 / 关系比较 / 状态·API 特权访问。
  - **目标规格：** 语言 / 目标图或演示 / 结构化·程序目标。
  - **输出：** 状态标量 / 进度增量 Δ / 排序偏好 / 可执行奖励程序。
- **对 wiki 的映射：**
  - [过程奖励建模](../../wiki/concepts/progress-reward-modeling.md)

### 3) 四种构造范式

- **链接：** arXiv §3；Awesome README
- **摘录要点：** (1) 冻结基础模型语义打分；(2) 时序/相对监督学习；(3) 指令微调式进度预测；(4) 程序化奖励构造（Text2Reward / Eureka 等）。
- **对 wiki 的映射：**
  - [过程奖励建模](../../wiki/concepts/progress-reward-modeling.md)
  - [Awesome-Progress-Models](../../sources/repos/awesome-progress-models.md)

### 4) 数据与评测透镜 + 开放问题

- **链接：** arXiv §4–§5
- **摘录要点：** 监督按人类介入度分三级；评测分 **保真度 / 鲁棒泛化 / 下游效用**——效用≠保真。开放问题：弱时间假设、部分可观测、标定/泛化、闭环可靠性与长程记忆。
- **对 wiki 的映射：**
  - [过程奖励建模](../../wiki/concepts/progress-reward-modeling.md)
  - [Progress Reward Survey](../../wiki/entities/paper-progress-reward-modeling-survey.md)
