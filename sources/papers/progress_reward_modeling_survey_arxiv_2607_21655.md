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
- **原文版本：** arXiv:2607.21655v1（2026-07-22；[HTML 正文](https://arxiv.org/html/2607.21655v1)、[摘要与版本](https://arxiv.org/abs/2607.21655v1)）
- **复核日期：** 2026-11-15
- **核对范围：** 逐节阅读 arXiv v1 HTML 全文：Abstract、§1 引言、§2.1–2.3 状态/目标/输出接口、§3.1–3.4 四类奖励构造、§4.1–4.4 数据监督和三类评测、§5 局限与未来方向，检查 Figure 1–4 的图注及参考文献条目；未逐项验证全部被引论文的独立实验结果，未实测任何策略或训练代码。
- **通读：** 已完成
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


## 2026-11-15 原文全文复核（arXiv v1）

> 本节核查的是该**综述论文**的原文及结构性主张，而不是将其所引用的几十篇独立论文一概标为已通读；全文 HTML 对应 [arXiv:2607.21655v1](https://arxiv.org/html/2607.21655v1)。

### §1–§2：进度状态与输入/输出

- 原文 §1 将进度定义为**受任务条件约束、依赖历史的潜在状态**；外表相似的两帧可能分别代表首个子目标与已完成数个子目标后的重复操作，因此单帧图片并不总能判断。
- §2.1 将当前状态接口拆成单观察、短窗或轨迹前缀/完整轨迹、before-after 等关系比较，以及仿真/状态 API；全轨迹视频适用于事后分析，但不能在未执行完时实时取得。
- §2.2 的目标接口是语言、目标图/演示与程序化谓词；§2.3 的输出是状态标量、局部增量、相对排序和可执行奖励代码。比较式进度不能直接当作标定绝对进度。
- **映射：** [过程奖励建模的“接口三维”](../../wiki/concepts/progress-reward-modeling.md) 与原文 §2 对齐。

### §3：四种构造范式的机制区别

1. **冻结基础模型**（§3.1）：CLIP 类相似度、TOPReward 的“true” token 概率、GVL 时间判断属于可零样本调用的**语义先验**，不是自然标定的绝对奖励；通常仍需归一、差分等后处理。
2. **时序/相对监督**（§3.2）：示范时序、goal proximity、偏好比较可扩规模，但“时间更晚”只在特定成功轨迹假设下才可能表示任务更接近完成。
3. **指令微调进度预测**（§3.3）：进度、成功、before-after Δ、偏好或带推理的分数作为显式监督目标（文中举 RoboReward / Robometer / Robo-Dopamine / VLAC / ProgressLM）。
4. **程序化奖励**（§3.4）：Text2Reward、Eureka、Code as Reward 等输出的是用于执行的奖励函数；强依赖状态访问和任务分解，不等于视觉信息不足时仍有可靠物理反馈。
- **映射：** [四类构造范式](../../wiki/concepts/progress-reward-modeling.md) 的优势和风险不是互斥排名，而是构造来源的分类。

### §4：监督的三种人机分工，评测的三种证据

- **数据监督**（§4.1）：人工标注（可以直接标尺度/时间锚）、人机协同标注（稀疏锚点 + 插值与复核）、全自动（时序、环境真值、VLM 教师与扰动），三者在精确语义和规模之间有权衡；自动失败样本也需符合物理合理性。
- **保真**（§4.2）：显式百分比校准、时间次序、相对排序、任务指令接地和不可判定条件是不同维度。仅有单调性并不证明**数值标定准确**，遇遮挡/模糊任务应支持拒答或不确定性。
- **鲁棒泛化**（§4.3）：新任务、场景/视角、不同本体，以及失败、回退、停滞、恢复的非单调情形；仅用成功轨迹不能覆盖最困难的退化操作。
- **下游效用**（§4.4）：在线 RL、离线重标、检索与规划各自测试的是实际用途，策略成功率提升不能单独证明评估器进度曲线保真。
- **映射：** [三类评测透镜](../../wiki/concepts/progress-reward-modeling.md)；[具身评测路线](../../roadmap/depth-embodied-eval.md) 将过程评价与政策成功区分开来。

### §5：论文明确的部署限制

- 稀疏时间采样、长平台期、细小接触变化使过程信号不敏感；真实物理接触可能需要力/触觉/本体感觉辅助。
- 大 VLM 多帧推理成本使每个底层控制步调用奖励模型不切实际，原文提出 streaming/caching/distillation/hierarchical querying **作为未来方向，不是实验已证成果**。
- 长程重复动作需要记住已完成、待完成、被撤销的子目标，固定短窗看不到的历史会使同一画面含义不同。
- 综述提供分类/分析，不提供其自身的训练算法、推理 checkpoint 或机器人控制性能提升数字；相关独立方法须回原论文验证。

## 原文与核对边界

- [全文 HTML（arXiv:2607.21655v1）](https://arxiv.org/html/2607.21655v1)
- [原始论文版本记录](https://arxiv.org/abs/2607.21655v1)
- [综述配套 Awesome 资源索引](https://github.com/sterzhang/Awesome-Progress-Models)
