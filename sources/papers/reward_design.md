# reward_design

> 来源归档（ingest）

- **标题：** Reward Design & Curriculum Learning — Locomotion RL 核心论文
- **类型：** paper
- **来源：** arXiv / NeurIPS / ICRA / CoRL
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 locomotion reward shaping、自动课程学习、LLM 辅助奖励设计等方法

## 核心论文摘录

### 1) Learning to Walk in Minutes Using Massively Parallel Deep RL（Rudin et al., 2022）
- **链接：** <https://arxiv.org/abs/2109.11978>
- **核心贡献：** legged_gym 的 reward 设计：线速度跟踪 + 角速度跟踪 + 竖直方向奖励 + 接触力惩罚 + 关节速度/加速度惩罚；地形课程：从平地渐进到复杂地形；奖励权重通过消融实验确定
- **关键洞见：** 密集 reward = 速度跟踪 + 稳定性 + 能耗三项的加权；课程通过统计成功率自动晋级
- **对 wiki 的映射：**
  - [reward-design](../../wiki/concepts/reward-design.md)
  - [curriculum-learning](../../wiki/concepts/curriculum-learning.md)
  - [locomotion](../../wiki/tasks/locomotion.md)
  - [legged-gym](../../wiki/entities/legged-gym.md)

### 2) Walk These Ways: Tuning Robot Walking（Margolis et al., CoRL 2022）
- **链接：** <https://arxiv.org/abs/2212.03238>
- **核心贡献：** 命令向量条件化 reward：将步态参数（步频、步幅、接触时序）作为奖励目标；teacher policy 包含地形特权信息；单策略支持 trot/pace/bound 多步态切换
- **关键洞见：** 参数化奖励 = 把步态目标显式化 → 泛化性 + 可调性大幅提升
- **对 wiki 的映射：**
  - [reward-design](../../wiki/concepts/reward-design.md)
  - [gait-generation](../../wiki/concepts/gait-generation.md)
  - [locomotion](../../wiki/tasks/locomotion.md)

### 3) EUREKA: Human-Level Reward Design via Coding LLMs（Ma et al., ICLR 2024）
- **链接：** <https://arxiv.org/abs/2310.12931>
- **核心贡献：** 用 GPT-4 自动生成、迭代优化 reward 函数；无需人工手工设计；在 Dexterous Manipulation 任务上超过人工 reward；提供 reward evolution 的评估框架
- **关键洞见：** LLM 可作为 reward 设计的代码生成器 + 进化优化器，但仍需仿真 oracle 验证
- **对 wiki 的映射：**
  - [reward-design](../../wiki/concepts/reward-design.md)
  - [curriculum-learning](../../wiki/concepts/curriculum-learning.md)

### 4) Terrain-Adaptive Locomotion Skills Using Deep RL（Peng & van de Panne, 2016）
- **链接：** <https://arxiv.org/abs/1606.01984>
- **核心贡献：** 早期用 RL 学习地形自适应步态；reward = 速度跟踪 + 姿态稳定；展示 RL 可从无先验知识学出合理步态
- **对 wiki 的映射：**
  - [reward-design](../../wiki/concepts/reward-design.md)
  - [locomotion](../../wiki/tasks/locomotion.md)

### 5) Automatic Curriculum Learning for Deep RL: A Short Survey（Portelas et al., 2020）
- **链接：** <https://arxiv.org/abs/2003.04664>
- **核心贡献：** 系统综述自动课程学习方法：Goal-GAN / ALP-GMM / POET 等；分类：teacher-based / self-paced / procedural；分析各方法在稀疏奖励环境下的适用性
- **对 wiki 的映射：**
  - [curriculum-learning](../../wiki/concepts/curriculum-learning.md)
  - [reward-design](../../wiki/concepts/reward-design.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
