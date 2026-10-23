# AtomicRL：长程机器人操作的原子任务强化学习（arXiv:2610.30097）

> 来源归档（ingest）

- **标题：** AtomicRL: Atomic Reinforcement Learning for Long-Horizon Robotic Manipulation
- **简称：** AtomicRL
- **类型：** paper / VLA / reinforcement-learning / long-horizon / manipulation / sim2real
- **arXiv：** <https://arxiv.org/abs/2610.30097>（v1，2026-10-21）
- **作者：** Likui Zhang、Zhe Zhu、Zhihao Zhan、Yibo Bai、Hanlin Wang、Dongwei Li、Bin Wang、Tianle Zhang、Xiaodan Liang、Liang Lin
- **项目页：** 本次核查未发现论文列出的官方项目页
- **代码：** 本次核查未发现论文列出的官方代码仓库
- **入库日期：** 2026-10-23
- **一句话说明：** 将长程操作拆成原子子任务，并用语义边界约束信用分配、按瓶颈调整探索，提升 VLA 的强化学习后训练效果。

## 摘要级要点

- **问题：** 轨迹级稀疏奖励会造成延迟且有偏的信用分配；子任务掌握不均又会形成长程任务瓶颈。
- **SAPO（Semantic-Aware Policy Optimization）：** 对失败原子子任务的 TD residual 进行语义门控，限制错误跨子任务边界传播，同时保留子任务内部信用分配与价值自举。
- **BATE（Bottleneck-Adaptive Target Exploration）：** 根据各原子子任务掌握情况调整任务采样和局部探索激励，把训练资源集中到未掌握的瓶颈。
- **Real2Sim2Real：** 将仿真在线强化学习与真实机器人部署连接起来。
- **摘要所报结果：** LIBERO-LONG 两种 VLA 架构相对 PPO 分别提升 3.6% 与 4.0%；CALVIN ABC-D 全任务表现最高提升 9.5%；结合 Real2Sim2Real 后，真实任务平均提升 21.3%。提升口径与基线细节以论文实验设置为准。
- **开源状态：** 本次核查没有定位到官方项目页或代码仓库链接；后续发现官方入口后再补充。

## 核心论文摘录（MVP）

### 1) 子任务边界感知的信用分配

- **链接：** [arXiv 摘要](https://arxiv.org/abs/2610.30097)
- **摘录要点：** SAPO 使用原子子任务执行结果调节 TD residual 的跨边界传播，针对长程轨迹中信用污染问题。

### 2) 面向瓶颈子任务的自适应探索

- **链接：** [arXiv 摘要](https://arxiv.org/abs/2610.30097)
- **摘录要点：** BATE 根据各子任务掌握度重分配采样与局部探索，避免已熟练阶段占用过多训练。

### 3) 评测与迁移

- **链接：** [arXiv 摘要](https://arxiv.org/abs/2610.30097)
- **摘录要点：** 论文在 LIBERO-LONG、CALVIN ABC-D 和真实任务上评估；摘要报告的提升分别为 3.6%/4.0%、最高 9.5%、真实任务平均 21.3%。

## 对 wiki 的映射

- 计划沉淀实体页：`wiki/entities/paper-atomicrl.md`（待建）
- 相关主题：VLA 强化学习后训练、长程操作、信用分配、层级任务探索
