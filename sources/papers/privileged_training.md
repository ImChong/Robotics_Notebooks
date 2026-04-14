# privileged_training

> 来源归档（ingest）

- **标题：** 特权信息训练（Teacher-Student / RMA）核心论文
- **类型：** paper
- **来源：** RSS / NeurIPS / CoRL / Science Robotics
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 teacher-student 蒸馏、RMA 适应模块、并发训练等 sim2real 特权信息方法

## 核心论文摘录

### 1) RMA: Rapid Motor Adaptation for Legged Robots（Kumar et al., RSS 2021）
- **链接：** <https://arxiv.org/abs/2107.04034>
- **核心贡献：** 两阶段训练：Phase 1 用特权信息（地形参数、摩擦系数）训练 base policy + 环境编码器；Phase 2 用真实历史 obs 训练适应模块模仿编码器输出；无需域随机化微调即可迁移真机
- **对 wiki 的映射：**
  - [privileged-training](../../wiki/concepts/privileged-training.md)
  - [sim2real](../../wiki/concepts/sim2real.md)

### 2) Learning to Walk in Difficult Terrain（Lee et al., Science Robotics 2020）
- **链接：** <https://www.science.org/doi/10.1126/scirobotics.abc5986>
- **核心贡献：** Teacher policy 使用高度图（特权信息）训练，Student policy 仅用 proprioception 通过行为克隆蒸馏；ANYmal 首次在复杂户外地形（楼梯/草地/碎石）实现鲁棒行走
- **对 wiki 的映射：**
  - [privileged-training](../../wiki/concepts/privileged-training.md)
  - [sim2real](../../wiki/concepts/sim2real.md)
  - [locomotion](../../wiki/tasks/locomotion.md)

### 3) Concurrent Training of a Control Policy and a State Estimator（Ji et al., RAL 2022）
- **链接：** <https://arxiv.org/abs/2202.05738>
- **核心贡献：** 并发训练范式：control policy 和 state estimator 同步训练，互相提供梯度；绕开了两阶段 teacher-student 训练的串行瓶颈；在 Unitree A1 实现了更快的收敛和更好的泛化
- **对 wiki 的映射：**
  - [privileged-training](../../wiki/concepts/privileged-training.md)
  - [state-estimation](../../wiki/concepts/state-estimation.md)

### 4) Walk These Ways: Tuning Robot Walking（Margolis et al., CoRL 2022）
- **链接：** <https://arxiv.org/abs/2212.03238>
- **核心贡献：** 用步态参数（命令向量）条件化策略，teacher 用特权地形信息，student 用历史观测；单一策略支持多步态（trot/pace/bound），可实时调节步频/步幅
- **对 wiki 的映射：**
  - [privileged-training](../../wiki/concepts/privileged-training.md)
  - [locomotion](../../wiki/tasks/locomotion.md)

### 5) DreamWaQ: Learning Robust Quadrupedal Locomotion（Nahrendra et al., ICRA 2023）
- **链接：** <https://arxiv.org/abs/2301.10602>
- **核心贡献：** 引入 Dreamer（世界模型）到 privileged training 框架；teacher 的隐状态包含地形感知；student 通过 recurrent 网络估计隐状态；提升样本效率
- **对 wiki 的映射：**
  - [privileged-training](../../wiki/concepts/privileged-training.md)
  - [model-based-rl](../../wiki/concepts/model-based-rl.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
