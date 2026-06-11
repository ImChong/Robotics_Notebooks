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
- **链接：** <https://arxiv.org/abs/2107.04034> · 项目页 <https://ashish-kmr.github.io/rma-legged-robots/> · 代码 <https://github.com/antonilo/rl_locomotion>
- **核心贡献：** 两阶段训练：Phase 1 用特权 $e_t$ 编码为 extrinsics $z_t$ 训练 base policy $\pi$ + encoder $\mu$；Phase 2 用 **on-policy** 历史 $(x,a)$ 监督 adaptation module $\phi$ 预测 $\hat{z}_t$；A1 **异步 10/100 Hz** 零微调部署，<1 s 适应变地形/载荷
- **对 wiki 的映射：**
  - [paper-rma-rapid-motor-adaptation](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)（本次升格实体页 + Mermaid）
  - [privileged-training](../../wiki/concepts/privileged-training.md)
  - [sim2real](../../wiki/concepts/sim2real.md)
- **一手归档：** [rma_arxiv_2107_04034.md](rma_arxiv_2107_04034.md)、[rma-legged-robots 项目页](../sites/rma-legged-robots-github-io.md)、[antonilo/rl_locomotion](../repos/antonilo_rl_locomotion.md)

### 2) Learning to Walk in Difficult Terrain（Lee et al., Science Robotics 2020）
- **链接：** <https://www.science.org/doi/10.1126/scirobotics.abc5986>
- **核心贡献：** Teacher policy 使用高度图（特权信息）训练，Student policy 仅用 proprioception 通过行为克隆蒸馏；ANYmal 首次在复杂户外地形（楼梯/草地/碎石）实现鲁棒行走
- **对 wiki 的映射：**
  - [privileged-training](../../wiki/concepts/privileged-training.md)
  - [sim2real](../../wiki/concepts/sim2real.md)
  - [locomotion](../../wiki/tasks/locomotion.md)
  - [curriculum-learning](../../wiki/concepts/curriculum-learning.md)

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
  - [reward-design](../../wiki/concepts/reward-design.md)
  - [gait-generation](../../wiki/concepts/gait-generation.md)

### 5) DreamWaQ: Learning Robust Quadrupedal Locomotion（Nahrendra et al., ICRA 2023）
- **链接：** <https://arxiv.org/abs/2301.10602>
- **核心贡献：** **CENet** 上下文估计 + 隐式地形想象；盲走（仅本体）单阶段非对称 AC；为 DreamWaQ++ 奠定基础
- **对 wiki 的映射：**
  - [privileged-training](../../wiki/concepts/privileged-training.md)
  - [dreamwaq-plus](../../wiki/entities/dreamwaq-plus.md)（T-RO 2026 多模态扩展）

### 6) DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion（Nahrendra et al., IEEE T-RO 2026）
- **链接：** <https://arxiv.org/abs/2409.19709> · 项目页 <https://dreamwaqpp.github.io/>
- **核心贡献：** 融合 **3D 点云外感知** 与 **本体 MLP-Mixer/CENet**；分层 $SE(3)$ 外感知记忆 + PointNet 置信滤波 + 多模态 Mixer；对比/VAE/versatility 辅助损失；楼梯/陡坡/OOD 与多传感器平台验证
- **对 wiki 的映射：**
  - [dreamwaq-plus](../../wiki/entities/dreamwaq-plus.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] RMA 独立 ingest 与实体页；关联 wiki 参考来源已链至 `rma_arxiv_2107_04034.md`
