# Chunk-wise Model-based Policy Optimization（CMPO，arXiv:2610.14564）

> 来源归档（paper）

- **标题：** Chunk-wise Model-based Policy Optimization
- **类型：** paper
- **作者：** Weixi Song（浙江大学 / 上海创新研究院 / 西湖大学）、Donglin Wang（西湖大学，通讯）、Yong-Lu Li（上海创新研究院 / 上海交通大学，通讯）
- **会场：** CoRL 2026（论文首页脚注：10th Conference on Robot Learning, Austin TX, USA）
- **arXiv：** <https://arxiv.org/abs/2610.14564>（v1，2026-10-10，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2610.14564>
- **代码：** 论文正文未给出仓库地址（仅标注 Project Website），本页不写 code 字段
- **入库日期：** 2026-11-12
- **一句话说明：** 在 TD-MPC2 骨架上把 actor 与世界模型都改成「动作块」级，一次前向得到 H 步想象轨迹，用价值展开直接对 actor 求模型梯度，避开循环展开与 BPTT。
- **核对范围：** 读了 arXiv v1 全文 PDF：摘要、第 1 引言、第 2 相关工作、第 3 方法（含式 1 到 8）、第 4 实验（4.1 到 4.3，表 1、表 2，图 2 到 5）、第 5 结论与局限、附录 A 证明、B.1 伪代码、B.2 网络设置、C.1 到 C.4（含表 3 奖励项）；参考文献只扫了列表未逐篇核对；图 7 的 19 条逐任务曲线只看到图例与标题，未逐条读数；项目页与 OpenReview 评审意见未读（环境内不可达或论文未给链接）。
- **通读：** 已完成

## 开源状态

- 论文摘要处仅有 Project Website 链接文字，PDF 文本中未出现 GitHub 仓库地址；开源状态未核实。

## 核心摘录

1. **问题（第 1、3.2 节）：** 循环展开 H 步需 O(H) 次串行模型调用；单步误差界为 ε、动力学 L_f-Lipschitz 时，展开误差上界为 ε·Σ L_f^i（式 2，附录 A 命题 1 给出证明）；对 actor 的梯度含 Jacobian 连乘（式 3、4），即 BPTT 的爆炸或消失风险。
2. **方法（第 3.3 节）：** 编码器 z=h(s)；块动力学 z_{t+1:t+H}=f(z_t, a_{t:t+H-1})；奖励、终端价值 Q、块策略 π(z_t)=a_{t:t+H-1}；在 TD-MPC2 上搭建，其余设计不变。
3. **损失（式 6 到 8）：** 模型损失用 stop-gradient 的潜在一致项加奖励两-hot 交叉熵；critic 用单步动作 Q 与 1 步 TD；actor 用 (H-1) 步价值展开加 SAC 式熵正则，避免在 z_{t+H} 处再查询 actor。推理时只执行块里的第一个动作。
4. **HumanoidBench（第 4.1、C.1 节）：** 19 个任务，Unitree H1hand，动作维度 61；基线为 PPO、SAC、DreamerV3、TD-MPC2（直接采用 HumanoidBench 原报告结果，C.2）与自实现的 MAAC 式更新（建在 TD-MPC2 架构上）；3 个随机种子，阴影为 95% 置信区间（图 2）。
5. **耗时（第 4.1 节，表 1）：** 单张 RTX 4090，Walk 与 Run 上 200k 训练步 TD-MPC2 3.57 与 3.54 小时对 CMPO 1.74 与 1.72 小时；单次推理 40.8 与 39.5 ms 对 1.4 与 1.3 ms。
6. **真机（第 4.2、C.3 节）：** Unitree Go2 双足站立行走，Isaac Gym 训练，12 个关节残差位置指令加 PD；对比 SAC 而非 TD-MPC2；保持双足姿态 20 s：CMPO 7/10、SAC 4/10。
7. **分析（第 4.3 节）：** MAAC 梯度范数振荡更大（图 4）；horizon 2 与 4 最好，5 退化，主实验统一用 3（图 5a）；表 2：w/o CM（块策略加单步模型）Walk 22.2、Run 9.6、Stair 37.1、Slide 36.1，对 CMPO 954.1、629.2、60.9、265.8；加 MPPI（w MPC）为 931.1、864.5、71.9、391.2。
8. **作者自认局限（结论后 Limitation）：** 架构大多继承 TD-MPC2；只执行块内第一个动作，其余被丢弃；动态块长在 RL 里仍少有探索。

## 对 wiki 的映射

- [paper-cmpo-chunk-wise-model-based-policy-optimization](../../wiki/entities/paper-cmpo-chunk-wise-model-based-policy-optimization.md)
