# system_identification

> 来源归档（ingest）

- **标题：** 机器人系统辨识核心论文
- **类型：** paper
- **来源：** IJRR / ICRA / Science Robotics
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖惯性参数辨识、激励轨迹设计、神经网络执行器模型等系统辨识方法

## 核心论文摘录

### 1) Physically Consistent State Estimation and System Identification for Contacts（Nguyen & Park, IJRR 2011）
- **链接：** <https://journals.sagepub.com/doi/10.1177/0278364911426458>
- **核心贡献：** 提出约束最小二乘框架，同时辨识机器人惯性参数（质量/质心/惯性张量）并满足物理一致性约束（正半定惯性矩阵、质心在物理范围内）；比标准 LS 辨识参数更可用于控制器设计
- **对 wiki 的映射：**
  - [system-identification](../../wiki/concepts/system-identification.md)

### 2) Exciting Trajectories for Identification（Gautier & Khalil, 1992 / 2013 更新）
- **链接：** <https://ieeexplore.ieee.org/document/163864>
- **核心贡献：** 提出最优激励轨迹设计：通过最大化观测矩阵条件数（或最小化参数估计方差）来设计持续激励轨迹；成为机器人标定领域的标准方法
- **对 wiki 的映射：**
  - [system-identification](../../wiki/concepts/system-identification.md)

### 3) Learning Agile and Dynamic Motor Skills for Legged Robots（Hwangbo et al., Science Robotics 2019）
- **链接：** <https://www.science.org/doi/10.1126/scirobotics.aau5872>
- **核心贡献：** 提出 ActuatorNet：用神经网络从历史关节误差（位置/速度/力矩序列）预测真实关节力矩；通过真实数据辨识执行器动力学模型，显著减少 sim2real gap
- **对 wiki 的映射：**
  - [actuator-network](../../wiki/methods/actuator-network.md)
  - [system-identification](../../wiki/concepts/system-identification.md)
  - [sim2real](../../wiki/concepts/sim2real.md)

### 4) Online System Identification for Legged Robots（Grandia et al., IROS 2019）
- **链接：** <https://ieeexplore.ieee.org/document/8967800>
- **核心贡献：** 在线辨识地面刚度和摩擦系数；与 MPC 耦合实现自适应步态规划；在 ANYmal 上验证了对未知地形（软地/冰面）的自适应能力
- **对 wiki 的映射：**
  - [system-identification](../../wiki/concepts/system-identification.md)
  - [model-predictive-control](../../wiki/methods/model-predictive-control.md)

### 5) Sim-to-Real Transfer of Robotic Control with Dynamics Randomization（Peng et al., ICRA 2018）
- **链接：** <https://arxiv.org/abs/1710.06537>
- **核心贡献：** 域随机化作为隐式系统辨识：通过随机化质量/摩擦/驱动增益覆盖真实参数分布；策略学习对参数变化鲁棒；比精确辨识更实用（无需测量真实参数）
- **对 wiki 的映射：**
  - [system-identification](../../wiki/concepts/system-identification.md)
  - [sim2real](../../wiki/concepts/sim2real.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
