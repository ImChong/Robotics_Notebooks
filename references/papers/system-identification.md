# System Identification

聚焦机器人动力学辨识、执行器建模与在线参数估计论文。

## 关注问题

- 如何精确获取连杆质量与惯性参数？
- 如何建立高保真的电机/驱动器动力学模型？
- 如何在运动过程中实时辨识地形与负载？
- 如何设计最优激励轨迹以提升辨识精度？

## 代表性论文

### 离线动力学辨识

- ** Nguyen & Park (IJRR 2011)** — *Physically Consistent State Estimation and System Identification for Contacts*. 解决了惯性参数辨识中的物理一致性约束。
- ** Gautier & Khalil (1992)** — *Exciting Trajectories for Identification*. 提出了如何设计激励轨迹来优化辨识矩阵的条件数。

### 执行器建模 (Actuator Dynamics)

- ** Hwangbo et al. (Science Robotics 2019)** — *Learning Agile and Dynamic Motor Skills for Legged Robots*. 提出了 **ActuatorNet**，用神经网络拟合驱动器特性，显著提升 sim2real 成功率。

### 在线系统辨识

- ** Grandia et al. (IROS 2019)** — *Online System Identification for Legged Robots*. 在 ANYmal 上实现了地面刚度与摩擦的实时辨识。

## 关联页面

- [System Identification (Concept)](../../wiki/concepts/system-identification.md)
- [Sim2Real (Concept)](../../wiki/concepts/sim2real.md)
- [Actuator Network (Method)](../../wiki/methods/actuator-network.md)
- [Model Predictive Control (Method)](../../wiki/methods/model-predictive-control.md)
