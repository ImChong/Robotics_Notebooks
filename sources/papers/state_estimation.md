# state_estimation

> 来源归档（ingest）

- **标题：** 机器人状态估计核心论文
- **类型：** paper
- **来源：** IROS / ICRA / IEEE TRO / RA-L
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 EKF 足式机器人里程计、不变 EKF（InEKF）、基于学习的状态估计等方法

## 核心论文摘录

### 1) State Estimation for Legged Robots on Unstable and Slippery Terrain（Bloesch et al., IROS 2013）
- **链接：** <https://ieeexplore.ieee.org/document/6696868>
- **核心贡献：** RSL（ETH）里程碑工作：基于 EKF 融合 IMU + 运动学 + 接触检测，实现 ANYmal 类四足机器人在复杂地形的基座位姿估计；提出接触概率加权以处理打滑
- **对 wiki 的映射：**
  - [state-estimation](../../wiki/concepts/state-estimation.md)

### 2) Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation（Hartley et al., IJRR 2020）
- **链接：** <https://arxiv.org/abs/1904.09251>
- **核心贡献：** 将不变 EKF（InEKF）应用于足式机器人：利用 SE_2(3) 李群的不变性简化线性化误差项；相比标准 EKF 在大旋转扰动下一致性更好；Cassie 双足机器人实验验证
- **对 wiki 的映射：**
  - [state-estimation](../../wiki/concepts/state-estimation.md)

### 3) Legged Robot State Estimation（Teng et al., ICRA 2021）
- **链接：** <https://arxiv.org/abs/2101.02630>
- **核心贡献：** 针对双足机器人（Cassie）的专用状态估计：在 InEKF 框架上处理间歇性接触；引入 IMU 预积分减少计算量；为后续 RL 策略的 observation 设计提供基准
- **对 wiki 的映射：**
  - [state-estimation](../../wiki/concepts/state-estimation.md)

### 4) Learning-Based Inertial Odometry（Chen et al., RAL 2021）
- **链接：** <https://arxiv.org/abs/2103.05560>
- **核心贡献：** 用 TCN 网络直接从 IMU 数据学习速度估计，绕开传统 EKF 的手工噪声建模；在足式机器人打滑场景下鲁棒性显著优于 EKF；展示学习与滤波混合的可行性
- **对 wiki 的映射：**
  - [state-estimation](../../wiki/concepts/state-estimation.md)

### 5) Proprioceptive State Estimation for Quadruped Robots（Nobili et al., ICRA 2017）
- **链接：** <https://ieeexplore.ieee.org/document/7989307>
- **核心贡献：** 仅用本体感知（IMU + 关节编码器）实现无外感知状态估计；EKF 框架中整合腿部里程计；为 whole-body control 提供轻量级基座估计
- **对 wiki 的映射：**
  - [state-estimation](../../wiki/concepts/state-estimation.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
