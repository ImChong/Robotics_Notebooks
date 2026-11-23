# RAIBO-Wagon：四足机器人牵引拖车的实时导航（arXiv:2610.39267）

> 来源归档（paper）：2026-10-29 发布的 arXiv v1 预印本，本文以**官方 HTML 全文**（含 §I–VII 与附录 A、B）为原始资料，不转存大体积 PDF。

- **标题：** RAIBO-Wagon: Real-Time Navigation System for a Legged Robot Towing a Wagon
- **中文标题：** RAIBO-Wagon：四足机器人牵引拖车的实时导航系统
- **作者：** Insung Yang、Eunchae Kim、Jungwoo Hur、Seunghun Jeon、Donghyuk Choi、Jemin Hwangbo
- **机构：** 韩国科学技术院（KAIST）Robotics and Artificial Intelligence Lab
- **论文版本：** [arXiv:2610.39267v1](https://arxiv.org/abs/2610.39267v1)，2026-10-29，预印本
- **原文全文：** https://arxiv.org/html/2610.39267v1
- **PDF：** https://arxiv.org/pdf/2610.39267
- **官方项目：** https://insungyang.github.io/raibo-wagon/
- **归档日期：** 2026-11-02；本轮深读核对：2026-11-23
- **核对范围：** 官方 arXiv HTML v1 全文：§I–III（问题、相关工作、MPPI/CFM 定义）、§IV-A–E（运动控制、离线专家、FDM、CFM 与 MPPI）、§V（建图与周期）、§VI（真机实验、对照、三项消融）、§VII（局限）、附录 A/B（控制与 FDM 验证）、Algorithm 1/2、Table I–V；与官方项目页逐项交叉核对。未在本地复现训练或部署。
- **通读：** 已完成

## 为什么要规划“拖车轮轨迹”

拖车是**欠驱动**负载，车轮主要沿其朝向滚动；腿式本体可横移、转向，却不能保证后方拖车沿本体几何路径通过窄道。只给四足机器人做圆形碰撞体的 A*，会忽视拖车摆角/扫掠空间；把机器人加拖车合成一个大圆，则过度保守。论文规划的是**机器人—拖车耦合状态下的速度指令序列**，不是分别直接控制每个车轮。

## 方法核对（正文 §IV，Fig.2，Algorithm 1–2）

1. **拖车感知的离线路径专家（§IV-B）：** 先生成仅按本体避障的 A* 路径，再以节点状态 `(x,y,θ_robot,θ_wagon)`、局部高度图和 CNN+MLP 学习风险预测器优化整车碰撞风险与路径平滑代价；采用有限差分梯度、25 次优化。该高成本模块**不在在线导航执行**。
2. **离线专家跟踪器（§IV-C）：** MPPI 使用学习的 Forward Dynamics Model（FDM）评估候选指令，步长 6、候选 1024、迭代 25。FDM 输入 133 维导航观测及速度序列，通过 state encoder、command encoder、GRU、轨迹/碰撞两个 decoder，预测本体位置与航向增量、机器人—拖车相对角增量及每步碰撞 logit；拖车位姿由本体位姿与相对角推导。预测未来的**拖车扫掠与碰撞**，不是使用一条独立车轮目标曲线。
3. **CFM 蒸馏（§IV-D）：** RaiSim 中运行离线专家 25,000 episode，约 250,000 组 `(观测 o, A* 的 3 m 前视 waypoint, 6 步专家速度序列)`，每 1.5 秒记录一组。CFM 条件变量 `[o; waypoint]`，训练时 CFG 条件丢弃率 0.1；由离线专家结果学习对局部路点有条件的多样指令分布。
4. **在线热启动与精修（§IV-E）：** CFM 通过 10 步 Euler ODE、CFG 权重 3，输出 256 条三秒指令候选；第一次 MPPI 用这些样本直接初始化，其后做 10 轮衰减相关高斯扰动 + FDM 推演 + softmax 加权更新。在线奖励为 `exp(-L_way) + exp(-L_coll/15)`（Eq.22），其中拖车通过预测碰撞项参与，而不是另设在线拖车参考轨迹。
5. **分层导航与底层控制（§IV-A、§V）：** MID-360 LiDAR → OctoMap（三维占据图 10 cm / 10 Hz）和 FAST-LIO2 → 2.5D grid A*（2 Hz）→ 3 m 前视路点 → CFM+MPPI（2 Hz，约 0.15+0.15=0.3 秒/次）→ 第 1 个 SE(2) 速度指令 → wagon-aware PPO-GRU+MLP 底层行走（100 Hz）。后车通过带编码器的三自由度被动连接测相对 yaw；低层策略不直接输入拖车状态，训练时随机化拖车惯量。
6. **为什么不平均 CFM 多峰样本（§IV-E）：** 多条绕障路径直接求均值可能穿过障碍。先由 A* 的 3 m 前视 waypoint 限定可通行走廊，再在相对单峰的候选族里做 MPPI 精修；路径封堵时全局 A* 更新路点，局部采样随之切换走廊。

## 关键实验（§VI，官方项目页）

| 条件 | 作者报告 | 口径 |
|---|---:|---|
| 真机八类室外 15 m 路线（6 静态 + 2 行人动态） | 34/40 = **85%** | 每类 5 次，无事先地图，全板载 |
| 最窄场景 Terrain F | 本方法 5/5；Hybrid A* 0/5；waypoint RL 0/5 | 每种 5 次；特定基线与设定，不代表普适性能 |
| Terrain F 附加载重 | +5 kg：5/5；+10 kg：5/5；+15 kg：5/5 | 拖车自重约 25 kg，最高拖曳总重约 40 kg |
| 仿真 hard Slalom | CFM+MPPI 96.5% vs vanilla MPPI 45.5% | 同采样数与迭代数；Table V |
| 仿真 hard Corridor | CFM+MPPI 72.5% vs vanilla MPPI 29.5% | 同采样数与迭代数；Table V |

## 失败分析与边界（§VI-A2、§VII、附录）

- 真机 **6 次失败**中，3 次由后方 LiDAR 掩蔽导致地图漏障：急转弯时拖车摆入侧方视野，掩蔽同时遮去真实障碍，发生明确碰撞；另 3 次来自局部规划：MID-360 非重复扫描单帧有过期/缺失回波，训练用完整瞬时二维扫描，Sim2Real 感知落差使拖车擦碰障碍。
- 系统暂不适于需要**前后反复腾挪/倒车对齐**的狭窄场景；未明确评估不同地面可通行性（腿能走但拖车轮未必能走）；CFM/FDM 针对一款定制拖车训练，跨拖车几何泛化有待研究。
- 附录 B：全动力学 RaiSim 在机载运行约需 5 分钟才能生成 0.5 s 指令（所述配置）；学习 FDM 与简化运动学模型的单条预测时延相近，而任务成功表现更稳健。不可把它读作官方已发布可复现实时代码。

## 官方资源与开放结论

- [官方项目页归档](../sites/raibo-wagon.md)
- 项目页可浏览视频与结果，**截至 2026-11-23 页面未列官方 GitHub 源码、训练数据或权重下载地址**。不能凭论文描述的系统模块视为已开源，故本轮不做“源码运行时序图”臆测。
- [原 wiki 详情实体](../../wiki/entities/paper-raibo-wagon-legged-towing-navigation.md)；[导航纵深路线](../../roadmap/depth-navigation.md)
