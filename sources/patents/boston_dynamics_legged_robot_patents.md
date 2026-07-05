# Boston Dynamics 足式机器人控制与硬件专利摘录

> 专利来源归档（ingest）

- **类型：** patent / quadruped / locomotion / actuator / trajectory-optimization
- **机构：** Boston Dynamics Inc.
- **入库日期：** 2026-07-05
- **一句话说明：** 五件与足式机器人 **在线轨迹优化、非线性规划、关节级电机—控制器一体化、机内数据路由、液压执行器** 相关的授权/申请专利，勾勒 Spot/Atlas 类平台的 **模型驱动运控 + 高集成执行器 + 分布式计算** 工程栈。

## 核心专利摘录（面向 wiki 编译）

### 1) Robot movement and online trajectory optimization（US11833680B2, 2023）

- **链接：** <https://patents.google.com/patent/US11833680B2/en> · PDF：<https://patentimages.storage.googleapis.com/a3/11/0d/6857f78699a46b/US11833680.pdf>
- **发明人：** Yeuhi Abe, Scott Kuindersma, Matthew P. Kelly, Twan Koolen, Benjamin Stephens, Robin Deits 等
- **核心主张：** 机载计算系统接收 **导航目标** 与 **运动学状态**，结合 **轨迹目标** 在线 **重定向（retarget）** 轨迹；再分解为与质心轨迹一致的 **centroidal trajectory** 与 **kinematic trajectory**，供运动控制模块执行——体现 **分层：导航意图 → 质心/全身轨迹 → 关节跟踪** 的实时闭环。
- **对 wiki 的映射：**
  - [`wiki/entities/patent-boston-dynamics-legged-control-stack.md`](../../wiki/entities/patent-boston-dynamics-legged-control-stack.md)
  - [`wiki/entities/boston-dynamics.md`](../../wiki/entities/boston-dynamics.md)
  - [`wiki/methods/model-predictive-control.md`](../../wiki/methods/model-predictive-control.md)

### 2) Nonlinear trajectory optimization for robotic devices（US12168300B2, 2024）

- **链接：** <https://patents.google.com/patent/US12168300B2/en> · PDF：<https://patentimages.storage.googleapis.com/9c/94/68/18b5059ffcd158/US12168300.pdf>
- **发明人：** Carmine D. Bellicoso, Neil M. Neville, Alexander D. Perkins, Logan W. Tutt 等
- **核心主张：** 接收机器人 **初始状态** 与 **目标状态**，用 **非线性优化** 生成 **候选轨迹**；判定可行性后下发运动控制模块，不可行则迭代修正——与 US11833680 的 **在线重定向** 形成 **规划层 + 执行层** 互补。
- **对 wiki 的映射：**
  - [`wiki/entities/patent-boston-dynamics-legged-control-stack.md`](../../wiki/entities/patent-boston-dynamics-legged-control-stack.md)
  - [`wiki/concepts/whole-body-control.md`](../../wiki/concepts/whole-body-control.md)

### 3) Motor and controller integration for a legged robot（US10525601B2, 2020）

- **链接：** <https://patents.google.com/patent/US10525601B2/en>（PDF 见 Google Patents 下载；液压执行器同族另见 US8126592）
- **发明人：** Adam Young, Zachary John Jackowski, Kyle Rogers（申请阶段曾属 Google Inc.）
- **核心主张：** 关节壳体内集成 **电机、面向电机的 FET 阵列 PCB、转子轴延伸上的磁编码器**——把 **驱动、功率级、位置传感** 压进关节模组，支撑四足高频力矩环与紧凑机身。
- **对 wiki 的映射：**
  - [`wiki/entities/patent-boston-dynamics-legged-control-stack.md`](../../wiki/entities/patent-boston-dynamics-legged-control-stack.md)
  - [`wiki/entities/boston-dynamics.md`](../../wiki/entities/boston-dynamics.md)

### 4) Data transfer assemblies for robotic devices（US20240184731A1, 2024）

- **链接：** <https://patents.google.com/patent/US20240184731A1/en>（用户链接 US2024184731A1 为笔误/截断，正确公开号为 US20240184731A1）
- **发明人：** Matthew Meduna, Devin Billings
- **核心主张：** 用 **交换设备** 在 **第一/第二主机处理器** 与 **第一/第二电子设备** 之间路由 **数据包**——反映 Spot 类机器人 **多计算单元 + 传感器/执行器总线** 的机内网络拓扑。
- **对 wiki 的映射：**
  - [`wiki/entities/patent-boston-dynamics-legged-control-stack.md`](../../wiki/entities/patent-boston-dynamics-legged-control-stack.md)

### 5) Actuator system（US8126592B2, 2012）

- **链接：** <https://patents.google.com/patent/US8126592B2/en> · PDF：<https://patentimages.storage.googleapis.com/04/57/c5/611d0ebefde7e0/US8126592.pdf>
- **发明人：** Marc Raibert, Aaron Saunders
- **核心主张：** 机器人/仿生连杆关节上的 **双活塞缸液压执行器**、负载传感与 **电控液压阀** 闭环——代表 BigDog/早期液压 Atlas 时代的 **高功率密度液压执行器** 专利基线；与全电 Spot 的 US10525601 形成 **液压 → 全电** 技术代际对照。
- **对 wiki 的映射：**
  - [`wiki/entities/patent-boston-dynamics-legged-control-stack.md`](../../wiki/entities/patent-boston-dynamics-legged-control-stack.md)
  - [`wiki/entities/boston-dynamics.md`](../../wiki/entities/boston-dynamics.md)

## 当前提炼状态

- [x] 五件专利要点摘录与 wiki 映射
- [x] US2024184731 → US20240184731 公开号校正
- [ ] 与 [`wiki/entities/paper-spot-rl-distributional-sim2real.md`](../../wiki/entities/paper-spot-rl-distributional-sim2real.md) 的 RL 低层 API 叙事交叉维护
