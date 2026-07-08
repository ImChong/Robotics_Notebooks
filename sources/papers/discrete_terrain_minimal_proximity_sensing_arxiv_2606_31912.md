# Learning Locomotion on Discrete Terrain via Minimal Proximity Sensing（arXiv:2606.31912）

> 论文来源归档（ingest）

- **标题：** Learning Locomotion on Discrete Terrain via Minimal Proximity Sensing
- **类型：** paper / quadruped / perceptive-locomotion / minimal-sensing / reinforcement-learning / sim2real
- **arXiv：** <https://arxiv.org/abs/2606.31912> · PDF：<https://arxiv.org/pdf/2606.31912.pdf>
- **项目页：** <https://sites.google.com/view/foot-tof/home>
- **视频：** <https://www.youtube.com/watch?v=K7_acxnVrdI>
- **机构：** ETH Zurich, Robotic Systems Lab（Jiale Fan, Connor Flynn, Tianao Xu, Junzhe He, Andrei Cramariuc, Marco Hutter, Robert Baines）
- **平台：** ANYmal-D（ANYbotics）
- **入库日期：** 2026-07-08
- **一句话说明：** 在四足 **足底** 集成 **低成本红外 ToF 接近传感器**（VL53L5CX，4×4 网格、60 Hz），把 **接触前（pre-contact）** 局部几何反馈直接喂给 **LSTM+PPO** 策略，在 **Isaac Gym** 两阶段课程上训出 **统一离散地形策略**，**无需相机/LiDAR/高程图管线**；实机验证踏石、碎石、平衡木与 **60 cm 沟** 穿越，平均约 **0.52 m/s**，相对从机载深度/理想高程图 **投影** 的基线在关节偏置、连杆误差与里程计噪声下更稳健。

## 核心摘录（面向 wiki 编译）

### 1) 动机：全局感知 vs 本体觉的缺口

- **要点：** 腿足 locomotion 的落脚、触地时机与冲击调制常依赖 **延迟或推断** 的接触信息；LiDAR/RGB-D 提供全局上下文但易受 **自遮挡、视场、同步与重建延迟** 影响；纯本体觉 **触地后才反应**，对 **沟、台阶边缘、垫脚石** 等离散地形不足。**接触前、足端局部、低算力** 的感知是本文切入点。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-discrete-terrain-minimal-proximity-sensing.md`](../../wiki/entities/paper-discrete-terrain-minimal-proximity-sensing.md)
  - [`wiki/concepts/terrain-adaptation.md`](../../wiki/concepts/terrain-adaptation.md)

### 2) 足底 ToF 模组与噪声标定

- **要点：** ANYmal-D 橡胶足内嵌 **STMicro VL53L5CX**，45°×45° FoV、**4×4** 分辨率、**60 Hz**（与运动控制同频）；光轴相对矢状面外倾约 **15°** 以覆盖摆腿期预期落脚区；USB 中继延迟约 **20–60 ms**；>1 m 读数裁剪。标定不同距离/入射角下的 **噪声与缺失率**，用于仿真 **静态偏置、比例/Gauss 噪声、随机缺失、时延** 等域随机化。
- **对 wiki 的映射：** 同上实体页；[`wiki/entities/anymal.md`](../../wiki/entities/anymal.md)

### 3) 训练：无地图栈的统一 LSTM 策略

- **要点：** **不依赖** 独立高程图生成/注意力地图处理；观测 = **四足 proximity 读数** + 关节位姿、重力、体线/角速度；仿真用足端 **ray caster** 模拟 4×4 网格（每格 5 条射线插值最近点）。**Stage 1** 以踏石为主、课程增大间隙/高度差；**Stage 2** 提高难度并加入楼梯、平衡木、坡面等，**Stage 2** 加大感知退化随机化。算法：**PPO + LSTM**，环境 **Isaac Gym**。
- **对 wiki 的映射：**
  - [`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)
  - [`wiki/entities/legged-gym.md`](../../wiki/entities/legged-gym.md)

### 4) 仿真鲁棒性：足端直连 vs 机身高程投影

- **要点：** 对照 **Mock (Ideal Height Scan)**（5 m×5 m、5 cm 分辨率无噪俯视射线）与 **Mock (Aggre. Elevation Map)**（六路机身深度 + 里程计融合高程图）在足端 **运动学投影** 得到类 proximity 读数。在 **静态关节偏置、小腿长度 ±20%、里程计噪声 0–400%** 下，足端传感器成功率下降更慢；机载投影因 **足位估计误差** 与 **地图漂移** 更快崩溃。
- **对 wiki 的映射：** 同上实体页；[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)

### 5) 实机离散地形

- **要点：** 仅足底 proximity：**60 cm** 水平沟；20×20 cm 踏石（含错落高度与支撑砖层）；20 cm 宽平衡木。策略展现 **主动扫描**（摆腿期读数排序变化推断侧壁/上表面）、**触地前检测近似水平面** 等行为；平均速度约 **0.52 m/s**。
- **对 wiki 的映射：**
  - [`wiki/entities/quadruped-robot.md`](../../wiki/entities/quadruped-robot.md)
  - [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)

## 局限（论文自述）

- ToF 对 **强吸收/镜面** 地面不可靠；足端开孔易 **泥污堵塞**；传感器密度与布置仍可优化。

## 当前提炼状态

- [x] 摘要与主方法摘录
- [x] wiki 页面映射确认
