# Handroid: Bridging Dexterous Hand and Humanoid

> 来源归档（ingest）

- **标题：** Handroid: Bridging Dexterous Hand and Humanoid
- **类型：** paper（arXiv 预印本）
- **机构：** University of North Carolina at Chapel Hill；Stanford University
- **原始链接：**
  - <https://arxiv.org/abs/2607.16187>
  - <https://arxiv.org/html/2607.16187>（HTML 版便于锚点跳转）
  - 项目页：<https://handroid.org/>
  - CAD：<https://cad.onshape.com/documents/d3de21915f3c9cacc1887cf3/w/dc7c7b68235fdbb205f27505/e/8673167885a4e16e9b6c2791>
  - BOM：<https://docs.google.com/spreadsheets/d/1ml2pJ9iSiDhcNiEPnRkoHqwzarEjfeZ8KSDNHGFpFh4/edit?usp=sharing>
- **入库日期：** 2026-07-20
- **一句话说明：** UNC × Stanford 提出 **桌面级双形态机器人 Handroid**：**27-DoF** 机电一体机体可在 **20-DoF 仿人灵巧手** 与 **含 12-DoF 下肢的桌面人形** 间滑轨重配置；统一控制与学习栈覆盖 **Vision Pro 遥操作**、**Diffusion Policy 抓取**、**RL 掌内重定向**、**ZMP 步态 + RL 跟踪/速度控制** 与 **跨形态长时程 loco-manipulation**；机械 **CAD + BOM 已公开**，控制代码截至入库日 **尚未链出**。

## 核心论文摘录（MVP）

### 1) 动机：灵巧手与人形通常割裂；能否复用同一形态学？

- **链接：** <https://arxiv.org/html/2607.16187#S1>
- **摘录要点：** 灵巧手擅长接触丰富操作，人形擅长全身移动与人体尺度交互，但二者 **常作为独立平台开发**——手多装于固定基座臂，人形手往往欠驱动。论文追问：**形态能否跨具身复用**，而非绑定单一机器人。
- **对 wiki 的映射：**
  - [Handroid](../../wiki/entities/handroid.md) — 双形态设计动机与定位。
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 移动与操作耦合的研究语境。

### 2) 结构：27-DoF 模块映射 + 双滑动机构切换形态

- **链接：** <https://arxiv.org/html/2607.16187#S3>
- **摘录要点：**
  - **手形态：** 五指各 1 外展/内收 + 3 屈伸 ≈ **20 主动 DoF**；拇指对掌与多指协调。
  - **人形态：** 4-DoF 头、双臂各 4-DoF、双腿各 6-DoF、1-DoF 髋；**关节 9、26** 为齿条–齿轮棱柱机构，将模块 II/V 滑移以完成手↔人形映射。
  - 尺寸 **0.33 m / 2.05 kg**；**全 3D 打印**、模块化、桌面可复现。
- **对 wiki 的映射：**
  - [Handroid](../../wiki/entities/handroid.md) — 模块–关节编号与形态对照。

### 3) 电气：ESP32-S3 + Dynamixel + 分布式 IMU

- **链接：** <https://arxiv.org/html/2607.16187#S3.SS2>
- **摘录要点：** **40×80 mm** 堆叠主板；**TTL 总线** 控全部 Dynamixel；**Wi-Fi** 向主机推状态；可 **板载执行基本运动原语** 或 **手柄遥控**；**PD 140 W** 外供或电池；指尖/足端 IMU + 躯干 IMU。
- **对 wiki 的映射：**
  - [Handroid](../../wiki/entities/handroid.md) — 工程实践与部署接口。

### 4) 手形态学习：Vision Pro 遥操作 + Diffusion Policy + RL 掌内操作

- **链接：** <https://arxiv.org/html/2607.16187#S4.SS1>
- **摘录要点：**
  - **遥操作：** Apple Vision Pro 手部重定向 + 腕部映射 **Franka Research 3** 末端，统一臂–手采集。
  - **抓取：** 约 **100** 条示范训练 **object-conditioned Diffusion Policy**；10 类物体真机成功率约 **60–90%**（Table 1）。
  - **掌内重定向：** 仿真 **PPO** 训练，真机 **30 Hz** 维持 3D 打印立方体朝向。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md) — 开源/可复现灵巧操作管线范例。
  - [RUKA-v2 Hand](../../wiki/entities/ruka-v2-hand.md) — 另一全栈开源灵巧手对照。

### 5) 人形态学习：ZMP 参考 + RL 跟踪、无参考速度 RL、关键帧编辑

- **链接：** <https://arxiv.org/html/2607.16187#S4.SS2>
- **摘录要点：**
  - **参考跟踪：** ZMP 规划 + LIPM + LQR preview → **Mink IK** 生成关节参考 → **MuJoCo PPO** 闭环跟踪（式 1 多分量奖励）。
  - **速度控制：** 无离线参考，平面速度 + 偏航率条件 **PPO**（式 2）。
  - **关键帧：** **Viser Keyframe Editor** 关节空间插值，可直控或导出为 RL 参考。
  - 真机验证：行走、转向、蹲起、俯卧撑、引体向上、pick-and-place。
- **对 wiki 的映射：**
  - [Handroid](../../wiki/entities/handroid.md) — 人形控制栈总览。
  - [ToddlerBot（161 索引）](../../wiki/entities/paper-loco-manip-161-141-toddlerbot.md) — 同类桌面/小型人形 ML 平台对照。

### 6) 长时程跨形态任务与开源声明

- **链接：** <https://arxiv.org/html/2607.16187#S5>
- **摘录要点：** 实验链：**形态切换 → 人形行走与物体交互 → 与 Franka 重新对接 → 灵巧 pick-and-place**。贡献之一写明 **design and open-source Handroid**；截至入库日项目页 **Code 未链出**，**CAD/BOM 已公开**。
- **对 wiki 的映射：**
  - [Handroid](../../wiki/entities/handroid.md) — 开源状态与局限。
  - [开源人形硬件对比](../../wiki/entities/open-source-humanoid-hardware.md) — 可后续补一行 Handroid 条目。
