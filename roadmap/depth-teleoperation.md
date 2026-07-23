# 路线（纵深）：如果目标是遥操作（人形全身遥操作 + 手指遥操作 → 示范数据/实时接管）

**摘要**：面向"想搭一套让人实时操控人形机器人、并顺带采集高质量示范数据"的纵深路线，从遥操作的问题定义与输出形态定位、输入接口与硬件通道选型、设备信号到机器人指令的映射与延迟/双边稳定，到**人形全身遥操作**（三点/全身追踪 → 全身跟踪策略 → WBC/流形约束低层）与**手指/灵巧手遥操作**（手部姿态估计 → 手部重定向 → 接触保持与触觉），再到下游模仿学习闭环与极端场景进阶，按 Stage 0–5 串通核心方法；本路线是 [运动控制主路线](motion-control.md) 的一条分支，向上承接 [动作重定向纵深](depth-motion-retargeting.md) 的重定向工具箱，向下供给 [模仿学习纵深](depth-imitation-learning.md) 与 [BFM 纵深](depth-bfm.md) 的训练数据。

## 路线一览

```mermaid
flowchart LR
  S0["**Stage 0**<br/>全景与定位<br/><em>问题定义 / 输出形态</em>"]
  S1["**Stage 1**<br/>输入接口与硬件<br/><em>XR / leader / 外骨骼 / 手套 / 视觉</em>"]
  S2["**Stage 2**<br/>映射与延迟<br/><em>笛卡尔 IK / 重定向 / 双边稳定</em>"]
  S3["**Stage 3**<br/>人形全身遥操作<br/><em>三点补全 → 全身跟踪 → WBC/流形</em>"]
  S4["**Stage 4**<br/>手指/灵巧手遥操作<br/><em>手姿估计 → 手部重定向 → 触觉</em>"]
  S5["**Stage 5**<br/>下游闭环与进阶<br/><em>IL / 无机器人采集 / 极端场景</em>"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5

  classDef stage fill:#0f2f30,stroke:#00cec9,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4,S5 stage
```

## 这条路径怎么用

- 目标读者是想搭"人操控 → 机器人执行 + 数据采集"闭环的人——人形数据飞轮的第一桶数据几乎都要穿过遥操作这道闸
- 遥操作解决 **人机异构映射 + 实时闭环** 两件事：人和机器人 DOF、工作空间、动力学都不同（Stage 2），又要在几十毫秒延迟内闭环（Stage 2 双边稳定），还要同时管**全身**（Stage 3）与**手指**（Stage 4）两条 DOF 差异极大的子链
- 本路线把用户最关心的两块——**人形机器人全身遥操作**与**手指遥操作**——分别单列为 Stage 3 与 Stage 4，各自有独立的映射难点、低层控制与硬件谱系
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 本路线横跨主路线 L2（运动学：臂部 IK 与全身重定向是 Stage 2/3 的直接前置）与 L5（RL 与模仿学习：全身跟踪策略与下游 IL）
- [动作重定向纵深](depth-motion-retargeting.md) 把"离线把人体动作变参考轨迹"展开成完整谱系；本路线关心的是**实时、在环、带延迟预算**的那一版重定向，并额外覆盖硬件接口与手指子链
- 如果目标偏"边走边动手"的任务面，走 [Loco-Manipulation 纵深](depth-loco-manipulation.md)；本路线聚焦"人怎么把动作灌进机器人"这一侧

---

## Stage 0 遥操作全景与定位

**先分清遥操作的三种输出形态（示范数据 / 部署期接管 / 竞技操控），再选接口，否则会按错误的目标选错误的系统。**

### 前置知识
- Python 熟练，能读懂旋转表示（四元数 / 旋转矩阵 / 6D）
- 对人形机器人 DOF 结构有基本概念（浮动基 + 双臂 + 双手 + 双腿）

### 核心问题
- 遥操作到底在解什么：人机异构下的**实时**动作映射，同时把操作过程沉淀为可学习的示范
- 三种输出形态的评价线为什么不同：**示范数据采集**（要覆盖度与一致性）、**部署期干预接管**（要接得住、不把犹豫学进策略）、**竞技/极端操控**（要低延迟与鲁棒）
- 遥操作在数据管线里的位置：采集 → 清洗 → 重定向 → 模仿学习训练 → 部署
- 遥操作与动捕、外骨骼、无机器人采集的边界在哪

### 推荐做什么
- 读遥操作主入口页，把"为什么重要 / 关键挑战 / 主流系统 / 下游 pipeline"过一遍
- 读一篇综述，建立"接口 × 机器人 × 数据规模"的三维选型直觉

### 推荐读什么
- [Teleoperation（遥操作）](../wiki/tasks/teleoperation.md)（本仓库）— 主入口：挑战 / 系统对比表 / 到策略学习的 pipeline
- [Teleoperation of Humanoid Robots: A Survey（论文笔记）](../wiki/entities/paper-notebook-teleoperation-of-humanoid-robots-a-survey.md)（本仓库）— 综述级全景
- [操作演示数据采集指南（Query）](../wiki/queries/demo-data-collection-guide.md)（本仓库）— 采集侧实操
- [ROVE](../wiki/entities/paper-rove-humanoid-vla-intervention.md)（本仓库）— 为什么"部署期接管 ≠ 专家示范"

### 学完输出什么
- 能一句话说清遥操作解决什么、三种输出形态各要什么
- 能画出"遥操作采集 → 训练 → 部署"数据管线并标出误差来源

---

## Stage 1 输入接口与硬件通道

**遥操作的上限先由接口决定：XR、leader、外骨骼、数据手套、纯视觉各自能采到什么、丢什么，是后续一切的前提。**

### 前置知识
- Stage 0 内容
- 基本的传感与通信概念（USB HID / evdev、OpenXR、DDS）

### 核心问题
- 五类接口谱系的取舍：**XR 头显 + 手柄**（AVP / PICO / Quest）、**leader–follower 主从臂**（ALOHA / GELLO）、**上肢外骨骼**（等动学 cockpit）、**数据手套 / 光学动捕**、**纯视觉**（无穿戴）
- 各接口的采集完备度：末端位姿够不够、有没有手指关节、有没有力/触觉、有没有下身
- 跨平台 XR 中间层怎么把设备差异收拢成统一追踪流
- 无机器人（UMI 式手持）采集为什么能把"采集"与"具体机器人"解耦

### 推荐做什么
- 用一套开源 XR 遥操作栈（跨平台中间层）把头 + 双手追踪流接到一个人形模型上，观察延迟与追踪抖动
- 把同一个抓取任务分别用 leader 臂与 VR 手柄采一遍，对比末端精度与操作员疲劳

### 推荐读什么
- [数据手套 vs 视觉遥操作（对比）](../wiki/comparisons/data-gloves-vs-vision-teleop.md)（本仓库）— 采集通道选型主入口
- [XRoboToolkit](../wiki/entities/paper-xrobotoolkit.md)（本仓库）— OpenXR 跨平台 XR 遥操作中间层（PICO/Quest）
- [ACE 跨平台视觉外骨骼系统（论文笔记）](../wiki/entities/paper-notebook-ace-a-cross-platform-visual-exoskeletons-system.md) 与 [NuExo 上肢外骨骼（论文笔记）](../wiki/entities/paper-notebook-nuexo-a-wearable-exoskeleton-covering-all-upper.md)（本仓库）— 外骨骼 cockpit
- [UME-EXO](../wiki/entities/paper-ume-exo.md)（本仓库）— 外骨骼实时力矩反馈 + 全身臂形采集
- [HandUMI](../wiki/entities/handumi.md) 与 [mimic wearable U1](../wiki/entities/mimic-wearable-u1.md)（本仓库）— 无机器人手持/外骨骼采集
- [RIO（Robot I/O）](../wiki/entities/robot-io-rio.md)（本仓库）— 多设备遥操作的 Node 化抽象
- [xpad](../wiki/entities/xpad.md) 与 [Oculus Quest Teleop](../wiki/entities/oculust-quest-teleop.md)、[Isaac Teleop](../wiki/entities/isaac-teleop.md)（本仓库）— 手柄内核驱动 / VR / 仿真内录制
- [ALOHA](../wiki/entities/aloha.md)（本仓库）— 低成本 leader–follower 双臂

### 学完输出什么
- 一份自己方向的接口选型表（末端 / 手指 / 力触觉 / 下身 / 成本五列）
- 一条能把 XR 或 leader 追踪流实时接到机器人模型的最小采集回路

---

## Stage 2 映射与延迟：从设备信号到机器人指令

**遥操作是"带延迟预算的实时重定向"：末端跟踪、构型连续、延迟补偿、双边稳定四件事要一起解。**

### 前置知识
- Stage 1 内容
- 正/逆运动学（FK/IK）概念（参考主路线 L2 与 [动作重定向纵深](depth-motion-retargeting.md) Stage 1）
- 线性代数与最小二乘

### 核心问题
- 臂部笛卡尔跟踪：VR/手柄每 tick 给末端 `T_target`，怎么求"离当前关节角最近的单解 IK"以保持构型连续
- 实时重定向 vs 离线重定向的差异：延迟预算、滑窗推理、安全限幅
- 双边遥操作稳定性：力反馈闭环下如何在通信延迟中保持无源/稳定
- 几何/闭式重定向为什么在上身遥操作里够用又便宜

### 推荐做什么
- 用解析 IK（如 ssik）对一个 6R/7R 臂做 VR 末端跟踪，验证 `q_seed` 最近分支与跳变检测触发重规划
- 给回路人为注入 100–200 ms 延迟，观察精细对齐任务的成功率退化，试一版预测性显示

### 推荐读什么
- [ssik（解析逆运动学）](../wiki/entities/ssik.md)（本仓库）— 6R/7R 臂 `q_seed` 最近分支与 `seed_tolerance` 跳变检测
- [Motion Retargeting](../wiki/concepts/motion-retargeting.md) 与 [Motion Retargeting Pipeline](../wiki/concepts/motion-retargeting-pipeline.md)（本仓库）— 重定向问题与管线
- [GMR（运动学重定向）](../wiki/methods/motion-retargeting-gmr.md)（本仓库）— 先几何对齐、物理留给下游
- [Whole-Body Bilateral Teleoperation（论文笔记）](../wiki/entities/paper-notebook-whole-body-bilateral-teleoperation-with-multi-st.md)（本仓库）— 多站点双边遥操作稳定
- [运动学可行与动力学可行](../wiki/concepts/kinematic-vs-dynamic-feasibility.md)（本仓库）— "摆得出姿势" ≠ "站得住、跟得上"

### 学完输出什么
- 一个能跑的 VR → 臂末端解析 IK 实时跟踪脚本，带构型连续与跳变保护
- 能说清"延迟从哪来、在哪补、双边稳定靠什么保证"

---

## Stage 3 人形全身遥操作

**全身遥操作难在"稀疏输入 + 全身 DOF + 站得住"：头手三点要补全下身，参考要经得起全身跟踪策略与低层 WBC 的执行。**

### 前置知识
- Stage 2 内容
- [RL 纵深](depth-rl-locomotion.md) Stage 0–2 水平（能在仿真里训全身跟踪策略）
- [Whole-Body Control](../wiki/concepts/whole-body-control.md) 基本概念

### 核心问题
- 稀疏遥操作输入（头 + 双手三点，或加腰/踝 tracker）如何补全成全身参考：运动学规划补下身 vs 学习式补全
- 全身跟踪策略（GMT）怎么把噪声人体参考变成真机可执行的全身动作：TWIST2 / SONIC 谱系
- 低层稳定的两条路：显式 WBC vs 行为基础模型的运动流形约束（BFM-Zero）
- 重载、非结构化地形、闭环全局位姿各自给全身遥操作加了什么约束
- 门控专家 / 运动先验为什么比"蒸馏成一个统一策略"更能保住高动态技能

### 推荐做什么
- 跑通一条"VR 三点 → SMPL/关键点重定向 → 全身跟踪策略"的仿真遥操作回路，观察下身补全质量与跌倒率
- 对同一段全身遥操作分别用"逐帧硬跟踪"与"运动流形约束"低层，对比稳定性与可行域

### 推荐读什么
- [Whole-Body Tracking Pipeline](../wiki/concepts/whole-body-tracking-pipeline.md) 与 [WBT 专题汇总](../wiki/overview/topic-wbt.md)（本仓库）— 全身跟踪主链路
- [SONIC（规模化运动跟踪）](../wiki/methods/sonic-motion-tracking.md) 与 [TWIST2](../wiki/entities/paper-twist2.md)（本仓库）— 双 GMT 后端，头手三点驱动上身 + 规划补下身
- [BFM-Zero](../wiki/entities/paper-bfm-zero.md)（本仓库）与 [BFM 纵深](depth-bfm.md)— 运动流形约束的低层，VR 遥操作是其接口之一
- [HEFT](../wiki/entities/paper-heft.md)（本仓库）— 嘈杂 raw VR + WPC 双手负载，全尺寸 L7 重载遥操作
- [PILOT](../wiki/entities/paper-pilot-perceptive-loco-manipulation.md)（本仓库）— VR 长程 loco-manip + 感知 MoE 全身 LLC
- [CLOT](../wiki/entities/paper-amp-survey-16-clot.md)（本仓库）— 闭环全局位姿的长时程无漂移全身遥操作
- [TeleGate](../wiki/entities/paper-telegate.md)（本仓库）— 门控选冻结专家 + VAE 运动先验，高动态跑跳/跌倒恢复
- [CWI](../wiki/entities/paper-cwi-composite-humanoid-whole-body-imitation.md)（本仓库）— Quest 双手 9D keypoint + 速度/身高蒸馏接口，无需全身 MoCap
- [CHILD 全身人形遥操作系统（论文笔记）](../wiki/entities/paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md) 与 [HOMIE 等动学外骨骼 cockpit（论文笔记）](../wiki/entities/paper-loco-manip-161-040-homie.md)（本仓库）

### 学完输出什么
- 一条从 VR 稀疏输入到全身可执行动作的端到端遥操作回路
- 能说清"稀疏输入怎么补全、低层用 WBC 还是流形、高动态怎么不塌"的取舍

---

## Stage 4 手指与灵巧手遥操作

**手指遥操作是全身遥操作里 DOF 密度最高、观测最难、接触最敏感的子链：人手 ~21 DOF 要映射到 6–20+ DOF 机械手，还要保住接触语义。**

### 前置知识
- Stage 2 内容（手部映射同样是带约束的实时 IK）
- [动作重定向纵深](depth-motion-retargeting.md) Stage 5 方向 B（灵巧手接触保持重定向）

### 核心问题
- 手部姿态怎么来：**数据手套/触觉手套**（关节角直接量）vs **纯视觉手姿估计**（AVP/相机，省穿戴但丢遮挡与力）
- 手部重定向的核心矛盾：**关节空间直接映射**（同构手）vs **指尖/接触任务空间映射**（异构手），以及统一手部动作空间怎么抽象
- 接触保持：从全身骨架级映射细化到**手-物接触拓扑保持**，避免穿模/脱触
- 触觉回灌：机器人端触觉（GelSight/F-T）进策略，操作员端力触觉显示尺寸/刚度
- 双手协调与灵巧手硬件谱系对可达重定向的约束

### 推荐做什么
- 用一副（或视觉估计的）手姿把人手关节角映射到一只开源灵巧手，先做关节空间直接映射，再做指尖任务空间 IK，对比抓取成功率与穿模
- 给手部重定向加一个接触保持项（指尖-物面距离约束），统计脱触帧比例下降

### 推荐读什么
- [数据手套 vs 视觉遥操作（对比）](../wiki/comparisons/data-gloves-vs-vision-teleop.md)（本仓库）— 手部采集两条通道的主对比
- [Dexterous Kinematics（灵巧手运动学）](../wiki/concepts/dexterous-kinematics.md) 与 [UHAS 统一手部动作空间](../wiki/methods/uhas-unified-hand-action-space.md)（本仓库）— 手部映射抽象
- [TopoRetarget](../wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md)、[SPIDER](../wiki/methods/spider-physics-informed-dexterous-retargeting.md) 与 [DynaRetarget vs TopoRetarget（对比）](../wiki/comparisons/dynaretarget-vs-toporetarget-retargeting.md)（本仓库）— 接触保持灵巧手重定向
- [Bunny-VisionPro 实时双手灵巧遥操作（论文笔记）](../wiki/entities/paper-notebook-bunny-visionpro-real-time-bimanual-dexterous-tel.md)、[DexUMI（论文笔记）](../wiki/entities/paper-notebook-dexumi-using-human-hand-as-the-universal-manipul.md)、[DexterCap（论文笔记）](../wiki/entities/paper-notebook-dextercap.md)（本仓库）— 视觉/手持手指采集
- [OSMO 开源触觉手套（论文笔记）](../wiki/entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md) 与 [ByteDexter 20-DOF 灵巧手遥操作（论文笔记）](../wiki/entities/paper-notebook-dexterous-teleoperation-of-20-dof-bytedexter-han.md)（本仓库）— 触觉手套 / 高 DOF 手
- [HapMorph](../wiki/entities/paper-hapmorph-pneumatic-haptic-render.md) 与 [Touch Dreaming 触觉策略](../wiki/methods/humanoid-transformer-touch-dreaming.md)（本仓库）— 操作员侧力触觉显示 / 机器人侧触觉进策略
- 灵巧手硬件谱系：[Shadow Hand](../wiki/entities/shadow-hand.md)、[Allegro Hand](../wiki/entities/allegro-hand.md)、[RUKA v2](../wiki/entities/ruka-v2-hand.md)、[ORCA Hand](../wiki/entities/orca-hand.md)、[mimic hand M1](../wiki/entities/mimic-hand-m1.md)、[MIDAS Hand](../wiki/entities/midas-hand.md)、[Handroid](../wiki/entities/handroid.md)（本仓库）
- [CoordEx 灵巧人形 loco-manipulation](../wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md)（本仓库）— 全身 + 灵巧手协同遥操作

### 学完输出什么
- 一条从人手姿态（手套或视觉）到灵巧手可执行指令的实时手部重定向回路
- 能说清"手姿从哪来、异构手怎么映射、接触怎么不丢、触觉要不要回灌"的取舍

---

## Stage 5 下游闭环与进阶方向

**遥操作不是终点：产物要经得起模仿学习训练、大规模采集与真机部署的检验。**

### 前置知识
- Stage 3 与 Stage 4 内容

### 核心问题
- 遥操作示范进入模仿学习的完整链路：`(obs, action)` 序列 → BC / ACT / Diffusion Policy → 部署
- 无机器人采集怎么把"采集"规模化：UMI 式手持 + 后期重定向到多种机器人
- 部署期干预接管的正确用法：为什么要价值引导提取而非一律当专家 BC
- 数据规模与 benchmark：家庭级大规模遥操作集、遥操作模拟 benchmark
- 极端场景：重载、竞技全接触、手术等对遥操作系统的额外要求

### 推荐做什么
- 把 Stage 3/4 采到的示范喂给一个开源 IL 管线（BC/ACT/扩散），观察成功率与数据效率
- 对"有/无失败示范过滤"两组数据各训一版策略，验证数据质量对收敛的影响

### 推荐读什么
- [Imitation Learning](../wiki/methods/imitation-learning.md) 与 [Diffusion Policy](../wiki/methods/diffusion-policy.md)（本仓库）— 遥操作数据的主要消费者
- [BifrostUMI](../wiki/entities/paper-bifrost-umi.md) 与 [HALOMI](../wiki/entities/paper-halomi-humanoid-loco-manipulation.md)（本仓库）— 无机器人示范 → 人形全身 visuomotor
- [ROVE](../wiki/entities/paper-rove-humanoid-vla-intervention.md)（本仓库）— 部署期 MoCap 接管与次优接管轨迹的 RL 后训练
- [HIW-500 数据集](../wiki/entities/hiw-500-dataset.md)（本仓库）— 500+ h / 23K+ 集家庭级全身遥操作开源集
- [TeleopBench（论文笔记）](../wiki/entities/paper-notebook-teleopbench-a-simulator-centric-benchmark-for-du.md) 与 [HumanoidArena](../wiki/entities/paper-humanoidarena.md)（本仓库）— 遥操作 / 分层控制评测后端
- [EgoDex（论文笔记）](../wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md)（本仓库）— 大规模 egocentric 灵巧数据

**进阶方向：**
- **无遥操作合成 vs 遥操作采集成本**——参考 [动作生成纵深](depth-motion-generation.md) 与 [Real2Sim/合成演示](depth-real2sim.md) 一侧的对照
- **极端与商用场景**：[HEFT](../wiki/entities/paper-heft.md) 重载、[REK](../wiki/entities/rek.md) / [URKL](../wiki/entities/urkl.md) 竞技格斗、[Humanoid Surgeon](../wiki/entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md) 手术 in vivo
- **跨具身遥操作**：一套人体参考驱动异构人形/机械手，参考 [动作重定向纵深](depth-motion-retargeting.md) Stage 5

### 学完输出什么
- 一条从遥操作采集走到可部署策略的端到端管线
- 对"遥操作质量 ↔ 下游策略效果"的因果链有实验级认识

---

## 快速入口汇总

| 阶段 | 核心问题 | 本仓库入口 |
|------|---------|-----------|
| Stage 0 | 问题定义与输出形态 | [Teleoperation](../wiki/tasks/teleoperation.md) |
| Stage 1 | 输入接口与硬件 | [数据手套 vs 视觉遥操作](../wiki/comparisons/data-gloves-vs-vision-teleop.md) |
| Stage 2 | 映射与延迟/双边稳定 | [ssik（解析 IK）](../wiki/entities/ssik.md) |
| Stage 3 | 人形全身遥操作 | [Whole-Body Tracking Pipeline](../wiki/concepts/whole-body-tracking-pipeline.md) |
| Stage 4 | 手指/灵巧手遥操作 | [Dexterous Kinematics](../wiki/concepts/dexterous-kinematics.md) |
| Stage 5 | 下游闭环与进阶 | [Imitation Learning](../wiki/methods/imitation-learning.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 其它纵深路径：
  - [动作重定向（人体动作 → 机器人参考轨迹）](depth-motion-retargeting.md) — 本路线 Stage 2/4 用其重定向工具箱；差别在"实时、在环、带延迟"
  - [模仿学习与技能迁移](depth-imitation-learning.md) — 遥操作示范的主要下游消费者
  - [BFM（人形行为基础模型）](depth-bfm.md) — 全身遥操作低层的流形约束来源与数据消费者
  - [Loco-Manipulation（移动操作）](depth-loco-manipulation.md) — 全身遥操作的典型任务面
  - [接触丰富的操作任务](depth-contact-manipulation.md) — 手指遥操作接触保持的邻接路线
  - [人形 RL 运动控制](depth-rl-locomotion.md) — 全身跟踪策略的训练侧前置
  - [VLA（视觉-语言-动作模型）](depth-vla.md) — 遥操作数据训练 VLA、部署期干预接管
  - [动作生成（文本/多模态 → 人形动作）](depth-motion-generation.md) — 无遥操作合成演示的对照侧
  - [Sim2Real（域差画像 → 执行器对齐 → 鲁棒训练 → 真机部署）](depth-sim2real.md) — 全身遥操作策略上真机的部署侧
  - [人形拳击（动作跟踪 → 潜空间技能 → 对抗自博弈）](depth-humanoid-boxing.md)、[人形群控展演](depth-humanoid-swarm-performance.md) — 竞技/展演向的遥操作极端场景
  - [感知越障（Perceptive Locomotion）](depth-perceptive-locomotion.md) — 非结构化地形遥操作的感知前置
- 人形控制全景图：[Humanoid Control Roadmap](../wiki/roadmaps/humanoid-control-roadmap.md)
- 技术栈地图：[tech-map/dependency-graph.md](../tech-map/dependency-graph.md)

## 参考来源

本路线基于以下原始资料的归纳：

- [Teleoperation（遥操作）](../wiki/tasks/teleoperation.md) 与 [数据手套 vs 视觉遥操作](../wiki/comparisons/data-gloves-vs-vision-teleop.md) — 主入口与接口选型
- Goertz, *Master-Slave Manipulator*（Argonne National Laboratory, 1954）— 主从机械手遥操作的奠基工作，遥操作/远程机器人学的起点
- He et al., *OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation* (2024) — 人形全身遥操作代表作
- "XRoboToolkit"（arXiv:2508.00097，SII 2026 Best Paper）— OpenXR 跨平台 XR 遥操作套件
- [TWIST2](../wiki/entities/paper-twist2.md)、[HEFT](../wiki/entities/paper-heft.md)、[TeleGate](../wiki/entities/paper-telegate.md)、[CWI](../wiki/entities/paper-cwi-composite-humanoid-whole-body-imitation.md) — 全身遥操作低层与接口谱系
- [TopoRetarget](../wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md)、[UHAS](../wiki/methods/uhas-unified-hand-action-space.md) — 手指遥操作重定向与手部动作空间
