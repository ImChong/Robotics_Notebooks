# 路线（纵深）：如果目标是人形群控展演（群舞同步 → 编队走位 → 群体特技）

**摘要**：面向"让一群人形机器人同台整齐跳舞、变换队形、协同炫技"的纵深路线（简写 **HSP**，Humanoid Swarm Performance）。从群控展演的任务谱系与"预编排时间轴 vs 在线协调"两大范式出发，先用动作重定向与跟踪 RL 打好单机舞蹈/特技动作基座，再解决多机毫秒级编舞同步（分布式时钟 + 组播广播 + 失步容错），然后进入编队走位（队形定义、轨迹规划与互避碰、场地定位），最后攀登群控特技（高动态单机技能 × 多机协同编排 × 安全边界），按 Stage 0–5 串通核心方法；本路线是 [运动控制主路线](motion-control.md) 的一条分支。

## 路线一览

```mermaid
flowchart LR
  S0["**Stage 0**<br/>全景与前置<br/><em>展演任务谱系 · 预编排 vs 在线协调</em>"]
  S1["**Stage 1**<br/>单机动作基座<br/><em>重定向 / 跟踪 RL / 音乐驱动生成</em>"]
  S2["**Stage 2**<br/>群舞同步<br/><em>时间轴编排 · 时钟同步 · 失步容错</em>"]
  S3["**Stage 3**<br/>编队走位<br/><em>队形变换 / 互避碰 / 场地定位</em>"]
  S4["**Stage 4**<br/>群体特技<br/><em>高动态技能 × 多机协同 × 安全</em>"]
  S5["**Stage 5**<br/>进阶方向<br/><em>在线互动 / 学习式协调 / 异构群</em>"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5

  classDef stage fill:#2d1f3d,stroke:#9b59b6,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4,S5 stage
```

## 这条路径怎么用

- 目标读者是已经能让单台人形跟踪参考动作、想把"很多台人形在同一个舞台上协同展演"做成完整系统的人
- 核心心智模型：群控展演不是"单机舞蹈 × N 份拷贝"——多机共享同一时间轴、同一块场地和同一份安全预算，**同步误差、队形冲突和单点失步都会被观众肉眼放大**；工程上要把"动作正确"升级为"群体正确"
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 本路线是主路线 L5（RL / IL）之后偏"多机展演系统"的应用方向：单机动作基座依赖 [动作重定向纵深](depth-motion-retargeting.md) 与 [RL 纵深](depth-rl-locomotion.md)，编舞素材生产可衔接 [动作生成纵深](depth-motion-generation.md)
- 同为多机方向，[人形足球纵深](depth-humanoid-soccer.md) Stage 4 解决**对抗环境下的在线战术群控**，本路线聚焦**合作环境下的展演群控**——前者赢在实时响应，后者赢在毫秒级同步与视觉美感，两者的通信与协调工具箱大量互通
- 高动态单机特技的训练方法与 [人形拳击纵深](depth-humanoid-boxing.md)、[BFM 纵深](depth-bfm.md) 的动作跟踪谱系同源；真机部署的鲁棒性问题在 [Sim2Real 纵深](depth-sim2real.md) 展开

---

## Stage 0 全景与前置：展演任务谱系与两大范式

**先建立问题意识：群控展演到底在控什么，预编排和在线协调的边界在哪里。**

### 前置知识
- 主路线 L3–L5 水平：能在仿真里训练并跟踪一段参考动作
- 对 [locomotion](../wiki/tasks/locomotion.md) 与 [模仿学习](../wiki/methods/imitation-learning.md) 有使用级直觉

### 核心问题
- 群控展演的三类任务：**群舞**（毫秒级动作同步）、**走位**（队形定义与变换）、**特技**（高动态技能的群体编排），三者对同步精度、规划复杂度与安全预算的要求各不相同
- 两大范式的取舍：**预编排时间轴**（离线编舞 + 分布式时钟回放，舞台展演主流）vs **在线协调**（实时感知 + 角色分配，足球/作业场景主流）——展演群控大多落在前者，但走位过渡与故障兜底离不开后者
- 里程碑演进：2016 央视春晚 **540 台 Alpha 1S 群舞**（规模化预编排回放）→ 2020 Boston Dynamics "Do You Love Me"（高动态编舞单/少机）→ 2025 央视春晚 **16 台 Unitree H1《秧BOT》**（全身动控 + 激光 SLAM 定位 + 集群调度的工程集成）
- 展演群控与 [人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md)（足球群控）的本质差异：合作 vs 对抗、时间轴驱动 vs 事件驱动、美感指标 vs 胜负指标

### 推荐做什么
- 看一遍 2016 Alpha 1S 群舞与 2025《秧BOT》录像，按"单机动作 / 同步机制 / 走位 / 安全"四栏拆解两者的工程含量差异
- 画一张"群舞 / 走位 / 特技"三任务对"同步精度 / 规划复杂度 / 安全预算"三轴的需求矩阵

### 推荐读什么
- [人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md)（本仓库）— 多机群控范式总览；其"常见误区"一节正是本路线与足球群控的分界线
- [Humanoid Soccer 任务页](../wiki/tasks/humanoid-soccer.md)（本仓库）— 在线协调范式的对照任务
- [Unitree G1](../wiki/entities/unitree-g1.md)（本仓库）— 展演与科研两栖的主力平台

### 学完输出什么
- 能说清一场机器人群舞里哪些部分是预编排、哪些必须在线闭环
- 拿到一段群控展演视频能判断其技术含量层级（回放 / 定位走位 / 动态特技）

---

## Stage 1 单机动作基座：重定向、跟踪 RL 与音乐驱动生成

**群体的下限由单机决定：先让一台人形把舞蹈和特技动作跳对、跳稳。**

### 前置知识
- Stage 0 内容
- [动作重定向纵深](depth-motion-retargeting.md) Stage 0–2 水平：能把人体动捕/视频动作转成机器人参考轨迹

### 核心问题
- 舞蹈动作的来源管线：动捕 / 视频估计 → 重定向 → 物理可行化，与 [动作重定向](../wiki/concepts/motion-retargeting.md) 通用管线的差异在于**节拍对齐**与**风格保持**
- 跟踪 RL 怎么选：[DeepMimic](../wiki/methods/deepmimic.md) 谱系单段跟踪、[BeyondMimic](../wiki/methods/beyondmimic.md) 通用跟踪、[SONIC](../wiki/methods/sonic-motion-tracking.md) 规模化预训练（其 token 接口原生支持**音乐条件**）各适合什么规模的曲目库
- 音乐/文本驱动的编舞素材生产：[GENMO](../wiki/methods/genmo.md)、[HY-Motion](../wiki/methods/hy-motion-1.md) 类生成模型怎么接进"生成 → 重定向 → 跟踪"管线
- 展演动作的特殊约束：动作要"好看"而不只是"不摔"——风格奖励、动作幅度与安全裕度的平衡

### 推荐做什么
- 挑一段 10–20 s 舞蹈动捕，跑通"重定向 → 跟踪 RL → 仿真回放"全管线，统计节拍偏差
- 用音乐驱动生成模型产一段候选编舞，对比其物理可行化前后的动作保真度

### 推荐读什么
- [DeepMimic](../wiki/methods/deepmimic.md) 与 [BeyondMimic](../wiki/methods/beyondmimic.md)（本仓库）— 动作跟踪两代代表
- [SONIC](../wiki/methods/sonic-motion-tracking.md)（本仓库）— 规模化跟踪预训练，音乐 token 条件接口
- [GENMO](../wiki/methods/genmo.md) 与 [HY-Motion 1.0](../wiki/methods/hy-motion-1.md)（本仓库）— 音乐/文本驱动动作生成
- [人形参考动作数据集对比](../wiki/comparisons/humanoid-reference-motion-datasets.md)（本仓库）— 曲目库选型
- [Disney Olaf 角色机器人](../wiki/methods/disney-olaf-character-robot.md)（本仓库）— "好看优先"的展演型控制范例

### 学完输出什么
- 一条能把任意舞蹈动捕变成单机可执行策略的管线，含节拍误差报表
- 对"跟踪单段动作 / 通用跟踪 / 生成式编舞"三档方案的成本收益判断

---

## Stage 2 群舞同步：时间轴编排、时钟同步与失步容错

**群舞的核心不是动作难，而是几十台机器人的动作在毫秒尺度上"齐"。**

### 前置知识
- Stage 1 内容
- 对网络通信基本概念（组播、丢包、抖动）有直觉

### 核心问题
- 编舞时间轴的表示：动作段 + 节拍锚点 + 队形关键帧，怎么设计一份既可人工编辑又可程序校验的"总谱"
- 分布式时钟同步：PTP / 无线授时怎么把几十台机器的时间基准压到毫秒级以内；参考 [时钟同步算法](../wiki/concepts/clock-synchronization-algorithms.md) 的主时钟选举 + 偏置/漂移估计框架
- 状态广播与监控：导演台如何用 [UDP 组播](../wiki/formalizations/udp-multicast-dynamics.md) 类机制下发启停/校正命令并回收各机健康状态，丢包与迟到怎么建模
- 失步容错：单机慢半拍 / 摔倒 / 掉线时，是就地追赶、跳段对齐还是退出队形——预案要写进时间轴而不是临场即兴

### 推荐做什么
- 在仿真里起 8–16 个人形实例，用统一时间轴回放同一段舞，注入时钟偏差与随机丢包，量化"观众可见的不齐"阈值
- 给时间轴加"跳段对齐"容错逻辑，测试单机摔倒后重新入列的恢复时间

### 推荐读什么
- [时钟同步算法](../wiki/concepts/clock-synchronization-algorithms.md)（本仓库）— PTP / 分布式时钟统一框架
- [UDP 组播动力学](../wiki/formalizations/udp-multicast-dynamics.md)（本仓库）— 丢包-迟到-乱序-不一致四类随机过程
- [硬件通信与协议专题](../wiki/overview/topic-communication.md)（本仓库）— 底层数据链路选型
- [Balance Recovery 任务页](../wiki/tasks/balance-recovery.md)（本仓库）— 摔倒恢复是失步容错的底座

### 学完输出什么
- 一份可执行的编舞时间轴格式定义 + 多机同步回放的仿真 demo
- 一张"时钟偏差 / 丢包率 → 同步质量"的定量对照表与容错预案清单

---

## Stage 3 编队走位：队形变换、互避碰与场地定位

**走位把群舞从"原地齐"升级为"移动中还齐"：队形是约束，避碰是底线，定位是前提。**

### 前置知识
- Stage 2 内容
- 轨迹规划基础（样条 / 优化式规划的基本概念）

### 核心问题
- 队形的数学表示与变换：关键帧队形 + 机器人-槽位分配（匈牙利算法类指派），怎么让 N 台机器人换队形时总行程短且无交叉
- 互避碰：无人机 swarm 的成熟工具箱（[Crazyswarm2](../wiki/entities/crazyswarm2.md) 的轨迹上传 + 碰撞避免、[EGO-Planner Swarm](../wiki/entities/ego-planner-swarm.md) 的 B-spline 局部规划、[MINCO 蜂群](../wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md) 的时空联合优化）哪些能平移到双足、哪些被"不能瞬时改向 + 会摔倒"打破
- 场地定位：动捕 / UWB / 激光 SLAM 三档精度与部署成本，舞台场景为什么常用"离线地图 + 激光定位"（《秧BOT》路线）
- 走位与舞步的耦合：全向行走命令空间（速度/航向/步频）怎么同时满足队形轨迹与节拍约束——与 [人形足球纵深](depth-humanoid-soccer.md) Stage 1 的参数化全向行走同源

### 推荐做什么
- 实现一个"两队形关键帧 + 指派 + 轨迹生成"的走位规划器，在仿真里跑 16 机换队形，统计最小机间距
- 把走位轨迹叠加到 Stage 2 的时间轴上，验证"边走边跳"时同步质量的退化量

### 推荐读什么
- [Crazyswarm2](../wiki/entities/crazyswarm2.md)（本仓库）— 大规模真机编队的工程参考（定位 → 轨迹 → 监控全链路）
- [EGO-Planner Swarm](../wiki/entities/ego-planner-swarm.md) 与 [Swarm in the Wild](../wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md)（本仓库）— 去中心化互避碰两代方案
- [人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md)（本仓库）— 编队/站位的人形侧约束
- [LiDAR SLAM / LIO / VIO 选型](../wiki/comparisons/lidar-slam-lio-vio-selection.md)（本仓库）— 场地定位技术选型

### 学完输出什么
- 一个 N 机换队形的走位规划器，含指派、避碰与节拍对齐
- 能说清无人机 swarm 方法迁移到人形时哪三件事必须重做（动力学、稳定性、定位）

---

## Stage 4 群体特技：高动态技能 × 多机协同 × 安全边界

**本路线的巅峰：把空翻、武术这类单机极限技能编排成群体节目，同时把风险关进笼子。**

### 前置知识
- Stage 1–3 内容
- [Sim2Real 纵深](depth-sim2real.md) Stage 0–2 水平：理解高动态动作上真机的域差风险

### 核心问题
- 高动态单机特技怎么练：[KungFuAthleteBot](../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md) 的高速武术数据集 + 抗扰跟踪、[KungfuBot](../wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md) 的物理可行化、[ASAP](../wiki/entities/paper-hrl-stack-25-asap.md) 的仿真-真机动力学对齐，各解决特技链路的哪一环
- 特技的群体编排：接力式（依次空翻的波浪）、镜像式（对称武术阵）、协同式（多机搬举/对抛）三类编排对同步与安全的要求梯度
- 多机物理协作的前沿：[TeamHOI](../wiki/entities/paper-amp-survey-17-teamhoi.md) 的去中心化多人协作搬运、[TeamPlay](../wiki/entities/paper-bfm-23-teamplay.md) 的协作行为基座——从"各跳各的"到"力学耦合"的跨越
- 安全边界工程：特技失败半径、机间安全距离、观众隔离区怎么从仿真统计量变成现场执行规范；摔倒检测与 [跌倒恢复](../wiki/tasks/balance-recovery.md) 的自动接管

### 推荐做什么
- 在仿真里给 4 机编排一段"接力空翻"，统计失败传播（一台失误对相邻机的干扰半径）
- 对一个高动态技能做消融：去掉动力学对齐（ASAP 类）后真机成功率掉多少，倒推特技上台的最低对齐要求

### 推荐读什么
- [KungFuAthleteBot](../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md)（本仓库）— 高动态武术跟踪 + 抗扰恢复
- [KungfuBot](../wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md) 与 [KungfuBot2](../wiki/entities/paper-notebook-kungfubot-2.md)（本仓库）— 物理可行化的功夫技能
- [ASAP](../wiki/entities/paper-hrl-stack-25-asap.md)（本仓库）— 高动态技能的仿真-真机对齐
- [TeamHOI](../wiki/entities/paper-amp-survey-17-teamhoi.md) 与 [TeamPlay](../wiki/entities/paper-bfm-23-teamplay.md)（本仓库）— 多机物理协作前沿
- [SMPLOlympics](../wiki/entities/smplolympics.md)（本仓库）— 高动态技能的仿真基准

### 学完输出什么
- 一段 4 机以上、含至少一个高动态技能的群体节目仿真 demo，附失败传播分析
- 一份"特技上台前检查单"：动力学对齐、安全半径、容错预案三项齐备

---

## Stage 5 进阶方向

### 前置知识
- Stage 4 内容

**方向 A：在线互动展演**
- 从固定时间轴走向观众/音乐实时驱动：现场音乐节拍提取 → 在线编舞生成 → 群体即兴，把 Stage 1 的生成模型搬到闭环里
- 关键词：[SONIC](../wiki/methods/sonic-motion-tracking.md)（音乐 token 实时条件）、[GENMO](../wiki/methods/genmo.md)

**方向 B：学习式多机协调**
- 用 [MARL](../wiki/methods/marl.md) 学走位与协作，替代手工指派与规则容错；对照 [CTDE vs 去中心化](../wiki/comparisons/ctde-vs-decentralized-marl.md) 的训练拓扑选型
- 关键词：[MARL](../wiki/methods/marl.md)、[TeamPlay](../wiki/entities/paper-bfm-23-teamplay.md)

**方向 C：异构群展演**
- 人形 + 四足 + 无人机同台：跨形态时间轴、跨尺度定位与统一安全预算
- 关键词：[Crazyswarm2](../wiki/entities/crazyswarm2.md)、[quad-swarm-rl](../wiki/entities/quad-swarm-rl.md)

**方向 D：从展演群控到作业群控**
- 展演打磨出的同步、走位与容错基建，迁移到多机巡检、协同搬运与工厂集群调度；对抗侧姊妹路线见 [人形足球纵深](depth-humanoid-soccer.md) Stage 4
- 关键词：[人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md)、[TeamHOI](../wiki/entities/paper-amp-survey-17-teamhoi.md)

---

## 快速入口汇总

| 阶段 | 核心问题 | 本仓库入口 |
|------|---------|-----------|
| Stage 0 | 任务谱系与范式 | [人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md) |
| Stage 1 | 单机动作基座 | [BeyondMimic](../wiki/methods/beyondmimic.md) |
| Stage 2 | 群舞同步 | [时钟同步算法](../wiki/concepts/clock-synchronization-algorithms.md) |
| Stage 3 | 编队走位 | [Crazyswarm2](../wiki/entities/crazyswarm2.md) |
| Stage 4 | 群体特技 | [KungFuAthleteBot](../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md) |
| Stage 5 | 进阶方向 | [MARL](../wiki/methods/marl.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 其它纵深路径：
  - [人形足球（全向行走 → 感知踢球 → 多机战术）](depth-humanoid-soccer.md) — 对抗环境在线群控的姊妹路线
  - [人形拳击（动作跟踪 → 潜空间技能 → 对抗自博弈）](depth-humanoid-boxing.md) — 高动态对抗技能的邻接路线
  - [动作重定向（人体动作 → 机器人参考轨迹）](depth-motion-retargeting.md) — 单机动作基座的数据侧前置
  - [动作生成（文本/多模态 → 人形动作）](depth-motion-generation.md) — 编舞素材的生成侧前置
  - [人形 RL 运动控制](depth-rl-locomotion.md) — 跟踪策略的训练侧前置
  - [模仿学习与技能迁移](depth-imitation-learning.md) — 参考动作进策略的方法论前置
  - [Sim2Real（域差画像 → 执行器对齐 → 鲁棒训练 → 真机部署）](depth-sim2real.md) — 特技上真机的部署侧前置
  - [Real2Sim（真实世界 → 可仿真资产/场景/孪生）](depth-real2sim.md) — 展演场地/道具的可仿真资产侧
  - [BFM（人形行为基础模型）](depth-bfm.md) — 单基座控全身的谱系路线
  - [导航（SLAM → VLN → 导航 VLA）](depth-navigation.md) — 场地定位工程的邻接路线
  - [感知越障（Perceptive Locomotion）](depth-perceptive-locomotion.md)
  - [Loco-Manipulation（移动操作）](depth-loco-manipulation.md)
  - [接触丰富的操作任务](depth-contact-manipulation.md)
  - [VLA（视觉-语言-动作模型）](depth-vla.md)
  - [WAM（世界–动作模型）](depth-wam.md)
  - [力矩控制电机设计（指标 → 电磁热 → FOC 力矩闭环）](depth-torque-motor-design.md)
  - [传统模型控制（LIP/ZMP → MPC → WBC）](depth-classical-control.md)
  - [安全控制（CLF/CBF）](depth-safe-control.md)
- 关联知识页：
  - [人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md)
  - [时钟同步算法](../wiki/concepts/clock-synchronization-algorithms.md)
  - [UDP 组播动力学](../wiki/formalizations/udp-multicast-dynamics.md)
  - [BeyondMimic](../wiki/methods/beyondmimic.md)
  - [MARL](../wiki/methods/marl.md)

## 参考来源

本路线基于以下原始资料的归纳：

- [人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md) — 群控范式与"预录时间轴 + 分布式时钟 vs 在线战术"的分界
- 2016 央视春晚 540 台 UBTECH Alpha 1S 群舞 — 规模化预编排回放的大众里程碑（本路线起点）；2025 央视春晚 16 台 Unitree H1《秧BOT》 — 全身动控 + 激光 SLAM 定位 + 集群调度的工程集成里程碑
- [KungFuAthleteBot 摘录](../sources/papers/kung_fu_athlete_bot.md)（arXiv:2602.13656）— 高动态武术跟踪与抗扰恢复
- [Swarm in the Wild](../wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md)（Science Robotics 2022）与 [Crazyswarm2](../wiki/entities/crazyswarm2.md) — 无人机 swarm 编队/避碰工具箱的迁移参照
- [TeamHOI](../wiki/entities/paper-amp-survey-17-teamhoi.md)（arXiv:2603.07988）— 多机物理协作的去中心化策略前沿
