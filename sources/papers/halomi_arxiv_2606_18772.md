# HALOMI: Learning Humanoid Loco-Manipulation with Active Perception from Human Demonstrations

> 来源归档（ingest）

- **标题：** HALOMI: Learning Humanoid Loco-Manipulation with Active Perception from Human Demonstrations
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2606.18772>
  - <https://arxiv.org/html/2606.18772v1>
  - <https://halomi-humanoid.github.io>
- **作者：** Zehui Zhao, Yuxuan Zhao, Gaojing Zhang, Chenxi Liu, Maolin Zheng, Wenzhao Lian（† 通讯）
- **机构：** 上海交通大学、英国萨塞克斯大学、华东理工大学
- **入库日期：** 2026-06-22
- **一句话说明：** 扩展 UMI 式无机器人示范到 **人形 loco-manipulation + 主动感知**：头盔 egocentric + 双 Pika Sense 夹爪同步采集头–手轨迹与多视角 RGB；**BFM-Zero 潜空间流形约束全身控制器** 跟踪稀疏世界系头手目标；**ego-view 对齐 + 控制器感知参考轨迹适配** 缩小人–机鸿沟；高层 **π₀.₅ VLA** 在 **Unitree G1（3-DoF 主动颈）** 上五项真机任务平均成功率约 **85%**。

## 核心论文摘录（MVP）

### 1) 问题：人示范可扩展，但主动感知与稀疏头手接口难落地

- **链接：** <https://arxiv.org/abs/2606.18772> §I
- **摘录要点：** 人形全身遥操作采集贵且慢；人类示范可规模化且天然含 **主动手眼协调**（搜索、抓取、放置等多阶段凝视切换）。既有 UMI+egocentric 路线多服务固定基座/轮式平台；扩展到人形 loco-manipulation 的工作（HuMI、BifrostUMI 等）常需 **骨盆/脚等下身参考** 或 **未显式研究 ego-view 迁移**。直接用世界系头手轨迹驱动全身控制器在 OOD 目标下脆弱，且存在 **观测（egocentric 视角差）** 与 **执行（跟踪误差累积）** 双重 embodiment gap。
- **对 wiki 的映射：**
  - [HALOMI（论文实体）](../../wiki/entities/paper-halomi-humanoid-loco-manipulation.md) — 问题定位与相对 HoMMI/HuMI/BifrostUMI 的差异。
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 无机器人示范 + 主动感知子路线。

### 2) 采集：UMI 双夹爪 + 可穿戴 egocentric 头手追踪

- **链接：** <https://halomi-humanoid.github.io>；arXiv §III-A
- **摘录要点：**
  - **双 Agilex Pika Sense 手持夹爪**：鱼眼 RGB + 夹爪宽度编码器。
  - **头盔**：Intel RealSense D435i egocentric RGB + HTC VIVE Tracker 3.0；Lighthouse 追踪双夹爪与头部 **6-DoF**，毫米级精度。
  - **统一动作空间**：左右手位姿 + 头部位姿 + 双夹爪开度；观测为 **egocentric + 双 gripper-centric RGB**，30 Hz 同步。
  - **任务中心接口**：只录头与双手，下身由控制器补全——相对 HuMI/BifrostUMI **不需演示者提供骨盆/脚目标**。
- **对 wiki 的映射：**
  - [HALOMI](../../wiki/entities/paper-halomi-humanoid-loco-manipulation.md) — 采集硬件表。
  - [Teleoperation](../../wiki/tasks/teleoperation.md) — Robot-Free 示范采集谱系。

### 3) 执行：BFM-Zero 流形约束全身控制器 + 3-DoF 主动颈

- **链接：** arXiv §III-B；项目页 Global Head-Hand Tracking Controller
- **摘录要点：**
  - **稀疏世界系跟踪**：双手 6-DoF + 头部 3D 位置；头朝向由 **自研 3-DoF 伺服颈**（yaw/pitch/roll，光心近似共轴）解析 IK 解耦，避免 G1 无独立颈时「转头牵全身」。
  - **流形约束**：直接在关节空间学世界系跟踪易多模态、激进不稳定；在 **BFM-Zero 球形潜空间** 上训 RL teacher（特权未来参考 + 全身跟踪误差），再 **DAgger** 蒸馏仅观测头手误差 + 本体的 student。
  - **数据**：>6000 条 loco-manipulation 动作序列；teacher–student 管线。
  - **鲁棒性**：相对 raw action-space 跟踪，流形控制器在 OOD 大跳变/不可行命令下仍保持可行全身行为。
- **对 wiki 的映射：**
  - [BFM-Zero](../../wiki/entities/paper-bfm-zero.md) — 行为先验与潜空间规划。
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)。

### 4) 人–机对齐：Ego-view 对齐 + 控制器感知参考轨迹适配

- **链接：** arXiv §III-C
- **摘录要点：**
  - **Ego-view alignment**：沿用 EgoHumanoid 流程——单目深度重建 3D 场景 → 重投影到人形视角 → inpainting 填洞。
  - **Controller-aware reference trajectory adaptation**：仿真中 rollout 原始头手参考，用 B-spline 残差 + **粗到细全局/局部 CEM** 并行优化，使闭环执行轨迹更接近人类示范；Table I 跟踪误差平均降约 **6.7%**。
- **对 wiki 的映射：**
  - [HALOMI](../../wiki/entities/paper-halomi-humanoid-loco-manipulation.md) — 离线处理管线 Mermaid。
  - [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) — 与几何重定向互补的「控制器感知」适配。

### 5) 高层策略：π₀.₅ VLA + FastUMI 式相对动作块

- **链接：** arXiv §III-D
- **摘录要点：** 处理后人示范微调 **π₀.₅**；输入同步 egocentric + 双 gripper RGB 与语言指令；输出 **左/右手 + 头** 的相对位姿块（FastUMI 风格）+ 夹爪；30 Hz 视觉缓冲异步，50 Hz 低层控制器跟踪世界系笛卡尔目标。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md)、[π0.7 Policy](../../wiki/methods/pi07-policy.md)（同族骨干）。

### 6) 真机评测：G1 + 五项 loco-manipulation

- **链接：** arXiv §IV；<https://halomi-humanoid.github.io>
- **摘录要点：**
  - **平台**：Unitree G1 + 机载 Pika 夹爪（与手持端几何/相机布局对齐）+ 3-DoF 主动颈。
  - **五项任务**：Bag Transfer（长距导航+视觉搜索）、Pick Bread and Place、Transfer Towel to Basket（双手+全身）、Squat-and-Grasp、Tossing。
  - **定量**（各 20 次）：Bag Transfer **90%**、Pick Bread **85%**、Transfer Towel **80%**；三任务平均 **85%**。
  - **消融**：去 ego-view 对齐 Bag Transfer **90%→10%**；关主动颈 Bag Transfer **30%**、Towel **80%→10%**；参考轨迹适配 Pick Bread **75%→85%**。
  - **泛化**：未见柜子位姿 60%、新毛巾外观 60%；面包–盘子相对布局变化时成功率降至 0/10。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)。

### 7) 与相邻无机器人人形路线对照（Related Works）

- **链接：** arXiv §II-C
- **摘录要点：** **HoMMI** 用 3D egocentric + look-at-point 头接口；**HuMI / BifrostUMI** 附加骨盆/脚参考且缺主动感知；**EgoHumanoid** 做 view/action 对齐与共训。HALOMI 以 **头手稀疏接口** + **统一流形约束 WBC** + **自动化离线对齐**，强调 **主动凝视行为** 可规模化采集并迁移。
- **对 wiki 的映射：**
  - [BifrostUMI](../../wiki/entities/paper-bifrost-umi.md)、[paper-notebook-humanoid-manipulation-interface](../../wiki/entities/paper-notebook-humanoid-manipulation-interface.md)（HuMI）、[paper-notebook-egomi](../../wiki/entities/paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md)。

## 当前提炼状态

- [x] arXiv 摘要、方法四模块与实验/ablation 已摘录
- [x] 项目页 Method / Controller / Results 结构已索引
- [x] wiki 映射：`wiki/entities/paper-halomi-humanoid-loco-manipulation.md`；交叉 [loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[teleoperation](../../wiki/tasks/teleoperation.md)、[bfm-zero](../../wiki/entities/paper-bfm-zero.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)
- [ ] 官方代码/数据发布后补充 `sources/repos/` 条目
