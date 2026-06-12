# BifrostUMI: Bridging Robot-Free Demonstrations and Humanoid Whole-Body Manipulation

> 来源归档（ingest）

- **标题：** BifrostUMI: Bridging Robot-Free Demonstrations and Humanoid Whole-Body Manipulation
- **类型：** paper
- **来源：** arXiv abs / PDF；项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2605.03452>
  - <https://arxiv.org/pdf/2605.03452>
  - <https://baai-aether.github.io/BifrostUMI/>
- **作者：** Chenhao Yu, Hongwu Wang, Youhao Hu, Jiachen Zhang, Yuanyuan Li, Shaqi Luo（BAAI Aether 项目页）
- **入库日期：** 2026-05-23
- **一句话说明：** 受 UMI 启发的 **无机器人** 人形全身操作数据采集：Pico 4 追踪 + 双腕鱼眼夹爪同步记录关键点、腕部视觉与夹爪开度；**扩散高层策略**（DINOv2 + 47-D 稀疏关键点动作）经 **SKR（Spatial Keypoint Retargeting）** 映射到 G1 的 36-D 运动表示，再由 **全身控制器 + mink IK** 闭环执行；真机验证杂乱桌面 pick-place 与桌下全身处置。

## 核心论文摘录（MVP）

### 1) 问题：遥操作采集人形全身数据的瓶颈

- **链接：** <https://arxiv.org/abs/2605.03452>
- **摘录要点：** 高质量人形 **全身 visuomotor** 策略依赖大规模示范，但主流 **机器人遥操作** 受硬件可得性与操作效率限制。UMI 证明「手持设备 + 腕部相机」可在 **无目标机器人** 时采集可迁移操作数据；BifrostUMI 将这一范式扩展到 **人形全身操作**，用类人三层级（高层策略 → 空间关键点重定向 → 低层 WBC）衔接自然人体示范与可执行人形行为。
- **对 wiki 的映射：**
  - [BifrostUMI（论文实体）](../../wiki/entities/paper-bifrost-umi.md) — 定位与相对 UMI / OmniH2O 的差异。
  - [Teleoperation](../../wiki/tasks/teleoperation.md) — 无机器人示范采集路线。

### 2) 硬件：Robot-Free 多模态采集

- **链接：** <https://baai-aether.github.io/BifrostUMI/>（Hardware）
- **摘录要点：**
  - **Pico 4**：双脚 + 腰部追踪器，经 SDK 得到对齐机器人坐标系的 **全身关键点**。
  - **双 instrumented gripper**：各带 **鱼眼相机**（腕部视角）与 **磁编码器** 测夹爪开度。
  - **三路同步流**：左右腕 RGB、全身关键点状态、夹爪宽度——全程 **无需接触真机**。
- **对 wiki 的映射：**
  - [BifrostUMI](../../wiki/entities/paper-bifrost-umi.md) — 采集栈表与 Mermaid 管线。

### 3) 方法：三层级 visuomotor 控制

- **链接：** <https://baai-aether.github.io/BifrostUMI/>（Method / High-Level / SKR）
- **摘录要点：**
  - **高层**：Whole-body **Diffusion Policy**；观测为 DINOv2 编码的双腕 RGB + 15-D 下半身本体 + 扩散步；动作 **47-D**（5 个关键点：骨盆、左右 TCP、左右脚，各 3D 平移 + 6D 旋转 + 2 夹爪标量）；预测 **H=48** 动作块；监督在 **各关键点局部坐标系** 编码以弱化世界系依赖。
  - **SKR**：相对 GMR 等全局/局部缩放，**仅缩放骨盆–脚垂直距离** 补偿身高，**保留其余度量空间关系**；状态估计用 Unitree SDK + 骨盆系 FK；与高层预测合成目标后 **mink（MuJoCo IK）** 求 29 关节 + 根位姿的 **36-D** 表示。
  - **低层**：全身控制器跟踪重定向运动（闭环 proprioception）。
- **对 wiki 的映射：**
  - [BifrostUMI](../../wiki/entities/paper-bifrost-umi.md) — 流程总览 Mermaid。
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)。

### 4) 实验：Unitree G1 真机

- **链接：** <https://baai-aether.github.io/BifrostUMI/>（Experiments）
- **摘录要点：**
  - **杂乱桌面 pick-and-place**：视觉定位、抓取、放置至目标盘，全程动态稳定。
  - **桌下全身处置**：抓纸团、后退、屈膝弯腰将物体投入桌下垃圾桶——超出纯臂操作范围的 **全身协调**。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)。

### 5) 公开资源

- **项目页：** <https://baai-aether.github.io/BifrostUMI/>
- **代码：** Coming Soon（项目页标注）
- **对 wiki 的映射：**
  - [sources/sites/bifrost-umi-project.md](../sites/bifrost-umi-project.md)

### 6) 与 TWIST2 / HuMI 等采集范式对照（arXiv Related Works）

- **链接：** <https://arxiv.org/abs/2605.03452> §II-A
- **摘录要点：** 论文将 **TWIST2、CLONE、Touch Dreaming** 归为 **机器人内环遥操作**（具身一致但成本高）；**EgoHumanoid** 偏 egocentric 人示范迁移；**UMI / HoMMI** 为便携无机器人臂/移动操作；**HuMI** 用 Vive+UMI 夹爪但标定重、重定向与 WBC 紧耦合。**BifrostUMI** 用 **Pico + UMI 夹爪** 做 **全身** 无机器人采集，并 **显式 SKR** 桥接扩散策略与 WBC。
- **对 wiki 的映射：**
  - [TWIST2](../../wiki/entities/paper-twist2.md) — 真机便携全身遥操作对照
  - [BifrostUMI](../../wiki/entities/paper-bifrost-umi.md) — 三层级 visuomotor 分解

## 当前提炼状态

- [x] arXiv 摘要与项目页 Method / Hardware / Experiments 已摘录
- [x] arXiv Related Works 与 TWIST2 等采集范式对照已摘录（2026-06-12）
- [x] wiki 映射：`wiki/entities/paper-bifrost-umi.md`；交叉 [teleoperation](../../wiki/tasks/teleoperation.md)、[motion-retargeting](../../wiki/concepts/motion-retargeting.md)、[diffusion-policy](../../wiki/methods/diffusion-policy.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[paper-twist2](../../wiki/entities/paper-twist2.md)
- [ ] 官方代码发布后补充 `sources/repos/` 条目
