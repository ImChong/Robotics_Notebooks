# Universal Manipulation Exoskeleton: Learning Compliant Whole-body Policies with Real-time Torque Feedback

> 来源归档（ingest）

- **标题：** Universal Manipulation Exoskeleton: Learning Compliant Whole-body Policies with Real-time Torque Feedback
- **类型：** paper
- **来源：** arXiv abs / PDF；项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2606.14218>
  - <https://arxiv.org/pdf/2606.14218>
  - <https://ume-exo.github.io/>
- **作者：** Litian Liang*, Jingxi Xu*, Xinda Qi, Yujun Cai, Houzhu Ding, Luqi Wang, Zhixin Sun, Jyh-Herng Chow, Ming Yang, Mark Cutkosky（Ant Group、Stanford University；*Equal contribution）
- **入库日期：** 2026-06-18
- **一句话说明：** 低成本（约 $1900）上肢外骨骼 UME：实时透明触觉力矩反馈 + 全身臂形配置与关节力矩同步记录；通用 3-1-3 子链重定向可遥操作 OpenArm / Franka / X-ARM；嵌入式 IMU 驱动移动底座；ACT 融合力矩模态学习主动柔顺双臂全身策略，在力主导、视觉遮挡、极窄空间与长时程移动操作任务上显著优于无力矩与 UMI 式末端示教基线。

## 核心论文摘录（MVP）

### 1) 问题：主流采集管线缺少力/力矩与全身臂形

- **链接：** <https://arxiv.org/abs/2606.14218> §1
- **摘录要点：** 家庭协作机器人需要 **主动柔顺** 与接触力感知，但 VLA / 世界模型路线普遍缺乏力矩数据。ALOHA / GELLO 等 leader–follower **不记录关节力矩**，既学不出主动柔顺，也无法把从臂阻力回传给操作者。UMI 式手持夹爪虽有隐式触觉，但通常只录 **末端位姿**，靠 IK 避碰，在杂乱/极窄空间表现差。UME 让操作者 **实时感受从臂阻力**，并记录 **全身关节位置 + 力矩** 以训练柔顺全身双臂策略。
- **对 wiki 的映射：**
  - [UME-EXO（论文实体）](../../wiki/entities/paper-ume-exo.md) — 三大能力 Cap1–3 与问题定位。
  - [Teleoperation](../../wiki/tasks/teleoperation.md) — 力矩反馈 vs 无反馈示教谱系。

### 2) 硬件：同轴 3-1-3 外骨骼 + 准直驱 + IMU

- **链接：** <https://ume-exo.github.io/>；arXiv §2.1–2.3
- **摘录要点：**
  - 人臂 **3-1-3** 结构：肩/腕虚拟球关节各 3 DoF，肘 1 DoF；J1–J3、J5–J7 **同轴** 布置，肩/腕轴心与人关节球心相交，复现大部分肩腕 RoM。
  - 采用 **Damiao / Unitree** 等 **低减速比准直驱** 电机（相对 Dynamixel 高减速），支撑透明力矩反馈；整机约 **$1900**。
  - 板载 **IMU**（约 $9）将上身欧拉角线性映射为移动底座速度 \([\dot{x}, \dot{y}, \dot{\theta}]\)；可换 VR 头显作 drop-in。
  - 重力/科氏/摩擦补偿提升交互透明度；盲fold 拔剑等演示验证实时触觉。
- **对 wiki 的映射：**
  - [UME-EXO](../../wiki/entities/paper-ume-exo.md) — 机械与控制设计表。

### 3) 方法：通用子链重定向 + ACT 力矩模态

- **链接：** arXiv §2.2、§2.4
- **摘录要点：**
  - **Universal Retargeting**：UME 与从臂均拆为 **肩 3DoF 球关节 + 肘 1DoF + 腕 3DoF** 子操纵器；位置/速度前馈在各子链独立 FK/IK 与 Jacobian 伪逆；**触觉力矩** 用从臂测得 \(\tau\) 经 **动力学一致 Jacobian 转置** 映射回外骨骼各子链（肘力矩直连）。
  - 子链解耦避免 6D 末端任务空间在约束空间附近的奇异与不稳定。
  - **策略学习**：**ACT**（Action Chunking with Transformers）；图像 + 关节本体与原版相同，**关节力矩与位置同样嵌入** 后拼接到 ResNet18 图像 embedding，再进 Transformer 预测目标关节位置（编码柔顺信息）。
- **对 wiki 的映射：**
  - [UME-EXO](../../wiki/entities/paper-ume-exo.md) — Mermaid 管线。
  - [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[Action Chunking](../../wiki/methods/action-chunking.md)。

### 4) 实验：四任务 + 消融 + 跨形态遥操作

- **链接：** <https://ume-exo.github.io/>；arXiv §3、Table 1
- **摘录要点：**
  - 自研双臂移动操作平台（约 **$9533**）：WowRobo **OpenArm 1.0** 双臂（$5200）+ Hexfellow **PCW-25** 全向轮（$900）等；四任务各 20 次评测成功率：
    - **Box Pushing**（26 demo）：UME **0.90** vs No-torque 0.50 vs UMI 0.40
    - **Box Flipping**（40 demo）：UME **0.85** vs 两基线 **0**
    - **GPU Picking**（42 demo）：UME **0.95** vs No-torque 0.75 vs UMI **0**
    - **Fridge Drink Retrieval**（157 demo）：UME **0.95** vs No-torque 0.90 vs UMI **0**
  - **No-torque**：同数据集去掉力矩 embedding；**UMI**：仅末端 6D 位姿 + 全身 IK，无力矩。
  - 用户研究：box flipping 上 UME DPM 为无力矩版的 **3.3×**（约为真人速度 71%）。
  - 跨形态：真机 **X-ARM** 双臂递接；仿真 **Franka** 遥操作。
- **对 wiki 的映射：**
  - [Bimanual Manipulation](../../wiki/tasks/bimanual-manipulation.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)。

### 5) 局限与资源

- **链接：** arXiv §5；项目页 Q&A
- **摘录要点：** PLA 连杆偏重、载荷有限；连杆长度固定；Franka 真机演示待硬件到位。代码 **Coming Soon**（项目页）。
- **对 wiki 的映射：**
  - [sources/sites/ume-exo-project.md](../sites/ume-exo-project.md)

## 当前提炼状态

- [x] arXiv 摘要、方法 §2、实验 Table 1 与项目页任务视频已摘录
- [x] wiki 映射：`wiki/entities/paper-ume-exo.md`；交叉 [teleoperation](../../wiki/tasks/teleoperation.md)、[bimanual-manipulation](../../wiki/tasks/bimanual-manipulation.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[motion-retargeting](../../wiki/concepts/motion-retargeting.md)、[action-chunking](../../wiki/methods/action-chunking.md)
- [ ] 官方代码发布后补充 `sources/repos/` 条目
