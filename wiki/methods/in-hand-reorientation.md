---
type: method
tags: [dexterity, manipulation, robot-hand, reinforcement-learning, control]
status: complete
updated: 2026-11-21
related:
  - ../entities/allegro-hand.md
  - ../entities/shadow-hand.md
  - ../tasks/manipulation.md
  - ../concepts/tactile-sensing.md
  - ../formalizations/cross-modal-attention.md
  - ./uhas-unified-hand-action-space.md
  - ../entities/paper-tacrefinenet-tactile-grasp-refinement.md
  - ../entities/paper-vtap-gripper.md
  - ../entities/paper-teledexter.md
  - ../entities/paper-adept-dexterity.md
  - ../entities/paper-wm-craftnet.md
  - ../entities/paper-artmanip.md
  - ../entities/paper-karma-hand-metric.md
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/uhas_arxiv_2607_03570.md
  - ../../sources/papers/tacrefinenet_arxiv_2509_25746.md
  - ../../sources/papers/vtap_gripper_arxiv_2607_15448.md
  - ../../sources/papers/teledexter_arxiv_2607_11481.md
  - ../../sources/papers/wm_craftnet_arxiv_2609_07002.md
  - ../../sources/papers/artmanip_arxiv_2609_12498.md
original:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/uhas_arxiv_2607_03570.md
  - ../../sources/papers/teledexter_arxiv_2607_11481.md
summary: "手内重定向（In-hand Reorientation）是指机器人灵巧手在不借助于外部环境（如桌面）的前提下，通过手指间的协同动作改变掌心中物体位姿的技术。"
---

# In-hand Reorientation (手内重定向)

## 一句话定义

**手内重定向（In-hand Reorientation）是指多指灵巧手在不释放物体、不依赖桌面等外在支撑面的条件下，仅通过手指间高频交替支撑–滑移–重抓将物体从当前位姿连续调整到目标 6-DoF/绕轴位姿的精细操作技能。**

**任务背景：** 这是灵巧操作（Dexterous Manipulation）中最核心也最具接触切换挑战的任务之一。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- **是工具使用与长程装配的前置技能：** 拧螺丝、拿取换向与在狭小空间内对齐插销前，往往先要靠指间微调解决抓取角度偏差。
- **是检验 RL、触觉反馈与跨本体动作空间的试金石：** 因为高频多指自遮挡与离散接触切换，UHAS 跨手动作空间、TeleDexter 共跟踪与 ADEPT 预训练微调均将其作核心基准。

## 核心挑战

1. **频繁的接触切换 (Hybrid Dynamics)**：为了让物体旋转，手指必须交替进行“支撑-滑动-抬起-再接触”的循环，系统动力学具有极强的离散非线性。
2. **感知遮挡 (Perception Occlusion)**：在手内旋转时，手指会频繁遮挡物体，导致基于视觉的位姿估计失效。
3. **欠驱动与滑移**：由于接触点面积有限，物体极易发生意外滑脱（Slip），需要极高频的力反馈调节。

## 主要技术路线

## 核心原理

### 1. 强化学习 (Deep RL)
目前最成功的手内重定向成果多来自于无模型强化学习（Model-free RL）。
- **代表作**：OpenAI 的 *Learning Dexterous In-Hand Manipulation*。
- **机制**：在仿真中通过海量试错发现奇特的“指尖舞蹈”策略。结合**域随机化 (Domain Randomization)** 解决 Sim2Real 问题。
- **跨具身动作空间**：[UHAS](./uhas-unified-hand-action-space.md) 把手内立方体重定向策略建在 **规范球面形变** 上，用 **单一 PPO 策略** 同时服务 Allegro、LEAP、Shadow、MANO 四手，并支持零样本迁移与快速微调（arXiv:2607.03570）。
- **遥操作共跟踪入口**：[TeleDexter](../entities/paper-teledexter.md) 不直接学任务专用重定向策略，而是用 **指尖+物体连续子目标 co-tracking** 在真机遥操作中执行 Cylinder/Cuboid/Bunny 手内重定向（SharpaWave 上 SR 66.7–80%），并采数训下游 Diffusion Policy。
- **RL 预训练摊销 reorient 段**：[ADEPT](../entities/paper-adept-dexterity.md) 在 16 primitive **object reposing** 预训练中已覆盖 lift / in-hand reorient / transport，再 post-train 对齐 FMB insert 等下游；真机 per-stage 表显示 vision-only 常在 Reorient 后失败，visuo-tactile 可维持至 Insert **8/10**（arXiv:2608.19182）。
- **世界模型循环状态（非想象 rollout）**：[WM-Craftnet](../entities/paper-wm-craftnet.md) 用 Dreamer 式 **WSM** 从 noisy 腕深+触觉+本体学习 **clean-depth 重建** 与 action-conditioned \(h_t\)，再喂 **PPO** 做无 object ID 多物体 z/x/y 轴旋转；Sharpa 真机 duck z 轴 **16.18 rad / 10/10**，九物体 WSM 先验可迁移四十九物体（CoRL 2026，arXiv:2609.07002；代码待发布）。
- **铰接物体 in-hand（抓稳 + 推动关节）**：[ArtManip](../entities/paper-artmanip.md) 针对 **内部 prismatic/revolute DoF** 而非整体 6D 重定向：物体 **自由漂浮在掌内**，策略须同时 **维持抓取稳定** 与 **驱动铰接关节**；程序化资产 + 功能抓取合成 → Teacher（特权状态 + 铰接 DR + 课程）→ Student 蒸馏；四类仿真 + 12 真机零样本（arXiv:2609.12498；[ArtGym](https://github.com/youngcv/artgym) 已开源）。

### 2. 轨迹优化 (Trajectory Optimization)
将重定向建模为带接触约束的最优控制问题。
- **代表作**：基于 Drake 的接触隐式优化（Contact-Implicit TO）。
- **机制**：显式地计算接触力和手指关节路径，物理意义明确但计算开销巨大。

### 3. 基于触觉的闭环控制
利用触觉传感器提供的接触点漂移信息进行实时补偿。
- **关联**：[触觉感知 (Tactile Sensing)](../concepts/tactile-sensing.md)。
- **外在灵巧对照**：[TacRefineNet](../entities/paper-tacrefinenet-tactile-grasp-refinement.md) 在欠驱动多指手上用 **腕部重定位 + regrasp**（而非纯指间 gait）把当前触觉图伺服到目标触觉图——与「不借外力的手内重定向」形成互补路径。
- **硬件 + 遥操作实例**：[VTAP Gripper](../entities/paper-vtap-gripper.md) 用三指协同演示球体重定向（约 \(\pm 15^\circ/\pm 20^\circ\)）与注射器手内对齐；主动掌参与柱塞按压等接触重分配（arXiv:2607.15448）。

## 工程实践

- **优先用大范围域随机化与物体形态多样化防止记忆单物体动力学：** 真机指尖摩擦和质量分布极易偏离 CAD 默认值，仿真阶段需随机化摩擦、质量与关节阻尼并用本体+历史或触觉对齐状态估计。
- **按可用硬件选纯指间 vs 借力外在灵巧：** 高自由度刚性/触觉手可走 PPO/UHAS 纯手内旋转；欠驱动多指手应结合腕部运动与外在接触重抓（参考 TacRefineNet）。

## 常见误区或局限

- **视觉遮挡与多指穿模：** 在真机高速手指交替阶段纯视觉位姿估计极易丢帧或漂移，往往需要视触觉融合或腕部传感器互补。
- **高维动作空间的真机寿命与样本成本：** 从零纯 RL 在真机直接训练容易加速腱绳磨损与掉物，通常必须借助大规模并行仿真或遥操作预训练初始化。

## 典型应用场景

- **拧螺丝**：在受限空间内，通过手指转动螺丝。
- **魔方还原**：展示极致的协调性。
- **工具使用**：调整手中工具（如锤子或笔）的抓握方向以适配后续任务。

## 关联页面
- [Allegro Hand 实体](../entities/allegro-hand.md)
- [Shadow Hand 实体](../entities/shadow-hand.md)
- [UHAS（统一手部动作空间）](./uhas-unified-hand-action-space.md)
- [Manipulation 任务](../tasks/manipulation.md)
- [Tactile Sensing (触觉感知)](../concepts/tactile-sensing.md)
- [TacRefineNet（论文实体）](../entities/paper-tacrefinenet-tactile-grasp-refinement.md) — 外在灵巧触觉精修对照
- [VTAP Gripper（论文实体）](../entities/paper-vtap-gripper.md) — 三指 + 主动掌的手内重定向 / singulation 硬件实例
- [TeleDexter（论文实体）](../entities/paper-teledexter.md) — 手–物 co-tracking 遥操作中的真机手内重定向
- [KaRMA（运动学指标）](../entities/paper-karma-hand-metric.md) — URDF rolling-pinch T/R/S 下界，补 task 前 kinematic 预筛
- [ADEPT（论文实体）](../entities/paper-adept-dexterity.md) — reposing RL 预训练覆盖 in-hand reorient，再 post-train 下游 insert
- [WM-Craftnet（论文实体）](../entities/paper-wm-craftnet.md) — WSM 循环视触觉状态 + PPO 鲁棒手内旋转（Sharpa，CoRL 2026）
- [ArtManip（论文实体）](../entities/paper-artmanip.md) — 类别级铰接物体手内操作：抓稳 + 推动内部关节（ArtGym 已开源）
- [Cross-modal Attention](../formalizations/cross-modal-attention.md)

## 参考来源
- OpenAI, et al. (2018). *Learning Dexterous In-Hand Manipulation*.
- Akkaya, I., et al. (2019). *Solving Rubik’s Cube with a Robot Hand*.
- [sources/papers/tacrefinenet_arxiv_2509_25746.md](../../sources/papers/tacrefinenet_arxiv_2509_25746.md) — TacRefineNet 外在灵巧 regrasp
- [sources/papers/teledexter_arxiv_2607_11481.md](../../sources/papers/teledexter_arxiv_2607_11481.md) — TeleDexter co-tracking 手内重定向
- [sources/papers/vtap_gripper_arxiv_2607_15448.md](../../sources/papers/vtap_gripper_arxiv_2607_15448.md) — VTAP 手内重定向与 singulation

## 推荐继续阅读

- [UHAS（统一手部动作空间）](./uhas-unified-hand-action-space.md)
- [Tactile Impedance Control](./tactile-impedance-control.md)
