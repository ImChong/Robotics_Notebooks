---
type: concept
tags: [locomotion, terrain, perception, footstep-planning, sim2real]
status: complete
updated: 2026-05-30
summary: "Terrain Adaptation 指机器人根据地形感知结果调整步位、身体姿态和接触策略，以在不平整环境中保持稳定移动。"
related:
  - ../tasks/locomotion.md
  - ./footstep-planning.md
  - ./sim2real.md
  - ./privileged-training.md
  - ../tasks/balance-recovery.md
  - ../entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md
  - ../entities/paper-faststair-humanoid-stair-ascent.md
  - ../entities/dreamwaq-plus.md
sources:
  - ../../sources/papers/footstep_and_balance.md
  - ../../sources/papers/privileged_training.md
  - ../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md
  - ../../sources/papers/contact_planning.md
  - ../../sources/papers/e_sds_arxiv_2512_16446.md
---

# Terrain Adaptation（地形适应）

**Terrain Adaptation**：让腿式或人形机器人根据地形感知结果，动态调整落脚点、身体姿态、接触时序和控制参数，从而在楼梯、碎石、草地、台阶和坡面上稳定行走。

## 一句话定义

地形适应不是“看见地面”，而是把看见的地形真正转成可执行的接触决策。

## 为什么重要

平地 locomotion 可以依赖固定步态和简化模型，但一旦进入真实世界：
- 可落脚区域离散且不规则
- 摩擦和高度变化引入失稳风险
- 感知误差会直接转化为踩空或绊倒

因此地形适应是 locomotion 从实验室走向真实环境的关键门槛。

## 典型闭环

```text
高度图 / 点云 / 触觉估计
        ↓
   可行落脚区域提取
        ↓
   步位 / 接触序列规划
        ↓
  MPC / RL / WBC 执行调整
```

## 常见信息来源

### 1. 高度图（Height Map）
最常见的结构化表示。把局部地形编码成栅格高度，可直接供 teacher policy、MPC 或步位规划器使用。

### 2. 点云 / 深度图
信息更丰富，但噪声更大、实时处理代价更高，常需先转成局部可行接触区域。

### 3. 接触反馈
足端是否滑动、接触是否建立、法向力是否异常，都是补救感知误差的重要信号。

## 主要策略路线

| 路线 | 做法 | 优点 | 局限 |
|------|------|------|------|
| 传统规划 | 感知 + footstep planning + MPC/WBC | 可解释、约束清晰 | 感知和规划耦合复杂 |
| 特权训练 | teacher 用高度图，student 蒸馏到本体感知 | sim2real 友好 | teacher/student 设计复杂 |
| 端到端 RL | 直接输入高度图/点云预测动作 | 反应式强 | 对训练分布依赖高 |
| 多模态点云 RL（DreamWaQ++） | 分层 $SE(3)$ 点云记忆 + PointNet + 本体 Mixer，单阶段非对称 AC | 障碍前瞻、传感器无关、OOD 本体回退 | 训练重、公开页暂无同署官方代码仓 |
| 环境感知自动奖励（E-SDS） | VLM 读地形统计 + 行为分解生成调用高度图/LiDAR 的 Python 奖励 | 跨四类地形少手工调参；楼梯下降为分水岭 | 每地形专用策略、仅仿真、首轮仍依赖 prompt 工程 |

### 近期案例：E-SDS 的环境感知奖励合成

[E-SDS（arXiv:2512.16446）](../entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md) 把 **地形适应** 从「控制器读传感器」前移到 **奖励设计阶段**：Environment Analysis Agent 在目标地形上跑千机短 rollout，统计 **缺口率、障碍密度、崎岖度**，与 SUS 行为分解一并喂给 VLM，生成显式调用 **27×21 高度栅格 + 144 线 LiDAR** 的奖励代码；再经双候选 PPO + 反馈迭代精炼。在 Isaac Lab + Unitree G1 上，相对手工 13 项感知基线 **速度跟踪误差降 51.9–82.6%**，且 **仅该方法完成 12 cm 台阶下降**。

## 与其他页面的关系

- [Footstep Planning](./footstep-planning.md) 决定“下一步踩哪里”。
- [Locomotion](../tasks/locomotion.md) 关注更完整的移动任务。
- [Privileged Training](./privileged-training.md) 展示了复杂地形上 teacher 用高度图、student 用本体感知的经典方案。
- [DreamWaQ++](../entities/dreamwaq-plus.md) 把 **3D 点云** 与 **本体历史** 在单阶段 RL 中融合，是四足 **点云地形适应** 的代表实现。
- [Sim2Real](./sim2real.md) 强调地形感知和真实传感器偏差是迁移痛点。

## 常见误区

- **误区 1：腿足机器人不需要地形感知。**
  在结构化平地上也许可勉强成立，但在户外和障碍环境里通常不现实。
- **误区 2：有高度图就自动等于稳定行走。**
  还需要把感知结果转成步位、姿态和接触约束。
- **误区 3：地形适应只属于高层规划。**
  实际上它影响从 perception 到控制执行的全链路。

## 参考来源

- [sources/papers/footstep_and_balance.md](../../sources/papers/footstep_and_balance.md) — 步位规划、DCM 与不平地形步行基础
- [sources/papers/privileged_training.md](../../sources/papers/privileged_training.md) — ANYmal 高度图 teacher / proprioception student 经典路线
- [sources/papers/contact_planning.md](../../sources/papers/contact_planning.md) — 不平整地形接触区域与多步接触规划
- [sources/papers/e_sds_arxiv_2512_16446.md](../../sources/papers/e_sds_arxiv_2512_16446.md) — E-SDS 环境感知 VLM 奖励合成
- [sources/papers/dreamwaq_plus_arxiv_2409_19709.md](../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md) — DreamWaQ++ 多模态点云四足 loco

## 关联页面

- [Locomotion](../tasks/locomotion.md)
- [Footstep Planning](./footstep-planning.md)
- [野外机器人排障指南](../queries/field-robotics-troubleshooting.md) — 应对非结构化地形下的感知与平衡失效
- [Privileged Training](./privileged-training.md)
- [Sim2Real](./sim2real.md)
- [Balance Recovery](../tasks/balance-recovery.md)

## 推荐继续阅读

- Lee et al., *Learning to Walk in Difficult Terrain*
- Deits & Tedrake, *Footstep Planning on Uneven Terrain with MICP*
- [Query：从零训练人形机器人 RL 策略的完整 checklist？](../queries/humanoid-rl-cookbook.md)
