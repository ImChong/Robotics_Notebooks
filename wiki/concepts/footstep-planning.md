---
type: concept
tags: [locomotion, planning, footstep, contact-sequence, dcm, mpc]
status: stable
summary: "Footstep Planning 负责决定腿式机器人下一步踩哪里、踩多久，是地形感知和控制执行之间的关键桥梁。"
---

# Footstep Planning（步位规划）

**Footstep Planning** 是腿式机器人运动规划中的核心子问题：在给定运动目标和地形约束下，**决定每一步脚应该落在哪里、何时落下**。步位规划的输出是一个时序接触点序列（contact sequence），是后续质心规划、WBC 和控制器的上游输入。

## 一句话定义

> 步位规划回答的问题是："接下来每只脚应该踩哪里、踩多久"——它连接高层导航目标与底层接触物理约束，是腿式机器人闭环运动规划的核心枢纽。

---

## 核心问题分解

步位规划一般需要同时解决三个子问题：

| 子问题 | 含义 |
|-------|------|
| **Step location** | 每只脚在哪个位置落地（3D 坐标 + 方向） |
| **Step timing** | 何时抬脚、何时落地（步频 / 支撑相时长） |
| **Contact sequence** | 哪只脚先迈、哪些脚同时支撑（步态模式） |

三者相互耦合，联合规划比分开求解更优但计算代价更高。

---

## 主流方法

### 1. 基于 Capture Point / DCM 的反应式规划

- **核心思路**：维持 DCM 在当前支撑多边形内；当 DCM 将要超出时触发步位更新
- **优点**：实时性好（解析解），实现简单
- **局限**：只考虑当前步，不能提前规划多步序列
- **代表工作**：Pratt et al. *Capture Point: A Step toward Humanoid Push Recovery* (2006)

### 2. MPC-based 步位规划

- **核心思路**：在有限时域内联合优化质心轨迹 + 步位位置（通常 3-10 步）
- **变量**：步位坐标 + 支撑时长 + 质心轨迹
- **约束**：运动学可达性、地形碰撞避免、稳定性（ZMP 或 DCM）
- **代表工作**：Tonneau et al., Abe et al., Bledt et al. (MIT Cheetah3 接触规划)

### 3. 基于图搜索的步位规划

- **核心思路**：在离散化步位候选集上做图搜索（A\* / D\*）
- **适用场景**：离散地形（垫脚石、楼梯）
- **局限**：连续地形上候选空间爆炸，需启发式剪枝
- **代表工作**：DARPA Robotics Challenge 参赛团队的阶梯步行规划器

### 4. 端到端学习方法

- **核心思路**：用 RL 或模仿学习直接输出步位建议，后处理对齐到可行地形
- **优势**：可隐式处理接触不确定性，策略鲁棒
- **局限**：可解释性弱，对越障地形泛化能力取决于训练分布

---

## 与其他模块的关系

```
地形感知 / 导航目标
       ↓
  步位规划（本页）
       ↓
质心轨迹规划（DCM / MPC）
       ↓
  全身运动控制（WBC）
       ↓
   关节力矩输出
```

- **上游**：地形估计（高度图）、导航路点、步态模式选择（Gait Generation）、[Terrain Adaptation](./terrain-adaptation.md)
- **下游**：质心轨迹优化（LIP / VHIP）、WBC 接触约束设置

---

## 工程实现要点

1. **步位可达性约束**：需检查步位是否在髋关节运动学工作空间内（圆形近似 vs 精确椭圆）
2. **地形对齐**：步位落点需对齐地形法向量（脚掌平整接触），而非直接用 xy 平面投影
3. **在线重规划**：扰动发生时应在 10-50ms 内更新未来 1-2 步位置
4. **步态约束**：步位规划不能违反步态时序（如 trot 中对角脚同步约束）

---

## 参考来源

- [sources/papers/mpc.md](../../sources/papers/mpc.md) — ingest 档案（MPC 接触规划相关论文）
- [sources/papers/footstep_and_balance.md](../../sources/papers/footstep_and_balance.md) — ingest 档案（Kajita ZMP / Pratt CP / Koolen DCM / Herdt / Deits MICP）
- [sources/papers/contact_planning.md](../../sources/papers/contact_planning.md) — ingest 档案（MICP / CITO / Tonneau 综述）
- Pratt et al., *Capture the Flag: Instantaneous Capture Point for Humanoid Push Recovery* (2006) — CP 步位规划基础
- Bledt et al., *MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot* (2018) — 接触序列在线规划

---

## 关联页面

- [Capture Point / DCM](./capture-point-dcm.md) — 步位规划的稳定性依据
- [Locomotion](../tasks/locomotion.md) — 步位规划是 locomotion pipeline 的核心模块
- [Model Predictive Control](../methods/model-predictive-control.md) — MPC 框架实现多步预测规划
- [Balance Recovery](../tasks/balance-recovery.md) — 扰动后的紧急步位更新
- [Gait Generation](./gait-generation.md) — 步态模式是步位规划的上游输入
- [Terrain Adaptation](./terrain-adaptation.md) — 把地形感知转成可落脚区域与在线重规划

---

## 推荐继续阅读

- Tonneau et al., *An Efficient Acyclic Contact Planner for Multiped Robots*
- Bledt et al., *Contact Model Fusion for Event-Based Locomotion in Unstructured Terrains*

## 一句话记忆

> 步位规划做的是"脚往哪里踩"——它不管如何迈腿，只管每步的落点和时序，是腿式机器人把意图转化成可执行接触序列的关键一步。
