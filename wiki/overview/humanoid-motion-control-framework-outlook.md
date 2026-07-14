---
type: overview
tags: [humanoid, motion-control, architecture, vla, bfm, navigation, planning, controller]
status: complete
updated: 2026-07-14
related:
  - ./humanoid-motion-control-know-how-technology-map.md
  - ./humanoid-motion-control-trends.md
  - ../queries/control-architecture-comparison.md
  - ../comparisons/wbc-vs-rl.md
  - ../concepts/behavior-foundation-model.md
  - ../methods/sonic-motion-tracking.md
  - ../concepts/motion-retargeting.md
sources:
  - ../../sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md
  - ../../sources/papers/humanoid_motion_control_know_how.md
summary: "飞书 Know-How「技术框架路线展望」：传统 Navigation–Planning–Controller–Estimator 栈 vs 端到端 RL，并给出五条未来路线思考（遥操作+VLA、上下肢分离、解耦 N+P+C、轮足、Physics-first）。"
---

# 人形机器人技术框架路线展望

> 编译自 RoboParty 飞书 Know-How 全文 §「人形机器人技术框架路线展望」；作者强调**一家之言**，供路线讨论而非定论。

## 一句话定义

人形运控的长期架构可能在 **经典分层（导航–规划–控制–估计）** 与 **端到端学习** 之间摆动；真正可规模化的路线取决于数据效率、动力学耦合与是否坚持全身模型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 传统栈高实时规划/控制 |
| VLA | Vision-Language-Action | 上层语言–视觉–动作大模型 |
| VLM | Vision-Language Model | 规划层可能的人类数据接口 |
| RL | Reinforcement Learning | 端到端与 Controller 训练主范式 |
| BFM | Behavior Foundation Model | 通用身体 Controller 候选 |
| TO | Trajectory Optimization | Planning 层轨迹优化 |

## 传统分层框架

经典腿足/人形栈（十余年稳定形态）：

| 模块 | 职责 |
|------|------|
| **Navigation** | 符合运动学约束的路径与可通行区域 |
| **Planning** | 轨迹优化（TO）：落脚点、平滑速度/加速度、全身协调参考 |
| **Controller** | 动力学可行控制输入（关节力矩）；传统研究主战场 |
| **Estimator** | 质心速度/位姿、角速度等不可直接观测状态 |

公开文献中框架最清晰者之一为 **ETH 分层优化 + WBC**（Humanoids 2016；TRO 2023 perceptive NMPC 等）。强化学习兴起后，ETH 等团队也走向 **端到端 blind → perceptive → navigation**，且多数场景效果优于旧栈。

## 五条未来路线思考（作者原文归纳）

### 思考一：端到端 → 全身遥操作采集 → VLA

- 波士顿动力路线：MPC 全身遥操作 → 采集数据 → VLA 泛化。
- 其他团队可用 **TWIST / GAE / SONIC** 类模仿+RL 绕开 MPC 遥操作，但 **in-the-air 动作** 仍难（腿部动力学突变、解不连续）。
- **瓶颈**：即便 Mimic 遥操作可行，**人形数据采集效率** 远低于机械臂 VLA。

### 思考二：上下肢分离 → 上肢机械臂 VLA

- 下肢小模型保鲁棒行走/越障，上肢走机械臂 VLA，复用臂域数据。
- **理论矛盾**：全身动力学耦合；控制层分离 ≠ 模型可分离。
- 分支困境：牺牲空间自由度（偏工厂） vs 让双腿 cover 一切（仍需模仿/无监督 BFM，面临分布偏移与突变动作）。

### 思考三：解耦 Navigation + Planning(VLM) + Controller(RL)

- RL 与 MPC 同属 OCP 家族；RL 是更强的**非凸**求解器，且建模**分布**。
- **Blind 模仿 + 大数据**（SONIC/GAE/TWIST）可优先利用现成**人体本体数据**；带视觉的人体数据稀缺得多。
- 设想：**Controller** 用 RL 训通用跟踪器；**Planning** 用 VLM 生成 SMPL 人类参考 → **retarget** → 机器人参考；**Navigation** 独立 VLN → 形成 **N + P + C** 解耦，降低机器人侧数据需求。

### 思考四：轮足人形

- 轮足非仿生但兼顾速度与越野；对人形而言若只拼越野不如四轮足。
- 通用 Whole-body 数据若仍依赖遥操作，轮足人形收集效率问题依旧；**仿人与工程平衡**（如 G1 类平台）仍可能是性价比路线。

### 思考五：若通用基于 LLM，具身需要更多 Physics

- 必须与物理世界交互 → **仿真更物理** 或 **算法框架更物理**；懂世界才能走向真具身智能。

## 工程实践

- 选型时先回答：数据从 **人形遥操作 / 人体 retarget / 无监督 BFM** 哪条来；Controller 是否已通用。
- 读 ETH Humanoids 2016 + Grandia TRO 2023 对照本站 [Centroidal NMPC+WBC](../methods/centroidal-nmpc-wbc-stack.md)、[SONIC](../methods/sonic-motion-tracking.md)。

## 局限与风险

- 作者明确「不能得出明确结论」；五条思考含未验证判断（如 blind 大数据优于 perception end2end）。
- 与具体产品路线（Boston Dynamics、Figure、Unitree 生态）需交叉验证。

## 关联页面

- [Know-How 技术地图](./humanoid-motion-control-know-how-technology-map.md)
- [发展趋势](./humanoid-motion-control-trends.md)
- [Motion Retargeting](../concepts/motion-retargeting.md)
- [BFM](../concepts/behavior-foundation-model.md)

## 参考来源

- [飞书 Know-How 全文](../../sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md)
- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)

## 推荐继续阅读

- Bellicoso et al., Humanoids 2016（ETH 分层 WBC）
- Grandia et al., TRO 2023（perceptive NMPC）
