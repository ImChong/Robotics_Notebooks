---
type: task
tags: [loco-manipulation, humanoid, whole-body, manipulation, locomotion]
status: complete
summary: "Loco-Manipulation 关注机器人边移动边操作的全身协调问题。2025-2026 年的趋势正从分层控制扩展到生成模型、VLA 与触觉增强的统一全身感知控制。"
updated: 2026-04-29
sources:
  - ../../sources/papers/teleoperation.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/humanoid_touch_dream.md
  - ../../sources/repos/awesome-humanoid-robot-learning.md
---

# Loco-Manipulation (移动操作)

**移动操作（Loco-Manipulation）**：机器人在运动（行走/移动）的同时执行操作任务（抓取/推动/交互），要求同时具备行走能力和上肢操作能力。

## 一句话定义

让机器人**边走边动手**——不是先停下来再操作，而是行走和操作在动力学层面高度耦合、在控制层面完全协调。

## 核心挑战

### 1. 全身动力学耦合
手臂运动会干扰质心平衡，步态振动会干扰操作精度。**独立优化行走和操作再简单合并通常无法实现复杂动作。**

### 2. 接触丰富与多约束
涉及足端地形接触与末端物体接触的并发管理，接触序列的规划空间巨大。

### 3. 高动态与精细度平衡
在进行跑酷或球类运动（高动态）的同时，需要保持末端对物体（球拍、托盘）的精密控制。

## 技术路线演进 (2024-2026)

### 1. 经典分层路线 (Modular/Hierarchical)
- **HLC (高层控制)**：VLA 或 RL 给出末端轨迹目标。
- **LLC (底层控制)**：WBC + MPC 负责全身执行。
- **代表作**：Humanoid Hanoi (2026), HiWET (2026)。

### 2. 统一生成式路线 (Unified Generative)
- **核心**：利用扩散模型（Diffusion）或概率流（Flow Matching）生成物理可行的全身运动序列。
- **特点**：天然支持多模态，能够生成极其自然的全身协调动作。
- **代表作**：SafeFlow (2026), DreamControl (2025), BeyondMimic (2025)。

### 3. 基础模型路线 (Foundation Models / VLA)
- **核心**：将视觉、语言和全身动作（Whole-body Actions）映射到统一的 Token 空间。
- **趋势**：强调从互联网规模的人类视频中学习，而非依赖昂贵的机器人演示。
- **代表作**：Ψ₀ (2026), WholeBodyVLA (2025), SENTINEL (2025)。

### 4. 残差与自适应学习 (Residual & Adaptive)
- **核心**：在高层规划器输出的基础上，通过轻量级 RL 学习补偿项（Residual），以处理复杂地形或扰动。
- **代表作**：SteadyTray (2026), ResMimic (2025), SEEC (2025)。

### 5. 触觉增强的行为克隆路线 (Touch-Aware BC)
- **核心**：把接触信号纳入全身操作策略训练，而不是只依赖视觉与本体感受。
- **代表作**：[HTD](../methods/humanoid-transformer-touch-dreaming.md) (2026) 使用 lower-body controller 保持全身稳定，并在模仿学习中预测未来手部力和触觉 latent，提升插入、折叠、工具使用和端杯移动等接触丰富任务的成功率。

### 6. 反向层级架构 (MPC-over-RL)
- **核心**：底层使用通用的 RL WBC 策略（如 Relic）提供稳定的运动基座；高层使用基于采样的 MPC（如 CEM）在底层策略的命令空间内进行在线规划。
- **代表作**：[Sumo](../methods/sumo.md) (2026) 实现了 Spot 和 G1 操纵比自身更重、更大的物体（如扶起轮胎、拖拽大型护栏）。

## 重点应用领域

| 领域 | 典型任务 | 代表研究 |
|------|---------|---------|
| **家务/生活** | 开门、端托盘、整理箱子 | BEHAVIOR Robot Suite (2025), StageACT (2025) |
| **体育竞技** | 网球、羽毛球、足球、滑板 | LATENT (2026), HITTER (2025), HUSKY (2026) |
| **极端环境** | 跑酷、徒步、复杂室内穿越 | Perceptive Humanoid Parkour (2026), Hiking in the Wild (2026) |
| **人类协作** | 共同搬运物体、人机交互 | Human-Humanoid Interaction (2026) |

## 关联页面

- [Humanoid Locomotion](./humanoid-locomotion.md)
- [Manipulation](./manipulation.md)
- [Diffusion-based Motion Generation](../methods/diffusion-motion-generation.md) — 2026 年的主流高层运动生成技术
- [Whole-Body Control](../concepts/whole-body-control.md)
- [VLA](../methods/vla.md)
- [Teleoperation](./teleoperation.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md)

## 参考来源

- [[sources/repos/awesome-humanoid-robot-learning]] — 持续更新的人形机器人学习论文集
- [ULTRA survey](./ultra-survey.md) — 统一多模态 loco-manipulation 综述 (2026)
- [arXiv 2603.23983](https://arxiv.org/abs/2603.23983), *SafeFlow: Real-Time Text-Driven Humanoid Whole-Body Control* (2026)
- **ingest 档案：** [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — 包含 ACT / Diffusion Policy 等基础
- **ingest 档案：** [sources/papers/teleoperation.md](../../sources/papers/teleoperation.md) — HOMIE / ALOHA / OmniH2O 
- **ingest 档案：** [sources/papers/humanoid_touch_dream.md](../../sources/papers/humanoid_touch_dream.md) — HTD / Touch Dreaming 触觉增强人形移动操作

## 一句话记忆

> Loco-Manipulation 正在从“行走 + 操作”的简单叠加，演变为基于生成式模型、VLA 与触觉增强行为克隆的全身统一感知控制，是实现人形机器人从实验室走向通用场景的关键瓶颈。
