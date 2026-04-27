---
title: Motion Retargeting（动作重定向）
type: concept
status: complete
created: 2026-04-14
updated: 2026-04-18
summary: 将人类或动物参考动作映射到异构机器人骨架上，在保留运动风格和语义的同时满足机器人的关节限制和动力学约束。
---

# Motion Retargeting（动作重定向）

## 是什么

Motion Retargeting 是将一个运动序列（通常来自人类或动物）**转换为适合目标机器人执行的动作序列**的过程。

核心挑战：源（人/动物）和目标（机器人）往往有不同的：
- 骨架拓扑（关节数量/连接方式）
- 肢体比例
- 关节限制
- 质量分布与动力学

---

## 为什么重要

| 应用场景 | 重定向的作用 |
|---------|------------|
| 模仿学习参考轨迹 | 将 MoCap 数据转为机器人可执行轨迹，作为 RL 奖励或 BC 数据 |
| 全身遥操（Teleoperation） | 实时将人类动作映射到人形机器人 |
| AMP / ASE 风格先验 | 为 RL 策略提供运动风格参考 |
| 技能库建立 | 一次录制，多种机器人复用 |

---

## 主要方法

### 1. 基于关节角度映射（Joint-Space Retargeting）
最简单的方法：对每个关节直接建立角度映射（scale + offset）：
```python
θ_robot[i] = scale[i] * θ_human[i] + offset[i]
```
**优点**：实时，无优化；**缺点**：不考虑运动学差异，可能末端位置偏差大

### 2. 基于任务空间的优化（Task-Space IK）
以末端执行器（手、脚）的目标位置为约束，求解机器人的关节角：
```
minimize  ‖θ - θ_ref‖²
subject to: FK(θ) = p_target (末端位置约束)
            θ_min ≤ θ ≤ θ_max
            接触约束
```
工具：Pinocchio + TSID / Crocoddyl

### 3. 基于物理的重定向（Physics-Based Retargeting）
用物理仿真器验证重定向后的动作是否可执行（不摔倒）：
- 先在仿真中播放参考动作，用 PD 控制器追踪
- 收集可行段，过滤摔倒片段
- 可结合 RL 做后续精修（AMP 风格）

### 4. 深度学习重定向（Learning-Based）
- Encoder-Decoder 架构：将人类骨架 embedding，再 decode 到目标机器人
- 可跨模态（视频 → 机器人关节）
- 近年的工程趋势是先做几何重定向，再接一个下游物理一致化或 tracking 层，避免“姿态像但动力学不可执行”

---

---

## 重定向不等于控制策略：两层架构模式

在实际的人形机器人工程中，**重定向只是起点**。单纯的几何映射往往无法直接上机。

### 1. 运动学重定向层 (Kinematic Layer)
- **代表方法**：[GMR (General Motion Retargeting)](../methods/motion-retargeting-gmr.md)。
- **作用**：解决姿态、角度、关键点坐标的映射。
- **局限**：不能保证质心平衡、加速度连续性、接触力可行性以及力矩安全。容易出现脚部滑动或自碰撞。

### 2. 动力学一致化层 (Dynamical Layer)
- **作用**：在重定向轨迹的基础上，补足物理约束。
- **实现手段**：
    - **QP 优化 (如 HALO)**：通过约束二次规划，强制满足固定脚位置和地表接触约束，修正 Base 位姿漂移。
    - **RL Fine-tuning**：以重定向轨迹作为参考，通过 RL 在仿真中进行鲁棒性训练。
    - **WBC 跟踪**：将重定向轨迹作为 WBC 的任务目标，由底层控制器实时处理平衡与力矩限制。

---

## 关键技术问题

### 1. 骨架拓扑匹配
人类有 23 个主要关节，大多数人形机器人有 40-70 个（含手指）或更少（G1 = 43 DoF，无手指）。

匹配策略：
- **子树对齐**：人类骨架子树 ↔ 机器人骨架子树
- **忽略细节关节**：将人类手指聚合为"手端点"

### 2. 比例缩放（Scale）
人类（1.7m）vs 机器人（0.8m ~ 1.9m）体型不同：
- 末端轨迹按比例缩放：`p_robot = (L_robot / L_human) * p_human`
- 注意脚到地面的高度归一化

### 3. 接触保真度（Contact Consistency）
- 重定向时需要保留"哪只脚在地面"的接触相位
- 否则物理仿真下机器人会穿地或飞起

### 4. 关节限制满足
人类关节活动度（ROM）与机器人关节限制可能不同，需 clip + 后处理优化

---

## 工具与数据集

| 工具/数据集 | 用途 |
|------------|------|
| CMU MoCap Database | 大量人类动作捕捉数据 |
| AMASS | 统一格式的 MoCap 集合（50+ 数据集） |
| SMPL / SMPL-X | 人类体型参数化模型，便于重定向 |
| phc (Perpetual Humanoid Control) | PHC 的重定向+RL 框架 |
| OmniH2O | 人形机器人遥操+重定向框架 |

---

## 与 AMP / ASE 的关系

AMP 和 ASE 的核心是：从 MoCap 数据中提取运动风格先验，引导 RL 策略。

流程：
```
MoCap 数据 → Motion Retargeting → 机器人参考轨迹
→ 判别器训练（是否像参考） → RL 策略风格约束
```

Motion Retargeting 的质量直接决定 AMP 能学到多自然的动作。

---

## 参考来源

- Peng et al., *AMP: Adversarial Motion Priors for Style-Preserving Physics-Based Character Control* (2021) — AMP 中的 motion retargeting 应用
- Choi et al., *SMPL-X: Expressive Whole Body Pose Estimation* (CVPR 2019) — 人体参数化模型
- Liao et al., *Real-Time Motion Retargeting to Highly Varied User-Specific Hand Anatomies* (CHI 2019) — 异构骨架重定向
- **ingest 档案：** [sources/papers/teleoperation.md](../../sources/papers/teleoperation.md) — ALOHA / OmniH2O / UMI / AnyTeleop 遥操作系统
- **ingest 档案：** [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — ACT（CVAE 动作块预测）
- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — GMR 的总结强调了“运动学重定向之后还需要动力学一致化层”

---

## 关联页面

- [Imitation Learning](../methods/imitation-learning.md) — 重定向后的轨迹常作为模仿学习的参考数据
- [Locomotion](../tasks/locomotion.md) — locomotion 的风格先验来自重定向后的 MoCap 数据
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 全身操作任务需要手臂 + 腿部的联合重定向
- [Whole-Body Control](./whole-body-control.md) — WBC 执行重定向后的参考轨迹
- [Sim2Real](./sim2real.md) — 重定向数据质量影响真实机器人策略的泛化性
- [GMR (通用动作重定向)](../methods/motion-retargeting-gmr.md) — 基于运动学优化的重定向代表实现
