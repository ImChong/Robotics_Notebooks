---
type: task
tags: [teleoperation, manipulation, loco-manipulation, data-collection, humanoid]
status: complete
summary: "Teleoperation 让人类通过远程接口直接操作机器人，是数据采集和复杂任务执行的重要桥梁。"
updated: 2026-05-14
sources:
  - ../../sources/papers/teleoperation.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/humanoid_touch_dream.md
---

# Teleoperation（遥操作）

**一句话定义**：操作员通过外部设备实时远程控制机器人完成任务，同时采集高质量示范数据用于后续策略学习。

## 为什么重要

遥操作是当前人形机器人获取**高质量演示数据**最可行的路线：

- 真实世界的操作任务（擦桌、开门、折衣物）难以在仿真中自动生成示范
- 大规模 RL 探索在真机上代价过高（执行器损耗、安全风险）
- 人类示范数据 + 模仿学习（BC/ACT/Diffusion Policy）已在多个任务上超过纯 RL

> "You can't imitate what you don't have." — 数据飞轮的起点是高质量遥操作系统。

## 关键挑战

### 1. 运动映射（Motion Retargeting）
人和机器人形态不同：
- 关节 DOF 不同（人手 21 DOF vs 机械手 6-12 DOF）
- 工作空间不同（机器人手臂更短/更长）
- 动力学不同（机器人惯性大、控制带宽有限）

解决方案：基于运动学约束的 IK 重定向 / 学习映射（端到端训练）

### 2. 延迟补偿
端到端延迟（感知→传输→执行）通常 50-200ms，影响精细操作：
- 降低带宽（关节命令压缩、关键帧插值）
- 预测性显示（基于模型预测未来状态）
- 减少不必要的处理步骤（低延迟 ROS2 / EtherCAT）

### 3. 数据质量与一致性
操作员疲劳、环境变化、设备噪声都影响示范质量：
- 失败示范过滤（自动 / 人工标注）
- 示范标准化（速度 / 姿态归一化）
- 多操作员数据融合策略

### 4. 双臂协调
全身遥操作需要同时控制移动基座 + 双臂：
- 基座运动 vs 手臂运动的优先级
- 碰撞回避（基座移动时避免手臂碰障碍物）
- 双臂时序协调（抓取一件物品时另一臂稳定）

## 主流遥操作系统

| 系统 | 机器人 | 输入设备 | 数据规模 | 特点 |
|------|--------|---------|---------|------|
| ALOHA（Stanford 2023） | 4 臂台式 | Leader Arms | ~50 任务 | 低成本（$20K），精细操作 |
| OmniH2O（CMU/Tsinghua 2024） | Unitree H1/G1 | VR + 手套 | 全身遥操作 | 全身 DOF 控制，含移动基座 |
| HTD（CMU/Bosch 2026） | 人形 + 灵巧手 | VR + 摇杆 + 分布式触觉 | 5 个真实接触丰富任务 | LBC 稳定下肢，VR 采集上身/手部示范，并同步手部力与触觉 |
| UMI（Stanford 2024） | 通用 | GoPro + 夹爪 | 可扩展 | 无需专用机器人，数据可迁移 |
| AnyTeleop（UCB 2023） | 多平台 | RGB 相机 | 通用 | 无传感器手套，仅视觉输入 |
| GELLO（Berkeley 2023） | 多 UR/Franka | Leader Arms | 低成本 | 低成本版 ALOHA |

## 全身人形：视频 / VR 条件 + 低层 tracking（工程参考）

NVIDIA **SONIC** 项目页（[GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/)）把遥操作与 **规模化 motion tracking policy** 放在同一套 **统一 token / 控制接口** 下展示：人体视频经 **GEM** 估计姿态后实时跟踪；VR 含 **头 + 双手三点** 驱动上身并由 **运动学规划器** 补全下身，以及 **全身 VR 追踪** 模式。知识库方法页见 [SONIC（规模化运动跟踪人形控制）](../methods/sonic-motion-tracking.md)。

## 遥操作到策略学习 Pipeline

```
遥操作采集
    ↓ 示范数据（obs, action 序列）
数据预处理（过滤/标准化）
    ↓
模仿学习训练
  ├─ Behavior Cloning（BC）：直接复制
  ├─ ACT：Action Chunking + CVAE
  ├─ Diffusion Policy：扩散生成动作序列
  └─ IL + RL fine-tune：提升鲁棒性
    ↓
部署（真机 / 仿真验证）
```

## 评估指标

| 指标 | 说明 |
|------|------|
| 任务成功率 | N 次尝试中完成任务的比例 |
| 完成时间 | 完成任务的平均时长（越短越好） |
| 操作员上手时间 | 新操作员达到可接受质量所需的训练时间 |
| 端到端延迟 | 操作员输入到机器人执行的延迟 |
| 示范数据效率 | 用多少条示范可以训练出成功率 ≥ X% 的策略 |

## 参考来源

- **ingest 档案：** [sources/papers/teleoperation.md](../../sources/papers/teleoperation.md) — ALOHA / OmniH2O / UMI / AnyTeleop
- **ingest 档案：** [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — ACT / Diffusion Policy（遥操作数据的下游学习方法）
- **ingest 档案：** [sources/papers/humanoid_touch_dream.md](../../sources/papers/humanoid_touch_dream.md) — HTD 的 VR 全身遥操作、LBC 和触觉同步采集系统
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (RSS 2023) — ALOHA
- He et al., *OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation* (2024)
- [机器人论文阅读笔记：OmniH2O](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_High_Impact_Selection/OmniH2O_Universal_Whole-Body_Teleoperation/OmniH2O_Universal_Whole-Body_Teleoperation.html)

## 关联页面

- [Loco-Manipulation](./loco-manipulation.md) — 遥操作在移动操作中的应用
- [Motion Retargeting](../concepts/motion-retargeting.md) — 人类动作到机器人动作的映射
- [Imitation Learning](../methods/imitation-learning.md) — 遥操作数据的学习方法
- [Humanoid Transformer with Touch Dreaming](../methods/humanoid-transformer-touch-dreaming.md) — 使用触觉遥操作数据训练接触感知策略
- [Diffusion Policy](../methods/diffusion-policy.md) — 遥操作数据训练的扩散策略
- [Manipulation](./manipulation.md) — 操作任务整体视角
- [Query：操作演示数据采集指南](../queries/demo-data-collection-guide.md) — 遥操作采集数据的实操指南

## 推荐继续阅读

- [ALOHA 项目主页](https://mobile-aloha.github.io/) — Stanford 双臂遥操作系统
- [OmniH2O 论文](https://arxiv.org/abs/2406.08858) — 人形全身遥操作
- [ACT 论文](https://arxiv.org/abs/2304.13705) — 遥操作数据 + 动作块学习
