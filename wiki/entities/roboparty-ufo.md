---
type: entity
tags: [humanoid, reinforcement-learning, unsupervised-rl, bfm, open-source, roboparty, teleoperation, mjlab]
status: complete
updated: 2026-07-14
related:
  - ./party-os.md
  - ../overview/roboparty-lab-party-os-technology-map.md
  - ./mjlab.md
  - ./paper-bfm-zero.md
  - ./paper-tech-humanoid-control.md
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../tasks/teleoperation.md
  - ./mimiclite.md
sources:
  - ../../sources/repos/roboparty_ufo.md
  - ../../sources/blogs/wechat_roboparty_lab_party_os_3_tools.md
  - ../../sources/sites/roboparty_lab_tech_humanoid_control.md
summary: "UFO 是 Roboparty 面向人形机器人的开源无监督强化学习控制开发框架：以 MJLab 为训练 backend，覆盖数据管线、BFM-Zero/TeCH 等表征研究与真机遥操部署，宣称 8×4090 不到 12 小时完成 BFM-Zero 训练。"
---

# UFO（Roboparty 无监督 RL 控制框架）

**UFO**（*Unsupervised RL Control Development Framework*）是 [Party OS](./party-os.md) 首批开源的 **无监督强化学习控制全栈框架**，覆盖训练基础设施、数据管线、算法研究与推理部署，目标降低无监督 RL 控制的研发门槛，使研究者能快速复现 SOTA、探索新行为表征、适配不同机器人平台，并实现训练到真机遥操的一体化开发。

> **命名辨析：** 本页 **UFO** 专指 Roboparty GitHub 仓库 [Roboparty/UFO](https://github.com/Roboparty/UFO)，与通用缩写无关。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| FB | Forward–Backward Representation | BFM-Zero 等使用的可 prompt 行为潜空间 |
| BFM | Behavior Foundation Model | 可复用、可适配的身体运控基座 |
| TeCH | Temporal Distance Modeling via Contrastive RL for Humanoid WBC | UFO 探索的新型行为表征 |
| WBC | Whole-Body Control | 全身协调控制 |

## 为什么重要

- **无监督线补位：** [MimicLite](./mimiclite.md) 聚焦 **监督跟踪**；UFO 覆盖 **无任务标签下的行为学习与 prompt 控制**，与 [BFM 学术地图](../overview/bfm-41-papers-technology-map.md) 的 forward-backward 表征线同向。
- **轻量训练 backend：** 采用 [mjlab](./mjlab.md) 而非重 Isaac Sim 栈，文称 8×4090 **不到 12 小时** 完成 BFM-Zero 训练，摆脱对大显存单卡依赖。
- **真机遥操开源：** 文称 **首次开源** 无监督 RL 控制的遥操作代码与完整验证方案，覆盖深蹲、跪地、打滚、跌倒恢复、抗外力扰动等复杂全身动作。

## 核心原理

### 1. Fast Training Infrastructure

| 配置 | 文内训练时长 | 说明 |
|------|--------------|------|
| 8×RTX 4090 | < 12 h（BFM-Zero） | MJLab backend；单卡/多卡并行 |
| 8×H200 | 6–8 h | 文称性能持续优于 BFM-Zero 原版 |

**机制要点：** 利用 mjlab 的 manager-based 模块化与 MuJoCo Warp GPU 并行，把无监督 RL 实验从「大显存单卡 + 长周期」推向 **多卡 consumer GPU 可复现**。

### 2. General and Extensible Framework

- **跨机器人形态：** 统一 codebase 可无缝适配不同人形平台，降低新本体迁移成本。
- **混合数据训练：** 支持多来源数据混合与灵活调度/配比。
- **高动态能力：** 文内展示侧手翻等高动态动作的无监督学习能力——须以公开评测为准。

### 3. New Representation Integration

| 表征 | 状态 | 说明 |
|------|------|------|
| **BFM-Zero（FB Representation）** | 已集成 | 对标 [paper-bfm-zero](./paper-bfm-zero.md) 无监督 RL + latent prompt 路线 |
| **TeCH** | 已集成 | [paper-tech-humanoid-control](./paper-tech-humanoid-control.md)：TLDR 对比时间距离表征；G1 上跟踪对标 SONIC(TER.)、真机抗扰与跌倒恢复 |

UFO 定位为 **多种行为表征的无监督学习统一实验平台**，而非单一算法仓库。

### 4. Teleoperation in the Real World

- **能力：** 真实机器人上完成深蹲、半蹲、跪地、打滚、跌倒恢复、抗外力扰动等。
- **意义：** 为无监督 RL 在真实场景的应用提供 **可复现参考实现**（代码 + 验证方案），填补学术 BFM 论文与工程遥操栈之间的缺口。

## 工程实践

| 阶段 | 建议 |
|------|------|
| 环境 | 安装 UFO + mjlab 依赖；确认 GPU 拓扑（8×4090 为文内基准） |
| 数据 | 利用框架内数据管线混合多来源 motion / 交互数据 |
| 训练 | 从 BFM-Zero 复现起步；再尝试 TeCH 等表征 ablation |
| 评测 | 仿真 prompt 跟踪、扰动恢复、技能多样性 |
| 真机 | 使用开源遥操与验证方案；与 MimicLite 部署层可分工（监督 vs 无监督） |

**仓库：** [github.com/Roboparty/UFO](https://github.com/Roboparty/UFO)

## 局限与风险

- **TeCH 发布形态：** 方法细节见 [Lab 成果页](https://lab.roboparty.com/outputs/tech) 与 [paper-tech-humanoid-control](./paper-tech-humanoid-control.md)；独立 arXiv 须以官方后续发布为准。
- **性能对比口径：** 「优于 BFM-Zero 原版」须明确任务、种子与硬件；避免跨仓库直接比 wall-clock。
- **无监督边界：** 技能发现质量依赖数据分布与表征设计；不等同于有参考的跟踪精度。
- **命名冲突：** 检索时注意与无关「UFO」项目区分，使用 `Roboparty/UFO` 全名。

## 关联页面

- [Party OS](./party-os.md)
- [MimicLite](./mimiclite.md) — 监督跟踪互补线
- [mjlab](./mjlab.md) — 训练 backend
- [BFM-Zero（论文实体）](./paper-bfm-zero.md)
- [TeCH（论文实体）](./paper-tech-humanoid-control.md)
- [行为基础模型](../concepts/behavior-foundation-model.md)
- [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md)
- [Teleoperation](../tasks/teleoperation.md)

## 参考来源

- [roboparty_ufo.md](../../sources/repos/roboparty_ufo.md)
- [wechat_roboparty_lab_party_os_3_tools.md](../../sources/blogs/wechat_roboparty_lab_party_os_3_tools.md)
- [roboparty_lab_tech_humanoid_control.md](../../sources/sites/roboparty_lab_tech_humanoid_control.md)

## 推荐继续阅读

- [UFO GitHub](https://github.com/Roboparty/UFO)
- [机器人论文阅读笔记：BFM-Zero](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/BFM-Zero__Promptable_Behavioral_Foundation_Model_for_Humanoid_Control_Using_Unsupervised_RL/BFM-Zero__Promptable_Behavioral_Foundation_Model_for_Humanoid_Control_Using_Unsupervised_RL.html)
- [mjlab 实体页](./mjlab.md)
