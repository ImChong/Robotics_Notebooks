---
type: entity
tags:
  - paper
  - world-action-models
  - mobile-manipulation
  - loco-manipulation
  - flow-matching
  - latent-action
  - mixture-of-transformers
  - dream-forcing
  - alibaba
status: complete
updated: 2026-07-24
arxiv: "2607.00678"
related:
  - ../concepts/world-action-models.md
  - ../tasks/loco-manipulation.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../methods/generative-world-models.md
  - ../entities/paper-motionwam-humanoid-loco-manipulation-wam.md
  - ../entities/paper-dit4dit-video-action-model.md
  - ../entities/qwen-robot-manip.md
  - ../overview/loco-manip-contact-category-05-vla-world-models.md
  - ../overview/loco-manip-161-category-04-generative-language-trajectory.md
sources:
  - ../../sources/papers/abot_m05_arxiv_2607_00678.md
summary: "ABot-M0.5（arXiv:2607.00678）：移动操作专用 WAM——帧级 latent action 桥接 video latent 与执行控制，双层 D-MoT 解耦移动/操作子空间，Dream Forcing 在自生成视频上对齐逆动力学；RoboCasa365 平均 46.6%（+Condensed Memory）、LIBERO 99.4%、LIBERO-Plus WAM 对照 83.4%。"
---

# ABot-M0.5（Unified Mobility-and-Manipulation · World Action Model）

**ABot-M0.5**（*ABot-M0.5: Unified Mobility-and-Manipulation World Action Model*，arXiv:2607.00678，AMAP CV Lab）提出：移动操作要在 **时间粒度、动作空间结构、训练–推理条件** 三层对齐，才能把 WAM 从 **桌面/静止操纵** 可靠扩展到 **长程导航 + 精细接触** 的联合 rollout。

## 一句话定义

**用 Video → Latent Action → Action 三级级联、双层 MoT 解耦移动与操作，并以 Dream Forcing 在自 dreamed 视频上训练逆动力学**——把 WAM 的结构瓶颈显式对准移动操作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合未来观测/潜变量与动作生成的具身策略 |
| D-MoT | Dual-level Mixture-of-Transformers | 模态级 + 动作级（move/manip）解耦的 MoT |
| CFM | Conditional Flow Matching | 视频/latent action/动作三阶段的统一生成目标 |
| VLA | Vision-Language-Action | 本文主要对照：反应式、缺显式世界建模 |
| ALAM | （论文采用之 latent action 框架） | 从帧对提取具身无关局部运动表征 |
| OXE | Open X-Embodiment | 预训练异构机器人数据之一 |

## 为什么重要

- **移动操作是 WAM 的「结构试金石」：** 粗视频 chunk、纠缠的导航–操作动作空间、以及 **GT 未来监督逆动力学** 会在长程 rollout 中放大误差；论文把这三点形式化为 **三层对齐原则**，而非仅堆模型规模。
- **与 [MotionWAM](./paper-motionwam-humanoid-loco-manipulation-wam.md) 形成对照轴：** 二者均属 **Joint WAM + flow matching**，但 MotionWAM 聚焦 **人形全身 loco-manip + SONIC 统一 token + 单次 Video DiT 前向实时闭环**；ABot-M0.5 聚焦 **轮式/移动底座 + 臂操纵** 的 **latent action 桥接粒度** 与 **Dream Forcing rollout 对齐**，并在 **RoboCasa365** 移动操作榜建立强基线。
- **Dream Forcing 针对 WAM 特有 exposure bias：** 相对 Teacher Forcing / Diffusion Forcing，在 **自生成 $\hat{z}, \hat{m}$** 上训动作，**5k 步** 即可把 RoboCasa365 Target atomic-seen 从 **67.55%** 拉到 **70.56%**，而继续 Teacher Forcing 同期 **降至 66.78%**。
- **评测覆盖面广：** 同时报告 **移动操作（RoboCasa365）**、**双臂操纵（RoboTwin 2.0）**、**桌面组合（LIBERO / LIBERO-Plus 零样本）** 与 **真机长程**，说明改进不只绑定单一 benchmark。

## 核心结构与方法

| 模块 | 作用 |
|------|------|
| **Video 路径（Wan2.2 骨干）** | 多视角观测 → 3D VAE latent $z_{t+1}$；自回归预测未来视频动力学 |
| **Latent action $m_t$** | 冻结 **ALAM** encoder 从 $(I_t, I_{t+1})$ 提取 **帧级、具身无关** 运动意图，桥接粗 video latent 与细控制 |
| **Action 路径（D-MoT）** | 将 $a_t$ 拆为 **mobility** $a^{\mathrm{move}}$ 与 **manipulation** $a^{\mathrm{manip}}$，子塔独立 FFN/头，共享注意力协调 |
| **Dream Forcing** | Phase A 少步去噪得 $\hat{z}_{t+1}, \hat{m}_t$；Phase B 仅在 dreamed 条件上优化动作 CFM |
| **渐进训练** | 世界模型预训练 → latent action 预训练 → SFT1（GT 未来联合）→ SFT2（Dream Forcing） |

### 流程总览

```mermaid
flowchart TB
  subgraph align [三层对齐]
    T[时间粒度\n帧级 latent action]
    A[动作空间\nD-MoT move/manip]
    R[训练–推理\nDream Forcing]
  end
  subgraph cascade [三级级联]
    OBS[多视角观测 + 语言 l]
    OBS --> Z[Video latent z_{t+1}\n世界建模 CFM]
    Z --> M[Latent action m_t\n运动抽象 CFM]
    M --> ACT[可执行动作 a_t\nmove + manip CFM]
  end
  align --> cascade
  subgraph deploy [部署]
    ACT --> ROBOT[移动底座 + 机械臂\n仿真 / Agilex Piper 真机]
  end
```

### 与相邻 WAM 的分界

| 维度 | 典型 Joint WAM（如 Motus / Fast-WAM） | ABot-M0.5 |
|------|--------------------------------------|-----------|
| **时间粒度** | 粗 video chunk → 动作 | **显式帧级 latent action 桥接** |
| **动作空间** | 单一纠缠向量 | **D-MoT 解耦 move / manip** |
| **逆动力学监督** | GT 或联合扩散噪声 | **SFT2：自 dreamed 视频条件** |
| **主评测语境** | 桌面/通用操纵 | **RoboCasa365 移动操作 + 多榜** |

## 实验要点（索引级）

| 轴 | 报告口径（以论文为准） |
|----|------------------------|
| **移动操作** | RoboCasa365 pretrain 平均 **40.4%**；+Condensed Memory **46.6%**；Target 100% **54.2%** |
| **双臂操纵** | RoboTwin 2.0 平均 **94.10%**（Clean **94.0%** / Randomized **94.2%**） |
| **桌面组合** | LIBERO 平均 **99.4%**；LIBERO-Plus 零样本 Total **83.4%**（WAM 对照领先） |
| **真机** | Agilex Piper 单臂；每任务 **50** demos；Peg Cylinder **70%** / 过程 **96%** |
| **关键消融** | 三阶段 latent action **94.0%** vs 直接 video→action **87.60%**（RoboTwin Clean） |
| **代码** | [amap-cvlab/ABot-Manipulation](https://github.com/amap-cvlab/ABot-Manipulation)（M0.5 权重 **coming soon**，2026-07-01） |

| 机构 | AMAP CV Lab（高德 CV Lab，阿里巴巴） |
|------|--------------------------------------|
| arXiv | [2607.00678](https://arxiv.org/abs/2607.00678) |
| 前序 | [ABot-M0](https://arxiv.org/abs/2602.11236)（VLA foundation，同仓库已开源训练/权重） |

## 结论

**用帧级 latent action、D-MoT 解耦与 Dream Forcing 三层对齐，把 WAM 从桌面操纵可靠扩展到移动操作。**

1. **三层对齐优先于堆骨干** — 关键是时间粒度（帧级 ALAM latent action）、动作空间（D-MoT move/manip）与训练–推理（Dream Forcing），而非更大 Wan  alone。
2. **Dream Forcing 对冲 exposure bias** — SFT2 在自 dreamed 视频条件上训逆动力学；文中 5k 步把 Target atomic-seen 从 67.55% 拉到 70.56%，同期继续 Teacher Forcing 降至 66.78%。
3. **评测以 RoboCasa365 为主战场** — pretrain 平均 40.4%，+Condensed Memory 46.6%，Target 100% 54.2%；LIBERO 99.4% 不能替代移动操作结论。
4. **三级级联可消融验证** — RoboTwin Clean 上三阶段 latent action 94.0% vs 直接 video→action 87.60%。
5. **落地注意权重与平台** — M0.5 全量权重仍 coming soon；真机以 Agilex Piper 单臂为主，与人形全身 loco-manip 平台不同。

## 常见误区或局限

- **误区：** 把 ABot-M0.5 等同于「更大 Wan 视频模型」；关键是 **latent action 桥接 + D-MoT + Dream Forcing** 的结构对齐，而非骨干参数 alone。
- **误区：** 认为 LIBERO 99.4% 说明只需桌面数据；**RoboCasa365** 与真机长程任务才是移动操作主战场，且 **预训练 vs 直接 SFT** 差距可达 **31.2%**（Target 10%）。
- **局限：** M0.5 全量权重截至技术报告仍 **coming soon**；真机验证以 **单臂** 为主，与 **人形全身 loco-manip**（如 MotionWAM）平台不同；**Condensed Memory** 扩展仅在文中预告。

## 与其他工作对比

| 工作 | 关系 |
|------|------|
| **[MotionWAM](./paper-motionwam-humanoid-loco-manipulation-wam.md)** | 同人形/WAM 脉络；人形 **SONIC 统一 token + 实时单次 Video DiT 前向** vs 移动底座 **latent action + Dream Forcing** |
| **[DiT4DiT](./paper-dit4dit-video-action-model.md)** | 同 **双路径 flow matching + Cosmos/Wan 系视频先验** 思想；DiT4DiT 偏 **VAM 联合训练**，ABot-M0.5 偏 **移动操作三层对齐** |
| **Fast-WAM / Lingbot-VA / Cosmos Policy** | RoboCasa365 Target 100% 直接对照；ABot-M0.5 **54.2%** vs Lingbot-VA **45.1%** |
| **[Qwen-RobotManip](./qwen-robot-manip.md)** | 同 **阿里巴巴** 生态；Qwen 走 **大 VLA + flow DiT**，ABot-M0.5 走 **显式 WAM 级联与 rollout 对齐** |
| **ABot-M0** | 同团队 VLA 前序；M0.5 升格为 **统一移动–操作 WAM** |

## 关联页面

- [World Action Models（WAM）](../concepts/world-action-models.md) — Joint 族与移动操作实例坐标
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 移动操作任务与技术路线总览
- [Manipulation](../tasks/manipulation.md) — RoboTwin / LIBERO 精细操纵语境
- [VLA](../methods/vla.md) — π₀.₅ / GR00T 等反应式对照基线
- [MotionWAM](./paper-motionwam-humanoid-loco-manipulation-wam.md) — 人形实时 WAM 对照
- [Loco-Manip 接触 · 05 VLA/WM](../overview/loco-manip-contact-category-05-vla-world-models.md) — 上层模型与接触结构接口

## 参考来源

- [ABot-M0.5 论文摘录（arXiv:2607.00678）](../../sources/papers/abot_m05_arxiv_2607_00678.md)

## 推荐继续阅读

- [ABot-M0.5 论文（arXiv:2607.00678）](https://arxiv.org/abs/2607.00678)
- [ABot-Manipulation 代码仓库](https://github.com/amap-cvlab/ABot-Manipulation)
- [ABot-M0 技术报告（arXiv:2602.11236）](https://arxiv.org/abs/2602.11236) — 同团队前序 VLA
- [MotionWAM 论文实体](./paper-motionwam-humanoid-loco-manipulation-wam.md) — 人形 loco-manip WAM 对照
