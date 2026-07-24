---
type: entity
tags:
  - paper
  - navigation
  - visual-navigation
  - image-goal
  - world-action-models
  - diffusion-transformer
  - egocentric-vision
  - mobile-robot
  - goal-conditioned
status: complete
updated: 2026-07-24
arxiv: "2606.13494"
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../methods/model-based-rl.md
  - ../tasks/vision-language-navigation.md
  - ../overview/world-models-route-02-joint.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ./paper-shenlan-wm-11-cosmos-policy.md
  - ./paper-motionwam-humanoid-loco-manipulation-wam.md
  - ./paper-worldvln-aerial-vln-wam.md
sources:
  - ../../sources/papers/navwam_arxiv_2606_13494.md
summary: "NavWAM（arXiv:2606.13494）：目标条件视觉导航的 Navigation World Action Model——在 Cosmos Predict 2（2B）九帧共享 latent 序列上联合去噪未来 egocentric 观测、goal-progress value 与 action chunk；policy 模式单次扩散前向直接闭环控制，无需 CEM；go stanford 与 Diablo 真机 24 episode 上优于 NWM+CEM 与 OmniVLA。"
---

# NavWAM（目标条件视觉导航 · Navigation World Action Model）

**NavWAM**（*A Navigation World Action Model for Goal-Conditioned Visual Navigation*，arXiv:2606.13494，[项目页](https://dachii-azm.github.io/navwam/)）提出：把 **导航世界模型的视觉前瞻** 从「预测模块 + 外部 planner」改成 **可直接执行的 diffusion-transformer 策略**——在 **共享 latent 序列** 里同时表示 **未来 egocentric 观测、目标进度 value 与 action chunk**，一次 **policy 模式** 扩散前向即可闭环导航，**无需 CEM 式测试时轨迹优化**。

## 一句话定义

**用九帧 latent canvas 把「会看到什么、离目标多近、下一步怎么走」绑进同一次扩散去噪**——让导航世界模型的视觉前瞻 **直接变成可部署控制**，而不是事后交给 CEM 规划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合世界预测与动作生成的具身策略 |
| NWM | Navigation World Model | 侧重导航场景的未来观测预测模块 |
| CEM | Cross-Entropy Method | 测试时常用于世界模型上的采样轨迹优化 |
| VLA | Vision-Language-Action | 直接观测→动作策略（OmniVLA 等对照基线） |
| ATE | Absolute Trajectory Error | 绝对轨迹误差（go stanford 等指标） |
| RPE | Relative Pose Error | 相对位姿误差 |

## 为什么重要

- **填补「导航 WM → 闭环控制」接口：** 既有 **navigation world models** 擅长 **egocentric 未来预测**，但闭环部署往往还要 **CEM / MPC** 在预测未来上搜动作；NavWAM 把 **未来图像、value、action** 放进 **同一扩散骨干**，推理时 **policy 模式单次查询** 即得 **可执行 action chunk**。
- **与 Cosmos 系 WAM 谱系衔接：** 架构继承 **Cosmos Predict 2（2B）** 与 [Cosmos Policy](./paper-shenlan-wm-11-cosmos-policy.md) 的 **latent-frame** 原则（state / action / value 编码为 **latent frame** 而非独立头），与 [MotionWAM](./paper-motionwam-humanoid-loco-manipulation-wam.md) 等同属 **Cosmos 视频骨干 + Joint WAM** 家族，但任务聚焦 **地面 image-goal 导航** 而非人形 loco-manip。
- **联合监督的实证价值：** 消融显示 **仅未来图像 + CEM** 不足；**action + state + value + 未来图像** 的 **联合 formulation** 才使前瞻 **对控制有用**——与 [World Action Models](../concepts/world-action-models.md) 中「未来预测须参与动作条件化」的讨论一致。
- **真机闭环证据：** **Diablo 移动机器人** 24 episode **image-goal** 任务达 **79.2%** 成功率，高于 **OmniVLA（58.3%）** 与 **NWM（16.7%）**，且 **2B 视频骨干** 相对 **7B VLA** 更轻量。

## 核心结构

| 模块 | 作用 |
|------|------|
| **Cosmos Predict 2（2B）骨干** | 预训练视频世界模型 + causal VAE；提供九帧 **latent canvas** 去噪引擎 |
| **底层四帧（条件）** | blank pad、当前 **state**、**goal frame**（目标图像）、当前 **egocentric 观测** |
| **顶层五帧（预测）** | **action chunk**、**future state**、**两帧未来 egocentric 图像**、**goal-progress value** |
| **三模式训练** | policy **50%** / world-model **25%**（额外条件 action）/ value **25%** |
| **policy 模式推理** | 直接输出 action chunk → **receding-horizon** 执行 → 重查询；附带可解释 **未来视角与 value** |

### 流程总览

```mermaid
flowchart TB
  subgraph cond [底层条件帧 latent]
    PAD[blank pad]
    ST[state_t]
    G[goal image]
    O[egocentric obs_t]
  end
  subgraph pred [顶层预测帧 latent]
    A[action chunk]
    ST2[state_{t+1}]
    O1[future obs 1]
    O2[future obs 2]
    V[goal-progress value]
  end
  cond --> DIT[Cosmos Predict 2\nDiffusion-Transformer\n单次 policy 去噪]
  DIT --> pred
  A --> EXEC[机器人执行\nreceding-horizon]
  EXEC --> O
```

### 与相邻范式的边界

| 范式 | 与 NavWAM 的分界 |
|------|------------------|
| **NWM + CEM** | 先滚未来再在动作空间 **采样优化**；NavWAM **默认 policy 模式无 CEM** |
| **Cosmos Predict 2 + CEM** | 同类视频 WM + 外部 planner；NavWAM **端到端联合 latent** |
| **OmniVLA 等导航 VLA** | **观测→动作** 直接策略；NavWAM 显式联合 **未来观测与 value** |
| **语言条件 VLN** | 见 [视觉–语言导航](../tasks/vision-language-navigation.md)；本文为 **image-goal**（目标图像而非自然语言） |
| **[WorldVLN](./paper-worldvln-aerial-vln-wam.md)** | 同为 WAM 导航实例，但面向 **UAV VLN** 与 **自回归潜转移** |

## 实验要点（索引级）

| 轴 | 报告口径（以论文 / 项目页为准） |
|----|--------------------------------|
| **离线基准** | **go stanford image-goal navigation**；指标 **ATE / RPE**（越低越好） |
| **相对 WM+CEM** | NavWAM **0.324 / 0.099** vs NWM **0.453 / 0.107**（无 FT）；**w/ FT 0.192 / 0.070** |
| **相对 VLA** | held-out **sit** benchmark：相对 **OmniVLA** **更低 ATE、略高成功率**（h=4 / h=8） |
| **真机平台** | **Diablo** 底盤；RealSense D455；Livox Mid-360；Jetson AGX Orin |
| **真机闭环** | 四室内环境 **24** episodes：**NavWAM 19/24（79.2%）** vs OmniVLA **14/24** vs NWM **4/24** |
| **训练路径** | **仿真预训练 + 真机适配** |
| **前瞻保真** | 相对 NWM **更高主体一致性**（预测与未来真值 egocentric 视觉特征相似度） |

### 监督消融要点（go stanford）

| 监督组合 | 推理 | 结论方向 |
|----------|------|----------|
| 仅未来图像 | CEM 规划 | 弱，即便 N=120 候选动作 |
| 图像 + 动作 + 状态 | policy | 大幅提升 |
| 图像 + 动作 + 状态 + value | policy | **最佳** |
| 无未来图像（动作+状态+value） | policy | 仍 **降精度** → 需 **联合** 而非单头 |

## 结论

**导航世界模型要直接变成可部署控制，就得把未来观测、goal-progress value 与 action chunk 绑进同一次扩散去噪，而不是预测后再挂 CEM。**

1. **默认走 policy 模式、丢掉 CEM** — 九帧 latent canvas 上单次扩散前向得 action chunk，receding-horizon 闭环；相对 NWM+CEM 离线 ATE/RPE 更低（0.324/0.099 vs 0.453/0.107，无 FT）。
2. **联合监督是关键，不是「多一个头」** — 仅未来图像 + CEM 弱；图像+动作+状态大幅提升；再加 value 最佳；去掉未来图像仍降精度。
3. **真机轻量优势可读** — Diablo 24 episode image-goal：79.2%（19/24）优于 OmniVLA 58.3% 与 NWM 16.7%；2B 视频骨干相对 7B VLA 更轻。
4. **任务边界要守** — 本文是地面 image-goal，不是语言 VLN、也不是人形全身；代码尚未开源，真机规模仅四环境 24 episode。
5. **与 Cosmos Policy / MotionWAM 同族不同任务** — 复用 latent-frame 原则，选型时按「导航闭环」而非「任意视频 WM + 导航头」来接。

## 常见误区或局限

- **误区：** 把 NavWAM 等同于「任意视频 WM + 导航头」；关键是 **九帧共享 latent 序列** 与 **Cosmos Policy 式非视觉 latent frame**，以及 **policy / WM / value 三模式联合训练**。
- **误区：** 认为去掉 CEM 必然牺牲前瞻质量；论文强调 **动作策略化后仍保留可解释未来视角预测**。
- **局限：** 任务为 **image-goal** 地面导航，与 **语言指令 VLN**、**人形全身** 栈不同；代码截至 ingest **尚未开源**；真机评测规模为 **24 episode / 四环境**（以原文为准）。

## 方法栈

见上文 **核心结构**、**流程总览** 与 **三模式训练**；骨干为 **Cosmos Predict 2（2B）** diffusion-transformer，非视觉量按 **Cosmos Policy latent-frame** 编码；部署为 **policy 模式 receding-horizon** 闭环，无 CEM。

## 与其他页面的关系

- [World Action Models（WAM）](../concepts/world-action-models.md) — Joint 族 **导航** 实例坐标
- [Cosmos Policy](./paper-shenlan-wm-11-cosmos-policy.md) — latent-frame 联合建模先例
- [Generative World Models](../methods/generative-world-models.md) — Cosmos Predict 2 视频动力学工具箱
- [VLA](../methods/vla.md) — OmniVLA 等直接策略对照
- [Model-Based RL](../methods/model-based-rl.md) — CEM 规划语境
- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md) — 更广导航任务族；本文为 **视觉目标图像** 子问题
- [MotionWAM](./paper-motionwam-humanoid-loco-manipulation-wam.md) — 同 Cosmos 系、不同形态与任务的 WAM 对照
- [WorldVLN](./paper-worldvln-aerial-vln-wam.md) — 另一 WAM 导航闭环部署实例

## 参考来源

- [NavWAM 论文摘录（arXiv:2606.13494）](../../sources/papers/navwam_arxiv_2606_13494.md)

## 推荐继续阅读

- [NavWAM 论文（arXiv:2606.13494）](https://arxiv.org/abs/2606.13494)
- [NavWAM 项目页](https://dachii-azm.github.io/navwam/)
- [Cosmos Policy 论文实体](./paper-shenlan-wm-11-cosmos-policy.md) — latent-frame WAM 先例（arXiv:2601.16163）
- [World Action Models 概念页](../concepts/world-action-models.md)
