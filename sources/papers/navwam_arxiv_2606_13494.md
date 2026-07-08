# NavWAM（arXiv:2606.13494）

> 来源归档（ingest）

- **标题：** NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation
- **缩写：** **NavWAM**
- **类型：** paper / world-action-model / goal-conditioned visual navigation
- **arXiv：** <https://arxiv.org/abs/2606.13494>
- **项目页：** <https://dachii-azm.github.io/navwam/>
- **代码：** 项目页标注 **Coming soon**（截至 ingest）
- **作者：** Daichi Azuma, Taiki Miyanishi, Koya Sakamoto, Shuhei Kurita, Yaonan Zhu, Petr Khrapchenkov, Motoaki Kawanabe, Yusuke Iwasawa, Yutaka Matsuo
- **机构：** 东京大学（The University of Tokyo）；国立情报学研究所（NII）；AIRoA；ATR
- **入库日期：** 2026-07-08
- **一句话说明：** 面向 **目标条件视觉导航（image-goal）** 的 **Navigation World Action Model**：在 **Cosmos Predict 2（2B）** 视频世界模型上，用 **九帧共享 latent 序列** 联合去噪 **未来 egocentric 观测、目标进度 value 与可执行 action chunk**；推理时 **policy 模式单次扩散前向** 直接输出动作，**无需 CEM 式测试时轨迹优化**；仿真预训练 + 真机适配，在 **go stanford image-goal** 与 **Diablo 移动机器人闭环** 上优于 **NWM + CEM** 与 **OmniVLA** 等基线。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / 项目页 Overview）

- **链接：** <https://arxiv.org/abs/2606.13494>；<https://dachii-azm.github.io/navwam/>
- **核心贡献：** **目标条件视觉导航** 需在 **部分可观测** 下预判运动如何改变未来 egocentric 视角、以及该变化是否 **更接近目标**。**Navigation world models** 提供此类视觉前瞻，但传统上仍是 **预测模块**，需 **外部 planner**（如 **CEM**）把预测未来转成闭环控制。NavWAM 提出 **diffusion-transformer 策略**，把导航世界模型预测 **直接变成可执行动作**：在 **共享 latent 序列** 中表示 **未来观测、目标进度 value、action chunk**，并 **联合学习** 未来预测与决定闭环行为的 **动作 / value 目标**，使视觉前瞻 **可直接用于机器人控制**。
- **对 wiki 的映射：**
  - [NavWAM 论文实体](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [Cosmos Policy](../../wiki/entities/paper-shenlan-wm-11-cosmos-policy.md)（latent-frame 联合建模先例）

### 2) 九帧 latent canvas 与三模式训练（Method）

- **链接：** <https://dachii-azm.github.io/navwam/> § Method
- **核心贡献：** 基于预训练 **Cosmos Predict 2（2B）** 视频世界模型，在 **Cosmos Policy** 的 **latent-frame** 原则上构建 **固定九帧 latent canvas**：
  - **底层四帧（观测条件）：** causal VAE 要求的 **blank pad**、**当前机器人 state**、**goal frame**（image-goal 设定下的目标图像）、**当前 egocentric 观测**。
  - **顶层五帧（预测目标）：** **可执行 action chunk**、**future state**、**两帧未来 egocentric 图像**、**goal-progress value**。
  - **非视觉量**（state / action / value）编码为 **latent frame**，而非独立输出头，使 **单模型** 联合预测视觉与非视觉导航变量。
- **三条件模式（每样本随机采样）：** **policy 50%**（预测 action + future state + future images + value）；**world-model 25%**（额外条件于 action）；**value 25%**（估计 goal progress）。
- **推理：** **policy 模式**——直接输出 action chunk，**receding-horizon** 执行后重查询；同时产出 **未来视角与 value** 作为可解释前瞻，**无外部 planner**。
- **对 wiki 的映射：**
  - [NavWAM 论文实体](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md) — 流程总览与 Mermaid
  - [Generative World Models](../../wiki/methods/generative-world-models.md)

### 3) 相对 planning-based world models：无需 CEM（World Models as Policies）

- **链接：** 项目页 § World Models as Policies
- **核心贡献：** 在 **go stanford image-goal navigation** 上，NavWAM 用 **单次 policy 查询** 替代 **NWM 式 CEM 规划**。**无域内微调** 已优于 **Cosmos Predict 2 + CEM** 与 **NWM（CEM）**；短 **in-domain fine-tune（w/ FT）** 进一步领先——且全程 **无 CEM 式 action search**。

| Method | ATE ↓ | RPE ↓ |
|--------|-------|-------|
| Cosmos Predict 2 + CEM | 0.455 | 0.109 |
| NWM (CEM) | 0.453 | 0.107 |
| NavWAM | 0.324 | 0.099 |
| NavWAM w/ FT | 0.192 | 0.070 |

- **对 wiki 的映射：**
  - [NavWAM 论文实体](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md) — 实验要点
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)（CEM 规划对照语境）

### 4) 联合监督消融：未来图像 + 动作 + value 缺一不可（Learning Useful Futures for Control）

- **链接：** 项目页 § Learning Useful Futures for Control
- **核心贡献：**
  - **仅未来图像 + CEM 规划** 不足以导航（N=120 候选动作仍弱）。
  - 加入 **action + state** 监督后轨迹精度 **大幅提升**；再加入 **goal-progress value** 得最佳结果。
  - **去掉未来图像监督**（仅 action + state + value）同样 **降精度**——关键是 **联合 formulation**，而非单一预测头。
  - 在 **held-out sit benchmark** 上相对直接 **VLA 策略 OmniVLA** 仍具竞争力：**更低 ATE、略高成功率**（h=4 / h=8），且 **2B 视频骨干** vs OmniVLA **7B VLA 骨干**，并额外产出 **未来视角与 value 预测**。
- **对 wiki 的映射：**
  - [NavWAM 论文实体](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)
  - [VLA](../../wiki/methods/vla.md)（OmniVLA 对照）

### 5) 真机闭环部署（Closed-loop Real-Robot Deployment）

- **链接：** 项目页 § Closed-loop Real-Robot Deployment
- **核心贡献：** **Diablo 移动机器人**（RealSense D455、Livox Mid-360、NVIDIA Jetson AGX Orin）在 **四个室内环境** 共 **24** 个闭环 **image-goal** episode：
  - **NavWAM：19/24（79.2%）**
  - **OmniVLA：14/24（58.3%）**
  - **NWM：4/24（16.7%）**
- 定性：NavWAM **更稳定到达目标区域**；NWM 易 **漂移**；OmniVLA 有时 **提前停止或路径不够直接**。
- **训练路径：** **仿真预训练 + 真机适配**（sim pretraining + real-robot adaptation）。
- **对 wiki 的映射：**
  - [NavWAM 论文实体](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)
  - [视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md)（导航闭环对照；本文为 **image-goal** 而非语言指令）

### 6) 保留视觉前瞻能力（Preserving Visual Foresight）

- **摘录要点：** 把导航世界模型变成 **动作策略** 并不牺牲 **未来预测**；NavWAM 在 **主体一致性**（预测与未来真值观测的视觉特征相似度）上优于 NWM，且每步可展示 **预测未来视角** 与后续真实观测的对照。
- **对 wiki 的映射：**
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint 族「预测 + 控制不互斥」实例

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md`](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)
- 互链参考：[World Action Models](../../wiki/concepts/world-action-models.md)、[Cosmos Policy](../../wiki/entities/paper-shenlan-wm-11-cosmos-policy.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[VLA](../../wiki/methods/vla.md)、[视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)

## BibTeX（项目页提供）

```bibtex
@misc{azuma2026navwam,
  title         = {NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation},
  author        = {Daichi Azuma and Taiki Miyanishi and Koya Sakamoto and Shuhei Kurita and Yaonan Zhu and Petr Khrapchenkov and Motoaki Kawanabe and Yusuke Iwasawa and Yutaka Matsuo},
  year          = {2026},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.13494},
  note          = {Project page: https://dachii-azm.github.io/navwam/}
}
```
