# AssistMimic: Learning to Assist — Physics-Grounded Human-Human Control via Multi-Agent RL

> 来源归档（ingest）

- **标题：** Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning
- **类型：** paper
- **来源：** arXiv / CVPR 2026（项目页 BibTeX 标注）
- **原始链接：**
  - <https://arxiv.org/abs/2603.11346>
  - 项目页：<https://yutoshibata07.github.io/AssistMimic/>
- **机构：** Carnegie Mellon University；Keio AI Research Center；Keio University
- **入库日期：** 2026-06-03
- **一句话说明：** 将 **紧密接触、力交换** 的双人 assistive 动作模仿表述为 **MARL**，联合训练 supporter / recipient 的 physics tracking 策略；以 **PHC 单人 prior 初始化**、**动态参考重定向** 与 **接触促进奖励** 解决高接触 MARL 探索难题，在 Inter-X / HHI-Assist 上首次稳定跟踪 assistive 交互参考。

## 核心论文摘录（MVP）

### 1) 问题：单人 GMT 无法覆盖 assistive 力交换

- **链接：** <https://arxiv.org/abs/2603.11346>
- **摘录要点：** 现有 **General Motion Tracking (GMT)** 多在 **无接触社交交互** 或 **孤立单人动作** 上成功；**护理 / 扶起** 场景需要持续感知伙伴姿态与动力学、在 **力交换** 下双向协调。先前 **kinematic replay**（冻结 recipient 轨迹）在 recipient 无法独立站稳时会 **穿透 / 独立站起**，破坏物理一致性。
- **对 wiki 的映射：**
  - [AssistMimic](../../wiki/entities/paper-assistmimic.md) — 问题定义、baseline 失败模式与 WBT 扩展语境

### 2) MARL  formulation + PHC prior 初始化

- **链接：** <https://arxiv.org/abs/2603.11346> §3
- **摘录要点：**
  - 两体 MDP：supporter (S) 与 recipient (R) **独立策略** $\pi_S, \pi_R$；recipient 用 **降低 PD 增益与力矩上限** 模拟 **物理 impairments**，迫使外部支撑。
  - 策略输入 $(s_{\mathrm{prior}}, s_{\mathrm{assist}}, g)$：在 **PHC** 单人 tracking 架构上扩展 **partner-aware assistive state**（伙伴观测、接触指示、自/他力、上一步动作）。
  - **Weight init**：复制 PHC 输入权重，assistive 维 **零初始化**，保留 locomotion prior 再学交互。
- **对 wiki 的映射：**
  - [AssistMimic](../../wiki/entities/paper-assistmimic.md) — 观测、PHC prior 初始化与机制归纳

### 3) Dynamic reference retargeting + contact-promoting reward

- **链接：** <https://yutoshibata07.github.io/AssistMimic/>
- **摘录要点：**
  - **动态重定向**：两体 root 足够近时，将 supporter 手腕目标 **锚定到 recipient 当前最近关节**，保持参考中的 **相对 offset**，避免 recipient 偏离 MoCap 后支撑点失效。
  - **接触促进奖励**：近距离时 **抑制不可靠 hand tracking 惩罚**，改为 **距离 + 接触力 + 稀疏接触 bonus**；缓解 MoCap **遮挡噪声** 导致的错误 kinematic 目标。
- **对 wiki 的映射：**
  - [AssistMimic](../../wiki/entities/paper-assistmimic.md) — Mermaid 流程与奖励分解

### 4) 训练协议、数据集与主要结果

- **链接：** <https://arxiv.org/abs/2603.11346> §4
- **摘录要点：**
  - **Specialist**：按 subject 聚类，**PPO** 训练；**Generalist**：**DAgger** 蒸馏多 specialist。
  - **数据集**：Inter-X 抽取 30 条 **Help-up**；HHI-Assist 10 条 **床/椅护理** clip。
  - **Baselines**：Kinematic-Recipient、Frozen-Recipient（解耦）、Sequential Training；Phys-Reaction 类方法在 recipient 无法独立跟踪时 **不适用**。
  - **指标**：Success Rate (SR, 0.5 m 阈值)、MPJPE；AssistMimic specialist **Inter-X SR 74.9%**、**HHI-Assist 85.8%**；去掉 weight init 在 Inter-X 上 **SR→0%**。
  - **泛化**：未见 recipient 动力学（质量 ×1.2/1.5、PD×0.5、hip torque×0.5）、扩散模型生成的 **kinematic 双人交互** 亦可跟踪。
- **对 wiki 的映射：**
  - [AssistMimic](../../wiki/entities/paper-assistmimic.md) — 训练协议、数据集、MARL/DAgger 与实验结果

## 引用（项目页 BibTeX）

```bibtex
@inproceedings{shibata2026assistmimic,
    title={Learning to Assist: Physics-Grounded Human-Human Control
           via Multi-Agent Reinforcement Learning},
    author={Shibata, Yuto and Yamazaki, Kashu and Jayanti, Lalit
            and Aoki, Yoshimitsu and Isogawa, Mariko
            and Fragkiadaki, Katerina},
    booktitle={Proceedings of the IEEE/CVF Conference on
               Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}
```
