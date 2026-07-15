# SRU — Spatially-Enhanced Recurrent Memory（IJRR 2025）

> 来源归档（ingest）

- **标题：** Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning
- **缩写：** **SRU**
- **类型：** paper / mapless-navigation / recurrent-policy / sim2real
- **期刊：** *The International Journal of Robotics Research*（IJRR），2025
- **arXiv：** <https://arxiv.org/abs/2506.05997>
- **DOI：** <https://doi.org/10.1177/02783649251401926>
- **项目页：** <https://michaelfyang.github.io/sru-project-website/>
- **作者：** Fan Yang*, Per Frivik, David Hoeller, Chen Wang, Cesar Cadena, Marco Hutter
- **机构：** 苏黎世联邦理工机器人系统实验室（ETH Zurich RSL）；布法罗大学空间 AI 与机器人实验室（University at Buffalo）
- **入库日期：** 2026-07-15
- **一句话说明：** 揭示标准 RNN（LSTM/GRU 等）擅长时序记忆却难以做 **跨视角空间配准**；提出 **Spatially-Enhanced Recurrent Units（SRU）**——在 LSTM/GRU 上增加可学习的 **逐元素空间变换（star operation）**，配合 **双阶段空间注意力 + 预训练深度编码器 + 稀疏奖励 RL**，实现 **单目前向深度** 的 **无地图长程导航**，仿真相对 LSTM/GRU **+23.5%** 成功率，真机 **Unitree B2W + ZedX** **零样本** 部署办公室/露台/森林等场景。

## 核心论文摘录（MVP）

### 1) 问题：RNN 能做时间记忆，难做空间记忆（Abstract / 项目页）

- **链接：** <https://michaelfyang.github.io/sru-project-website/>；<https://arxiv.org/abs/2506.05997>
- **核心贡献：** 端到端 RL 导航常借 RNN 把历史观测压进隐状态做 **隐式建图**；但实验显示 **LSTM、GRU、S4、Mamba-SSM** 等虽能拟合 **时序依赖**，却 **难以把不同视角下的地标位置对齐到一致空间表示**——而这正是无地图长程导航所必需的「在哪里见过障碍/通道」能力。
- **对 wiki 的映射：**
  - [SRU 论文实体](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)（大规模合成深度预训练 + 零样本真机）

### 2) SRU 模块：star operation 增强空间变换（Method）

- **链接：** 项目页 § Our Solution: Spatially-Enhanced Recurrent Units
- **核心贡献：** 在标准 **LSTM / GRU** 递推中引入额外 **可学习空间变换项**，用 **逐元素乘法（star operation）** 让网络从 **egocentric 观测序列** 中 **隐式学习齐次变换式对齐**，同时保留时序记忆能力。项目提供独立 PyTorch 模块 **[sru-pytorch-spatial-learning](https://github.com/ManifoldTechLtd/sru-pytorch-spatial-learning)**（`pip install` 可集成到自有架构）。
- **对 wiki 的映射：**
  - [SRU 论文实体](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md) — 机制与流程图

### 3) 感知–注意力–SRU 导航架构（Perception & Attention）

- **链接：** 项目页 § Two-Stage Spatial Attention；§ Network Architecture
- **核心贡献：**
  - **RegNet + FPN 深度编码器**：在 **TartanAir** 等 **10 万+ 合成环境** 上预训练，配合 **并行化深度噪声模型** 做 sim-to-real 桥接。
  - **两阶段注意力**：**self-attention** 丰富全局视觉上下文；**cross-attention**（由机器人状态与目标查询）把高维特征压成 **任务相关空间线索**。
  - **端到端管线**：深度 → 注意力 → **SRU** → MLP（含 **Temporally Consistent Dropout**）→ 速度指令。
- **对 wiki 的映射：**
  - [SRU 论文实体](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md)
  - [isaac-gym-isaac-lab](../../wiki/entities/isaac-gym-isaac-lab.md)（训练栈：IsaacLab + rsl_rl）

### 4) 训练策略：稀疏奖励 + DML + TC-Dropout（Training Strategy）

- **链接：** 项目页 § Training Strategy
- **核心贡献：**
  - **稀疏奖励**：episode 末时间奖励 + 随机早停检查，减少中间奖励对探索的干扰。
  - **Deep Mutual Learning（DML）**：两条策略并行训练，**KL 蒸馏** 抑制早期过拟合，促进鲁棒时空特征。
  - **Temporally Consistent Dropout（TC-Dropout）**：rollout 与训练阶段 **跨时间步共享 dropout mask**，稳定循环记忆学习。
- **对 wiki 的映射：**
  - [SRU 论文实体](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md)

### 5) 仿真结果：相对 RNN 与显式建图基线（Results）

- **链接：** 项目页 § Results & Performance
- **核心贡献（成功率 Overall）：**

| Method | Overall |
|--------|---------|
| LSTM | 63.5% |
| GRU | 61.0% |
| **SRU** | **78.9%** |
| GTRL（堆叠历史帧） | 38.2% |
| EMHP（显式建图 + 路径） | 60.4% |
| GTRL* + SRU 记忆 | 66.3% |
| SRU + 完整架构 | 78.3% |

- **关键发现：** 仅把 GTRL 的堆叠观测换成 **SRU 隐式记忆** 即 **+73%**；楼梯等需 **3D 空间记忆** 的场景 SRU 相对 LSTM **约 2.5×**；**>50 m** 长程仍保持 **>80%** 成功率（显式记忆约 20 m 后陡降）。
- **对 wiki 的映射：**
  - [SRU 论文实体](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md)
  - [视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md)（对照：本文为 **坐标/向量目标** 的无地图导航，非语言指令）

### 6) 真机零样本部署（Real-World Deployment）

- **链接：** 项目页 § Real-World Deployment
- **核心贡献：**
  - **平台：** **Unitree B2W** + **ZedX** 立体相机 + **NVIDIA Jetson AGX Orin**（策略 5 Hz）；**DLIO** LiDAR 定位仅用于状态估计（非建图导航）。
  - **零样本：** 仅合成深度预训练，**无真机微调**；办公室、大厅、露台、森林四类环境；单目标 **70 m+**（训练最长 30 m）、森林单次任务 **100 m+** 穿行。
  - **能力：** 死胡同 **回溯重路由**、动态环境变化适应——LSTM 策略易在死胡同 **无限循环**。
- **代码族（上游五仓）：** `sru-navigation-sim`（IsaacLab 任务）、`sru-navigation-learning`（rsl_rl 训练）、`sru-depth-pretraining`、`sru-robot-deployment`、`sru-pytorch-spatial-learning`。
- **对 wiki 的映射：**
  - [SRU 论文实体](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md)
  - [SRU-Odin 部署仓](../../wiki/entities/sru-odin.md)（第三方 Odin1 传感器 + Go2 移植）
  - [Unitree](../../wiki/entities/unitree.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md`](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md)
- 部署移植：[`wiki/entities/sru-odin.md`](../../wiki/entities/sru-odin.md)
- 互链参考：[Sim2Real](../../wiki/concepts/sim2real.md)、[isaac-gym-isaac-lab](../../wiki/entities/isaac-gym-isaac-lab.md)、[视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)、[NavWAM](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)（同为闭环导航，但走 WAM 扩散路线）

## BibTeX（项目页提供）

```bibtex
@article{yang2025sru,
  author = {Yang, Fan and Frivik, Per and Hoeller, David and Wang, Chen and Cadena, Cesar and Hutter, Marco},
  title = {Spatially-enhanced recurrent memory for long-range mapless navigation via end-to-end reinforcement learning},
  journal = {The International Journal of Robotics Research},
  year = {2025},
  doi = {10.1177/02783649251401926},
  url = {https://doi.org/10.1177/02783649251401926}
}
```
