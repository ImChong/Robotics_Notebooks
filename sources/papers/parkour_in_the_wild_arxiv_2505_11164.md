# Parkour in the Wild: Learning a General and Extensible Agile Locomotion Policy Using Multi-Expert Distillation and RL Fine-tuning（arXiv:2505.11164）

> 来源归档（ingest · 一手论文）

- **标题：** Parkour in the Wild: Learning a General and Extensible Agile Locomotion Policy Using Multi-Expert Distillation and RL Fine-tuning
- **短名：** Parkour in the Wild / PITW
- **类型：** paper / quadruped / perceptive-locomotion / multi-expert-distillation / dagger / rl-finetuning
- **arXiv：** <https://arxiv.org/abs/2505.11164>
- **arXiv HTML：** <https://arxiv.org/html/2505.11164v1>
- **PDF：** <https://arxiv.org/pdf/2505.11164>
- **IJRR DOI：** <https://doi.org/10.1177/02783649261455067>
- **作者：** Nikita Rudin、Junzhe He、Joshua Aurand、Marco Hutter
- **机构：** Robotic Systems Lab, ETH Zurich；Nikita Rudin 亦隶属 NVIDIA Switzerland
- **发表：** IJRR（DOI 2026）；arXiv 2025-05
- **入库日期：** 2026-09-22
- **一句话说明：** 三阶段管线：9 个地形 **RL 专家**（高程图）→ **DAgger 式多专家蒸馏** 为 **四深度相机 + LSTM** 统策 → **RL 微调**（含搜救训练场 3D 扫描网格）；ANYmal D 真机室内外泛化。

## 开源状态（步骤 2.5）

- **核查日：** 2026-09-22；检索 arXiv、IJRR 页、ETH RSL / leggedrobotics 公开仓库与 ANYmal Research 社区入口。
- **已发布：** 论文 PDF、arXiv HTML、IJRR 正式版、演示视频（论文 Fig.1 / 补充材料）。
- **未发布：** **无** 独立 GitHub 训练/部署仓库；**无** 官方权重下载链接。
- **邻近开源（非本文实现）：** [leggedrobotics/wild_visual_navigation](https://github.com/leggedrobotics/wild_visual_navigation) 为 **Wild Visual Navigation**  traversability 系统，**不是** 本文多专家蒸馏控制器；[Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour) 为同作者系 **CoRL 2023** 四足跑酷蒸馏先例。
- **结论：** **确认未开源**（截至入库日）。wiki 实体页「源码运行时序图」写 **不适用**。

## 摘录 1：问题与三阶段贡献

足式机器人在非结构化环境中仍难 **复用** 已训技能：每遇新任务常 **从零 RL**，缺乏把多技能 **合成通用控制器** 且 **可增量扩展** 的管线。感知侧，敏捷越障时 **状态估计漂移 + 障碍出视场** 使传统高程图不足；需 **端到端深度策略 + 记忆** 推断未见结构。

**三阶段框架（Fig. 2）：**

1. **Expert skill training：** 9 种基础地形各训 **独立 RL 专家** $\pi_{\text{expert},i}$，感知为 **基座周围高程图**；在 [ANYmal Parkour（SciRob 2023）](https://doi.org/10.1126/scirobotics.adi7566) 五技能（walk / climb / climb down / jump / crouch）上增 **低墙跳、踏石、窄梁、乱石堆** 四类。
2. **Multi-expert distillation：** 并行仿真中按地形分配专家；学生 $\pi_{\text{student}}$ rollout，收集 $(o_{\text{student}}, a_{\text{expert}})$，监督最小化 $\|\pi_{\text{student}}(o_{\text{student}})-a_{\text{expert}}\|^2$（**DAgger 式在线聚合**）；学生改用 **4 路 RealSense D435i 深度** + **LSTM** 替代专家高程图。
3. **RL fine-tuning：** 蒸馏策略作 **foundation model**，在 **9 旧地形 + Parkour line + 15 个搜救设施 3D 扫描** 上 PPO 微调；支持 **重复 fine-tune** 增量加地形而不遗忘旧技能。

论文对比 **分层技能选择**（Hoeller SciRob 2023）、**VAE 技能编码** 与 **多专家蒸馏**：前两者在 **无专家的新复杂地形** 上易局部最优；纯蒸馏 **多模态平均** 导致单地形性能低于专家且泛化有限，但 **可作 foundation 再 RLFT**。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-parkour-in-the-wild.md`](../../wiki/entities/paper-parkour-in-the-wild.md)；方法总览 [`wiki/methods/multi-expert-distillation.md`](../../wiki/methods/multi-expert-distillation.md)。

## 摘录 2：蒸馏算法与网络（§2.2–2.3）

**Algorithm 1 要点：**

- 每 epoch：学生带 **高斯动作噪声** rollout → 各 env 按地形查对应专家动作 → 聚合数据集 → 监督更新学生。
- 学生需学两件事：$o_{\text{student}}\mapsto i$（隐式 **地形/技能识别**）与 $a\approx\pi_{\text{expert},i}(o_{\text{expert}})$。

**网络（Fig. 3）：** 每路深度 **CNN（3 conv + max-pool + 2 FC→64）** → 拼接本体 → **2 层 LSTM** → 再拼本体与 **位置/朝向/剩余时间指令** → **3 层 ELU MLP** 输出动作。

**RL 微调稳定三件套：** 蒸馏期 **动作噪声** 使 foundation 鲁棒；**保守 RL 超参**；**先冻 actor 预训 critic** 再联合更新。

**深度 sim2real（Fig. 4）：** 仿真 48×32 深度 → clip / 边缘 shuffle / Perlin 孔洞 / 近距 blind spot / 高斯 blur；真机同 clip、下采样与 blur。

**对 wiki 的映射：** 与 [DAgger 方法页](../../wiki/methods/dagger.md)「locomotion 变体」、[Robot Parkour Learning](../../wiki/entities/paper-robot-parkour-learning.md) 五专家蒸馏对照。

## 摘录 3：Table 4 成功率与泛化（节选）

| 地形 | 蒸馏 $\pi_D$ | RLFT $\pi_{RL}$ | 备注 |
|------|-------------|-----------------|------|
| Walk | 99.3 | **100.0** | 9 专家训练地形 |
| Climb | 98.1 | 99.5 | |
| Stepping stones | 73.0 | **98.8** | 蒸馏明显低于专家 |
| Parkour line | 5.8 | **98.5** | **仅 RLFT 阶段加入** |
| Scanned meshes (train) | 11.9 | **99.1** | 搜救 3D 扫描 |
| Scanned meshes (test) | 14.9 | **94.9** | **未见扫描** |
| Gap - climb | 10.2 | **82.0** | 未见组合地形 |
| Down - stones | 11.3 | 54.4 → **92.4**（$\pi_{RL^*}$ 再 FT） | 展示 **重复 fine-tune** 增技能 |

**真机：** ANYmal D；室内外、搜救训练场 **未见 rubble/岩石/坍塌结构**；对 **高草、光照变化、反光、泥泞、滚动石块、钢筋 foot trap** 等扰动鲁棒。

**对 wiki 的映射：** [深度感知 locomotion 路线](../../roadmap/depth-perceptive-locomotion.md) Stage 3；[Athena-WBC](../../wiki/entities/paper-athena-wbc-humanoid-longtail.md) 引用其 **蒸馏→RLFT** 配方。

## 摘录 4：与分层 ANYmal Parkour 的方法论对照（Related work 口径）

[Hoeller et al., SciRob 2023](https://doi.org/10.1126/scirobotics.adi7566) **分层 RL**：高层按感知 **离散选** 低层专家；作者在 Discussion 指出 **蒸馏版在 Barkour benchmark 上可能弱于非蒸馏**，且 **height scan 蒸馏难区分 box vs table**（crouch 技能）。本文 **PITW** 用 **四深度 + 记忆 + RLFT** 正面推进 **单策略蒸馏** 的可扩展性与 **真机 wild 部署**，并报告 **重复 fine-tune** 可 **加技能而不 catastrophic forgetting**。

**对 wiki 的映射：** [`wiki/methods/multi-expert-distillation.md`](../../wiki/methods/multi-expert-distillation.md)「与分层 MoE / 路由对照」节。

## BibTeX（arXiv）

```bibtex
@article{rudin2025parkourinthewild,
  title={Parkour in the Wild: Learning a General and Extensible Agile Locomotion Policy Using Multi-Expert Distillation and RL Fine-tuning},
  author={Rudin, Nikita and He, Junzhe and Aurand, Joshua and Hutter, Marco},
  journal={arXiv preprint arXiv:2505.11164},
  year={2025}
}
```
