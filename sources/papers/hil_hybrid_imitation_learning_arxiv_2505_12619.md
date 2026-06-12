# HIL: Hybrid Imitation Learning of Diverse Parkour Skills from Videos

> 来源归档（ingest）

- **标题：** HIL: Hybrid Imitation Learning of Diverse Parkour Skills from Videos
- **缩写：** **HIL**（Hybrid Imitation Learning）
- **类型：** paper / physics-based character animation / human-scene interaction
- **arXiv：** <https://arxiv.org/abs/2505.12619v1>（PDF：<https://arxiv.org/pdf/2505.12619v1>）
- **演示视频：** <https://youtu.be/le4248gIMME>
- **作者：** Jiashun Wang, Yifeng Jiang, Haotian Zhang, Chen Tessler, Davis Rempe, Jessica Hodgins, Xue Bin Peng
- **机构：** Carnegie Mellon University, NVIDIA
- **入库日期：** 2026-06-12
- **一句话说明：** 在**并行多任务**环境中联合 **motion tracking** 与 **AMP 式对抗模仿**，配合**统一观测空间**（角色状态 + 场景点云）与 **PSI 扰动初始化**，从互联网跑酷视频重建参考动作，训练能在**新障碍布局**中组合多种跑酷技能的统一物理角色控制器。

## 核心摘录

### 1) 问题与动机

- **Motion tracking** 能精确复现单技能，但难以适应新障碍、编排技能序列。
- **纯 AIL/AMP** 更灵活，但易 **mode collapse**（反复用同一动作、与场景脱节）。
- 跑酷需要：多动态特技串联 + 按场景调整行为；参考数据稀缺且缺对齐场景几何。

### 2) 混合模仿框架

并行两类任务、等概率采样：

| 模式 | 任务 | 奖励 |
|------|------|------|
| **Motion tracking** | 逐帧跟踪参考跑酷片段 | 位姿/速度/根高跟踪 \(r^{track}\) + 共享 **style** \(r^{style}\) |
| **Target following（AMP）** | 沿障碍序列跟随随机目标点 \(g_t\) | 目标进度 \(r^{target}\) + style \(r^{style}\) |

- **Style reward**：场景条件判别器 \(D(s_{t-n:t}, c_{t-n:t})\)，类似 AMP + 梯度惩罚；判别器输入含 **10 步状态转移** 与 **场景点云** \(c\)，判断动作是否既自然又**贴合当前障碍**。
- **统一观测**：策略输入 \((s_t, o_t, g_t)\)——**不用**相位变量或未来参考姿态；场景观测在 tracking 中充当**空间相位**，在 target 任务中提供障碍感知。

### 3) 实现要点

- **角色**：SMPL 物理角色；PD 控制；4096 并行环境 × 4×V100。
- **数据**：互联网跑酷视频 → 现成姿态重建 + **交互式场景标注工具** 获得对齐场景几何的参考片段。
- **训练障碍**：每课 5 个障碍、间距 2–3 m；评估时对位置/朝向/尺度加高斯噪声，可泛化到 20 障碍序列。
- **PSI（Perturbed State Initialization）**：从参考采样初始状态时加扰动，促进技能间过渡、减轻 mode collapse。
- **Critic**：含 **task indicator** \(k_t\) 区分两任务奖励结构。

### 4) 结果与局限

- 相对纯 tracking / 纯 AMP / 无场景判别器基线：**动作质量更高、技能覆盖更广、任务完成率更好**。
- 可与 SAMP 坐姿等日常交互**同一策略**混训（跑酷 + 坐椅子）。
- **局限**：偶发不自然恢复；训练假设**顺序排列**障碍课；对数据外布局/障碍类型适应有限。

## 对 wiki 的映射

- 新建方法页：[`wiki/methods/hil-hybrid-imitation-learning.md`](../../wiki/methods/hil-hybrid-imitation-learning.md)
- 交叉更新：
  - [`wiki/methods/amp-reward.md`](../../wiki/methods/amp-reward.md) — 场景条件判别器与 HIL 联合训练
  - [`wiki/methods/deepmimic.md`](../../wiki/methods/deepmimic.md) — tracking 分支对照
  - [`wiki/methods/mtrg-reference-goal-driven-rl.md`](../../wiki/methods/mtrg-reference-goal-driven-rl.md) — 同作者团队后人形「参考塑形 + 目标泛化」路线（无对抗）
  - [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) — 跑酷 / 障碍穿越技能组合

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2505.12619v1>
- 视频：<https://youtu.be/le4248gIMME>
