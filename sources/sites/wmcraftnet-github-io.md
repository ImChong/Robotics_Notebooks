# wmcraftnet.github.io（WM-Craftnet 项目页）

- **标题：** WM-Craftnet — World Synesthesia Model for Generalizable and Robust Dexterous In-Hand Manipulation
- **类型：** site / project-page
- **URL：** <https://wmcraftnet.github.io/>
- **arXiv：** <https://arxiv.org/abs/2609.07002>
- **会议：** CoRL 2026
- **入库日期：** 2026-09-11
- **复核日期：** 2026-09-11
- **配套论文：** [WM-Craftnet（arXiv:2609.07002）](../papers/wm_craftnet_arxiv_2609_07002.md)

## 一句话摘要

Sharpa Robotics 提出的 **WM-Craftnet** 官方站点：Dreamer 式 **World Synesthesia Model（WSM）** 从本体、腕部 depth、触觉、动作学习 action-conditioned 循环表征，以 **clean depth 重建** 为核心监督；部署时把 WSM 确定性特征作为 PPO 策略的 **可复用任务上下文**（非想象 rollout），在 Sharpa 真机上实现多物体、多旋转轴、扰动恢复与 sim-to-real 手内旋转。

## 公开信息要点（截至 2026-09-11 复核）

- **机构：** Sharpa Robotics（Jie Yin、Zeyuan Zhao、Xiaojing Tan、Yang Liu、Chiyu Wang、Xinyang Gu）。
- **TL;DR：** WSM 把 depth、touch、proprioception 与控制历史压成 **action-conditioned recurrent state**，供单策略在 pose shift、扰动、未见物体、不同旋转轴与工具式平移下闭环调整。
- **管线：** RSSM 重建 depth + 低维 proprio-tactile + reward 预测 → 确定性循环特征 → asymmetric actor–critic（PPO）。
- **代码 / 数据（步骤 2.5）：** 页头 **We will release the codes soon!**；**无** 可点击 GitHub 链。用户提供的 `https://github.com/sharpa-robotics/WM-Craftnet` 经 API 核查为 **404**。按 **宣称将开源 / 待发布** 处理。

### 仿真消融（z 轴，项目页表）

| Method | Return ↑ | EpLen ↑ | RotR ↑ | Fall ↓ |
|--------|----------|---------|--------|--------|
| Best raw-sensor baseline | 386.9 | 362.5 | 1.018 | 0.047 |
| WM-Craftnet from scratch | 414.3 | 264.2 | 0.742 | 0.002 |
| WSM pretrained, prop-only | 684.5 | 424.0 | 1.191 | 0.009 |
| WSM pretrained, prop+tac | 698.9 | 435.4 | 1.192 | 0.004 |
| WSM pretrained, prop+tac+depth | **753.3** | 435.2 | **1.293** | 0.005 |

- Clean-depth WSM 优于 noisy-depth 监督（**+45.3** Return）。
- t-SNE：\(h_t\) 按物体几何/接触工况聚类，**无 object ID**。

### 真机 duck z 轴（RR / SR）

| Method | Duck RR/SR |
|--------|------------|
| In-Hand Rotation (IHR) | 1.83 / 5/10 |
| IHR + WSM-denoised depth | 2.76 / 8/10 |
| **WM-Craftnet** | **16.18 / 10/10** |

跨物体（cross / corner / unseen）项目页亦有完整表；WM-Craftnet 均显著高于 IHR 系基线。

### 四十九物体下游（WSM 先验）

- 九物体 z 轴预训练 WSM → 四十九物体下游：**9.37±0.13 rad/ep** vs 无先验 **3.28**；fall **0.3%** vs **6%**（3000 epoch）。

### 评测范围（项目页自述）

- **z 轴：** 仿真九物体（corner block、duck、apple 等）；真机多物体连续 rollout、扰动恢复、零样本三件未见物。
- **y / x 轴：** 不同物体集与接触模式（elongated、rolling、侧向力矩）。
- **重力不变：** 紧凑抓型（HORA-inspired）下掌姿变化仍旋转。
- **工具：** 螺丝刀平移与快转（定性）。
- **局限：** 定量聚焦短视界旋转；平移/更广 sensing 变化多为 future work。

## 对 wiki 的映射

- [paper-wm-craftnet.md](../../wiki/entities/paper-wm-craftnet.md)
- [wm_craftnet_arxiv_2609_07002.md](../papers/wm_craftnet_arxiv_2609_07002.md)
- [wm-craftnet.md](../repos/wm-craftnet.md)

## 参考来源

- 项目页：<https://wmcraftnet.github.io/>
- arXiv：<https://arxiv.org/abs/2609.07002>
