# Muninn: Your Trajectory Diffusion Model But Faster

> 来源归档

- **标题：** Muninn: Your Trajectory Diffusion Model But Faster
- **类型：** paper
- **出处：** 2026 · RSS 2026 · arXiv preprint
- **论文链接：** <https://arxiv.org/abs/2605.09999>
- **HTML：** <https://arxiv.org/html/2605.09999v1>
- **代码/项目：** <https://github.com/gokulp01/Muninn>
- **入库日期：** 2026-07-16
- **一句话说明：** UIUC 等提出的 **免训练** 轨迹扩散 **缓存包装器**：用廉价 **probe 特征** + **采样器解析灵敏度系数** 做 split-conformal 标定，在可证 **轨迹偏差预算** 下选择性复用 denoiser 输出，D4RL / 构型空间规划 / visuomotor 扩散策略上最高约 **4.6×** 墙钟加速且保持任务指标，真机导航与操作闭环验证。

---

## 核心摘录（策展，非全文）

### 问题与动机

- **轨迹扩散规划器 / 扩散策略** 能合成丰富多模态运动，但 **逐步去噪** 使在线规划与控制 **过慢**。
- 现有加速路线：**改采样器**、**压缩网络**、**蒸馏 / 早退** — 往往牺牲轨迹质量、需重训，且 **缺乏与下游控制风险的显式联系**。
- 本文问题：如何 **在不重训、不改骨干** 的前提下，通过 **复用内部计算** 加速，同时 **行为接近全量计算教师**？

### 关键洞察

1. **廉价 probe**：denoiser 的 **stem / 前几层** 对 $(\tau_t, t, c)$ 的中间表示 $F_t$ 变化缓慢时，完整 denoiser 输出也趋于稳定 → 可安全复用缓存。
2. **解析灵敏度**：DDPM/DDIM 等采样器对噪声预测误差有 **已知仿射更新**；可写出 $\|e_t\|$ 如何经 $L_t$ 放大到最终轨迹偏差 $\| \Delta_0 \|$。
3. **预算式复用**：离线 **ghost reuse** 标定 $(s_t, \epsilon_t)$ 对，split-conformal 得 $U_t(s_t)$；在线按 **剩余预算** $B_{\mathrm{rem}}$ 决定 reuse vs recompute。

### 方法要点

| 维度 | Muninn |
|------|--------|
| **定位** | 模型无关的 **training-free wrapper**，不改 $\varepsilon_\theta$ 权重 |
| **Probe** | $\Psi(\tilde{\tau}_t,t,c)$：stem / 注意力前缀 + mean-pool；目标成本 ≪ 完整 forward |
| **Score** | $s_t = \|F_t - F_{t+1}\|_1 / (\|F_{t+1}\|_1 + \omega)$ |
| **偏差界** | $d(\tau_0^{\mathrm{full}}, \tilde{\tau}_0) \leq \sum_t \Gamma L_t \|e_t\|$ |
| **标定** | Split-conformal regression：$U_t(s)$ 高概率上界 $\|e_t\|$ |
| **策略** | Algorithm 1：若 $\hat{c}_t(s_t) \leq B_{\mathrm{rem}}$ 则 reuse，否则 recompute；$B_{\mathrm{rem}} \leftarrow \eta_{\mathrm{traj}}$ |
| **证书** | 用户给定 $\eta_{\mathrm{traj}}$（偏差容忍）与 $\alpha$（风险）；$\mathbb{P}(d > \eta_{\mathrm{traj}}) \leq \alpha$ |
| **工程** | 禁止 reuse 的前缀/后缀步；批采样时每轨迹独立预算 |

### 实验摘要

- **离线 RL / 规划（D4RL）**：Diffuser、Dec. Diff.、Diff-QL、AdaptDiff、CompDiff 等；任务分 **基本持平**，延迟与 **#evals/step** 大幅下降（例：Diffuser HalfCheetah 580→145 ms，100→17 evals）。
- **构型空间规划**：MPD、EDMP；成功率与碰撞率接近 Full，延迟约 **−30%～−40%**。
- **Visuomotor**：Diffusion Policy（RLBench / Meta-World）、DP3 Pour；成功率略降 <1%，延迟约 **−40%**。
- **真机**：SeaRobotics ASV 2D 导航、Crazyflie 3D 航点、SO-ARM100 操作；Muninn 在相近成功率下 **2–3×** 更低延迟。
- **对照**：优于 FewSteps / FixedSkip / ProbeThresh 等 **无风险建模** 的 inference-time 基线；可与蒸馏模型 **叠加**。

### 局限（论文自述）

- 证书相对 **选定度量 $d$** 与 **标定分布**；分布漂移需重标定。
- 保证 **接近教师轨迹**，不直接保证碰撞/约束/闭环稳定。
- Union bound 在极小 $\alpha$ 或长 horizon 时可能保守。

### 对 wiki 的映射

- [paper-muninn-trajectory-diffusion-acceleration](../../wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md)
- [diffusion-policy](../../wiki/methods/diffusion-policy.md)
- [diffusion-motion-generation](../../wiki/methods/diffusion-motion-generation.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2605.09999>
- 代码：<https://github.com/gokulp01/Muninn>
