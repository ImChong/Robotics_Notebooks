# Gated Memory Policy: In-Context Memorization and Adaptation

> 来源归档（论文全文通读与核对 · `arXiv:2604.18933v2`）

- **标题：** Gated Memory Policy: In-Context Memorization and Adaptation
- **类型：** paper
- **状态：** RSS 2026 · arXiv 2026（v2，2026-08-03）
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2604.18933>
  - 项目页：<https://gated-memory-policy.github.io/>
- **代码：** <https://github.com/real-stanford/gated-memory-policy>
- **权重与数据：** <https://huggingface.co/datasets/yihuai-gao/gated-memory-policy> / <https://huggingface.co/yihuai-gao/gated-memory-policy/tree/main>
- **作者：** Yihuai Gao, Jeff Jinyun Liu, Shuang Li, Shuran Song
- **机构：** Stanford University
- **入库日期：** 2026-08-18
- **更新日期：** 2026-11-19
- **核对范围：** 通读 `arXiv:2604.18933v2` 全文（§1–§9 与参考文献）、项目页、GitHub 代码仓 `real-stanford/gated-memory-policy`（`README.md`、`imitation-learning-policies/` 脚本与网络/门控源码）。
- **通读：** 已完成
- **一句话说明：** GMP（arXiv:2604.18933）面向需要跨阶段或跨试次记忆的非马尔可夫操作任务，结合「每块紧凑历史摘要 + 历史动作末步扩散加噪 + 基于无记忆/有记忆策略验证误差标定的学习型二进制门控」，在 MemMimic 相对长历史基线平均 SR +30.1%、在 MIKASA-Robo 相对 MemoryVLA +26.6%，同时在 RoboMimic 马尔可夫任务保持零退化。

## 核心论文摘录（全文核对）

### 问题与贡献

- **核心矛盾：** 盲目把所有过去帧或动作塞入 visuomotor 策略会在非关键阶段造成历史伪相关过拟合（如 RoboMimic 马尔可夫任务上长历史反而不如无记忆策略），且计算量随历史长度线性/二次膨胀。
- **三大关键机制：**
  1. **Compact Historical Summary & Cached DiT：** 基于 DiT（`MemoryTransformer`）做历史交叉注意力；每段历史轨迹块仅压缩为 1 个冻结 SigLIP2-B/16 MAP 池化视觉 token 与 `h` 个动作 token，图像特征在重叠轨迹训练与推理时一次提取并缓存，推理时延在 **5–600 秒延迟（最多 6,000 帧 @ 10 fps）** 上稳定保持约 **0.16 s**（RTX 5090，8 步 DDIM）。
  2. **History Action Noise (`last_step`)：** 训练时向历史动作注入接近推理最终步级数的微弱噪声 $A_{t-nh:t}^{k-1}$（消融显著优于 Diffusion Forcing、随机噪声及无噪声），防止策略退化为复读历史动作而忽视新观测变化。
  3. **Offline-Calibrated Binary Memory Gate：** 在验证集 $D_{\text{val}}$ 上对比无记忆基线 $\pi$ 误差 $\delta_t$ 与全时记忆策略 $\pi_{\text{mem}}$ 误差 $\delta_t^{\text{mem}}$，取 20 步滑窗与阈值 $\theta=10.0$ 生成二进制伪标签以 BCE 训练 `MemoryGate`；随后冻结门控全量微调门控策略 $\pi_{\text{gated}}$，推理由门控决定开启或完全关闭历史分支（`BinaryGatingSTE`）。

### 实验与评测细节

- **MemMimic（6 项仿真+真机非马尔可夫与跨试次自适应任务）：**
  - 包含 T1 Match Color（仿真）、T2 Discrete Place Back（仿真）、T3' Continuous Place Back（真机 UR5+WSG50）、T3 In-the-Wild Flip & Place Back（真机 ARX X5 + iPhUMI，6 个未见环境 / 10 种杯子 / 20 episodes）、T4 Iterative Pushing（仿真 6 trials）、T5 Iterative Flinging（仿真 5 flings）、T6 Iterative Casting（真机 3 trials）。
  - 整体相对长历史基线平均成功率 **+30.1%**；T1 Match Color 支持 **5–600 s** 跨延迟记忆保持 **`99.0%±1.0%`** SR。
  - **真机 T3 备注（Finding 8.9）：** 真机 ARX X5 长跨度任务直接选用无门控版本（始终开记忆），门控主要收益体现在需要反应控制与防过拟合的混合/高频阶段任务。
- **MIKASA-Robo（5 项标准视觉记忆任务，每任务 250 demos）：** 平均成功率相对 MemoryVLA 提升 **+26.6%**（如 Remember Color / Shape / ShellGame）。
- **RoboMimic 马尔可夫基准（Tool Hang / Square / Transport ph/mh）：** Match Color 与 Iterative Pushing 上门控关闭率分别达 **73%** 与 **58%**，在 Tool Hang 等强反应操作任务上关闭记忆从而避免 120 步长历史基线常见的掉点。

### 关联来源

- 项目页归档：[`sources/sites/gated-memory-policy-github-io.md`](../sites/gated-memory-policy-github-io.md)
- 代码仓归档：[`sources/repos/real-stanford-gated-memory-policy.md`](../repos/real-stanford-gated-memory-policy.md)
- 对应 Wiki 实体：[`wiki/entities/paper-gated-memory-policy.md`](../../wiki/entities/paper-gated-memory-policy.md)

## 当前提炼状态

- [x] `arXiv:2604.18933v2` 全文、项目页与 GitHub 代码仓核对完成
- [x] wiki 实体页：`wiki/entities/paper-gated-memory-policy.md`
