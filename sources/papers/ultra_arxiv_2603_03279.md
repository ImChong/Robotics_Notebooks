# ultra_arxiv_2603_03279

> 来源归档（ingest）

- **标题：** ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation
- **类型：** paper
- **来源：** arXiv:2603.03279（IROS 2026；arXiv 评论含 **Mobile Manipulation Best Paper Finalist**）
- **项目页：** <https://ultra-humanoid.github.io/>
- **代码：** <https://github.com/Sirui-Xu/ULTRA>（**已开源**，入库日 2026-09-29 核查）
- **入库日期：** 2026-09-29
- **一句话说明：** 用 **物理驱动神经重定向** 规模化 OMOMO 类人–物 MoCap → G1 可行轨迹，再 **蒸馏统一多模态 student**：稠密参考跟踪与 **无测试时参考** 的稀疏目标/第一人称深度 goal following 共用一套策略；MuJoCo sim2sim 与 **Unitree G1 真机** 验证。

## 核心论文摘录

### 1) 问题与总贡献（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2603.03279>
- **核心贡献：** 现有 loco-manipulation 多 **依赖预录参考** 或 **固定 conditioning**，难以在 **参考缺失 / 传感降级** 时统一工作。ULTRA 两支柱：(i) **RL 神经重定向**（单策略、数据集级、可物体/轨迹增广）；(ii) **privileged teacher → 多模态 student**（Mask 统一 tokenization + **变分 skill bottleneck** + **RL finetune**），支持 MoCap 状态 / 盲 / **egocentric 点云**。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [ULTRA 实体页](../../wiki/entities/paper-notebook-ultra-unified-multimodal-control-for-autonomous.md)

### 2) 神经重定向（Sec. 4.1）

- **核心贡献：** SMPL-X + 物体轨迹 → **PPO 轨迹优化**（奖励：末端稀疏锚点、链方向、物体 pose/速度、掌–面 offset、接触对齐、能耗）；**heading-aligned** 观测；默认站立初始化 + 平滑权重过渡；**理想高频 PD**（非真机 PD）换吞吐；**各向异性轨迹缩放 + 物体尺度增广** 无需重训策略。
- **对 wiki 的映射：**
  - [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)
  - [InterMimic 实体](../../wiki/entities/paper-bfm-15-intermimic.md)
  - [Domain Randomization](../../wiki/concepts/domain-randomization.md)（重定向阶段刻意无 DR，鲁棒性留给 student）

### 3) Teacher 与多模态 Student（Sec. 4.2–4.3）

- **核心贡献：** Teacher：**4096 env PPO**，全状态 + 稠密参考 + 物体信号。Student：**Transformer prior/encoder–decoder** + **64 维潜变量** + **FiLM**；**availability masking** 统一稠密参考 / 稀疏 root–object 目标 / 点云或 MoCap 物体状态；**DAgger 式在线蒸馏**；跟踪模式可走 **residual shortcut** 绕过随机 latent。RL finetune：在 student 观测下 **closed-loop goal stabilization**，扩大 OOD 交互状态覆盖。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [DAgger](../../wiki/methods/dagger.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md)

### 4) 仿真与真机评测（Sec. 5）

- **核心贡献（Table 1 读点）：** 相对 **HDMI / OmniRetarget** 重实现，ULTRA student **人–物联合跟踪成功率** 明显更高（OOD 物体尺度尤甚）；**纯 student 观测 RL** 远差于 **蒸馏**；统一 all-task 训练略降 ID 跟踪但 **OOD 更稳**。Retargeting **Table 2**：largebox/suitcase 上 **穿透/滑步** 优于 PHC/GMR/OmniRetarget。**Goal following（Table 3，MuJoCo）：** RL finetune 使 OOD 稀疏目标成功率 **点云 +80% / 位置 +200%**（相对未 finetune）。**真机 Table 4：** 稠密参考 **73% (19/26)**；稀疏 MoCap **80–90%**；稀疏 egocentric **50–60%**；失败主因摩擦 slip、深度噪声、扰动超恢复裕度。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Unitree G1](../../wiki/entities/unitree-g1.md)

## 参考来源（原始）

- PDF：<https://arxiv.org/pdf/2603.03279>
- 项目页归档：[ultra-humanoid-github-io.md](../sites/ultra-humanoid-github-io.md)
- 代码归档：[ultra-humanoid.md](../repos/ultra-humanoid.md)
