# Behavior Prompting Policy: Demonstrations as Prompts for Manipulation

> 来源归档（论文全文通读与核对 · `arXiv:2606.30457v2`）

- **标题：** Behavior Prompting Policy: Demonstrations as Prompts for Manipulation
- **类型：** paper
- **状态：** arXiv 2026（v2，2026-10-05）
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2606.30457>
  - 项目页：<https://behavior-prompting.github.io/>
- **代码：** <https://github.com/real-stanford/behavior_prompting>（策略训练与部署） / <https://github.com/real-stanford/iPhUMI>（手持采集处理）
- **作者：** Austin Patel, Ben Pekarek, Joel Enrique Castro Hernandez, Shuran Song
- **机构：** Stanford University; UC Berkeley
- **入库日期：** 2026-08-18
- **更新日期：** 2026-11-19
- **核对范围：** 通读 `arXiv:2606.30457v2` 论文全文与附录（§I–§VIII & Appendix A–H）、项目页、GitHub 代码仓 `real-stanford/behavior_prompting`（`README.md`、`docs/*.md`、`train_network/` 与 `deployment/` 核心源码）。
- **通读：** 已完成
- **一句话说明：** BPP（arXiv:2606.30457）将单条人类手持或机器人示范编码为测试时上下文 behavior prompt，通过 Prompt Chunk Embedding + Transformer Cross-Attention + 扩散动作解码器实现零梯度任务指定与风格迁移，并指出任务多样性比单任务示范数量更关键。

## 核心论文摘录（全文核对）

### 问题与贡献

- **核心设定：** 将人类一条示教视频与轨迹视为测试时行为提示（behavior prompt），输入统一多任务扩散策略（151M 参数，57M 提示编码器 + 扩散 UNet 解码器），无需微调即可执行新任务、长短结合或用户偏好风格。
- **架构设计：**
  - **Prompt Chunk Embedding：** 对 prompt 按每 $\Delta t$ 步切块，将单帧低频观测（如 1 Hz）和本体感觉与该时间窗口内全分辨率 $\Delta t$ 步动作拼成块，经 `AttentionPoolLatent` 池化为单个 prompt chunk token $p_i$。
  - **Prompt Encoder：** 用 4 层（或配置中 6 层）`TransformerDecoderWithAttn` 让当前观测 query 跨注意力读取已缓存的历史与当前 prompt token（LIBERO-Gen 中额外引入 learned attention sink token 以防注意力分布畸变）。
  - **Action Decoder：** FiLM 条件化 `ConditionalUnet1D` 扩散策略（16 步去噪、约 49 ms 推理，相比 ICRT 的 L1 回归 6.5 ms）。
- **核心发现（Diversity > Demos/Task）：** 在固定总训练示范条数预算下，增加独特任务数比增加单任务示范数更能决定未见任务泛化与防过拟合。

### 实验与评测细节

- **四大评测域：**
  1. `DrawAnything-Sim`（2,000 训练任务 × 5 条示范 + 50 手绘未见测试任务）：未见任务 Chamfer 误差相对 Goal-Image 下降 **80.7%**，相对 ICRT（`5.07±0.22`）下降 **32.3%** 至 **`3.43±0.10`**；在滚动中途擦除画布干扰测试下，BPP 误差 **`5.48±0.15`** 优于 ICRT **`10.47±0.98`**。
  2. `DrawAnything-Real`（ARX 机械臂 + 弹簧笔座；1,000 训练任务：200 iPhUMI × 5 + 800 脚本 × 6；10 评测任务：4 训练 + 6 未见）：未见任务 Chamfer 误差相对 Goal-Image 降低 **47.9%**。
  3. `LIBERO-Gen`（基于 LIBERO-Spatial / Goal 扩展 +164 组合任务与 +311 串联任务，各 10 个未见组合/两步链）：未见 Combination 相对语言策略提升 **+13.2%**，未见 Chain 提升 **+10.7%**（去除第二步独立训练样本时提升 **+20.8%**），并比肩微调版 $\pi_{0.5}$；在原始 LIBERO 4 套基准上 BPP 平均成功率 **`97.48±0.33%`**（Language DP `96.23±0.55%`，$\pi_{0.5}$ `97.15%`）。
  4. `Laundry Folding`（仅 3 种折叠任务，各约 150 条示范 + ~1,000 条纠错示范）：在任务多样性极低时，Language DP 成功率 `96% / 100% / 100%` 优于 BPP `76% / 100% / 60%`（分别出现 `8% / 0% / 40%` 错折）。
- **Prompt 敏感性与消融：**
  - DrawAnything-Sim prompt 质量退化对 Chamfer 误差影响：1.25× 减速 `+5.9%`、抖动（$\sigma=5$ px）`+7.4%`、3 次停顿 `+24.0%`。
  - 去掉 prompt 动作或将 prompt 观测降采样低于 1 Hz 会明显损伤未见泛化。

### 关联来源

- 项目页归档：[`sources/sites/behavior-prompting-github-io.md`](../sites/behavior-prompting-github-io.md)
- 代码仓归档：[`sources/repos/real-stanford-behavior-prompting.md`](../repos/real-stanford-behavior-prompting.md)
- 对应 Wiki 实体：[`wiki/entities/paper-behavior-prompting-policy.md`](../../wiki/entities/paper-behavior-prompting-policy.md)

## 当前提炼状态

- [x] `arXiv:2606.30457v2` 全文、项目页与 GitHub 代码仓核对完成
- [x] wiki 实体页：`wiki/entities/paper-behavior-prompting-policy.md`
